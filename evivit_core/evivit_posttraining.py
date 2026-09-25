"""Shared, auditable utilities for EviViT answer-only post-training.

This module deliberately contains no model loading.  Keeping prompt rendering,
answer-label construction, schedule resolution, and trainable-parameter audits
separate makes it possible to test the Base and EviViT lanes with the exact
same contract before spending GPU time.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


SYSTEM_PROMPT = (
    "Answer the visual question using the image. "
    "Return only the concise final answer."
)

LANGUAGE_LORA_TARGET_PATTERN = (
    r"^model\.language_model\.layers\.\d+\.self_attn\."
    r"(q_proj|k_proj|v_proj|o_proj)$"
)

VARIANTS: dict[str, dict[str, Any]] = {
    # TraceSearch answer-only matched control.  The pixel bounds exactly match
    # the Q1M original-only and multi-turn policy evaluations, so this lane
    # isolates answer supervision from human search-process supervision.
    "base_q1m": {
        "family": "base",
        "max_pixels": 1 * 1024 * 1024,
        "min_pixels": 196 * 1024,
        "global_token_budget": None,
        "fine_token_budget": None,
        "max_regions": None,
    },
    "base_q4m": {
        "family": "base",
        "max_pixels": 4 * 1024 * 1024,
        "min_pixels": 4096,
        "global_token_budget": None,
        "fine_token_budget": None,
        "max_regions": None,
    },
    "base_q16m": {
        "family": "base",
        "max_pixels": 16 * 1024 * 1024,
        "min_pixels": 4096,
        "global_token_budget": None,
        "fine_token_budget": None,
        "max_regions": None,
    },
    # Token-forced control used only when a source image is smaller than the
    # Q16M ceiling.  Standard Qwen preprocessing never upscales such an image,
    # while EviViT intentionally spends a fixed budget by retokenizing global
    # and evidence views.  This control makes the realized VStar token budget
    # comparable instead of treating a max-pixel ceiling as an allocation.
    "base_q16m_force14k": {
        "family": "base",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": None,
        "fine_token_budget": None,
        "max_regions": None,
        "force_token_budget": 14336,
    },
    "evivit_a2": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 1024,
        "fine_token_budget": 3072,
        "max_regions": 5,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
    },
    "evivit_a4": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
    },
    "evivit_vnext_h3_a4_salience": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "salience_adapter_required": True,
        "experimental_only": True,
    },
    # EviViT-vNext H1 matched pair.  H0 and H1 keep the complete frozen A4
    # allocator/token contract; their only inference difference is whether a
    # Bridge trained under the same directional contract may update Global
    # tokens.  They are research candidates and cannot replace frozen v4
    # without passing the preregistered multi-benchmark gate.
    "evivit_vnext_h0_a4_matched_bidir": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "bridge_mode": "bidirectional",
        "experimental_only": True,
    },
    "evivit_vnext_h1_a4_preserve_global": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "bridge_mode": "preserve_global",
        "experimental_only": True,
    },
    "evivit_v9_adaptive_balanced": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "adaptive_density_mass",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "experimental_only": True,
    },
    # EviViT-v9 Stage 2-A.  Stage 1's adaptive decoder failed the strict
    # Full515 gate, so all three size policies return to the frozen v4-A4
    # residual-focus geometry.  Only the source-size-to-total-budget function
    # differs; Global/Fine remains a fixed 40/60 split.
    "evivit_v9_size_saturating": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "continuous_budget": {
            "policy": "saturating_log",
            "reference_pixels": 4 * 1024 * 1024,
            "minimum_total_tokens": 4096,
            "maximum_total_tokens": 8192,
            "global_fraction": 0.4,
            "slope": 0.75,
            "bias": -2.258628190095221,
            "calibration_manifest_sha256": (
                "a04de0675e2a653f0953f8066e23d4ceb"
                "8839222a8d48ce47a797578856fb2a8d"
            ),
        },
        "experimental_only": True,
    },
    # Conditional Stage 2-B candidate, preregistered before Stage 2-A external
    # QA completes.  It reuses the exact Saturating-Log total budget but gives
    # smaller source images a larger Global fraction and larger images a
    # larger Fine fraction.  The offset is calibrated on the 1,144 training
    # image sizes so the mean fraction remains 0.40.  This variant must not be
    # evaluated unless the Stage 2-A Saturating main candidate passes its
    # frozen Full515/VStar gate.
    "evivit_v9_size_saturating_global_adaptive": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "continuous_budget": {
            "policy": "saturating_log",
            "reference_pixels": 4 * 1024 * 1024,
            "minimum_total_tokens": 4096,
            "maximum_total_tokens": 8192,
            "global_fraction": 0.4,
            "slope": 0.75,
            "bias": -2.258628190095221,
            "global_fraction_policy": "calibrated_log_clip",
            "global_fraction_log_slope": -0.04,
            "global_fraction_offset": 0.4601457213536029,
            "minimum_global_fraction": 0.32,
            "maximum_global_fraction": 0.55,
            "calibration_manifest_sha256": (
                "a04de0675e2a653f0953f8066e23d4ceb"
                "8839222a8d48ce47a797578856fb2a8d"
            ),
        },
        "conditional_on_stage2a_pass": True,
        "experimental_only": True,
    },
    "evivit_v9_size_power05": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "continuous_budget": {
            "policy": "power",
            "reference_pixels": 4 * 1024 * 1024,
            "minimum_total_tokens": 4096,
            "maximum_total_tokens": 8192,
            "global_fraction": 0.4,
            "exponent": 0.5,
            "scale": 2286.1060602970356,
            "calibration_manifest_sha256": (
                "a04de0675e2a653f0953f8066e23d4ceb"
                "8839222a8d48ce47a797578856fb2a8d"
            ),
        },
        "experimental_only": True,
    },
    "evivit_v9_size_power075": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "continuous_budget": {
            "policy": "power",
            "reference_pixels": 4 * 1024 * 1024,
            "minimum_total_tokens": 4096,
            "maximum_total_tokens": 8192,
            "global_fraction": 0.4,
            "exponent": 0.75,
            "scale": 1471.870716670089,
            "calibration_manifest_sha256": (
                "a04de0675e2a653f0953f8066e23d4ceb"
                "8839222a8d48ce47a797578856fb2a8d"
            ),
        },
        "experimental_only": True,
    },
    # EviViT-v8 spatial-improvement screen.  Unlike A4, the global stream is
    # kept at the image's native Qwen tokenization whenever the complete image
    # fits in the 5K budget.  Only the remaining budget is spent on evidence
    # rereading.  High-resolution inputs that do not fit fall back to A4.
    "evivit_v8_nativeglobal_bidir": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "native_global_total_token_budget": 5120,
        "native_global_bridge_mode": "bidirectional",
        "fallback_bridge_mode": "bidirectional",
        "experimental_only": True,
    },
    "evivit_v8_nativeglobal_preserve": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "native_global_total_token_budget": 5120,
        "native_global_bridge_mode": "preserve_global",
        "fallback_bridge_mode": "bidirectional",
        "experimental_only": True,
    },
    # EviViT-v8 topology-preserving injection. High-resolution evidence is
    # encoded and sparsely written into its native global parents at Block16,
    # but fine tokens are not appended to the LLM visual span. Consequently
    # the emitted token count/order/MRoPE topology exactly match native Qwen.
    "evivit_v8_topologyinject_f1": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 1024,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "native_global_total_token_budget": 5120,
        "native_global_bridge_mode": "bidirectional",
        "fallback_bridge_mode": "bidirectional",
        "visual_output_mode": "global_only",
        "experimental_only": True,
    },
    "evivit_v8_topologyinject_f3": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "native_global_total_token_budget": 5120,
        "native_global_bridge_mode": "bidirectional",
        "fallback_bridge_mode": "bidirectional",
        "visual_output_mode": "global_only",
        "experimental_only": True,
    },
    # Same topology-preserving F3 injection, but high-resolution images that
    # exceed the native 5K fit threshold retain a 4K global fallback instead
    # of A4's 2K global fallback. CV-Bench is unchanged because every image
    # already fits natively; this arm isolates the fine-grained fallback.
    "evivit_v8_topologyinject_g4f3": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 4096,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "native_global_total_token_budget": 5120,
        "native_global_bridge_mode": "bidirectional",
        "fallback_bridge_mode": "bidirectional",
        "visual_output_mode": "global_only",
        "experimental_only": True,
    },
    "evivit_v5_a4_global_anchor_relay": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "relay_required": True,
        "relay_dim": 256,
        "relay_heads": 4,
        "relay_anchor_grid": [4, 4],
        "relay_max_relative_residual": 0.1,
        "minimum_anchor_attention_mass": 0.0,
        "experimental_only": True,
    },
    "evivit_v5_a4_global_anchor_relay_floor25": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "relay_required": True,
        "relay_dim": 256,
        "relay_heads": 4,
        "relay_anchor_grid": [4, 4],
        "relay_max_relative_residual": 0.1,
        "minimum_anchor_attention_mass": 0.25,
        "experimental_only": True,
    },
    # Capacity-matched control for Base Q16M.  Its nominal 14,336-token budget
    # is close to Q16M's measured 14,568-token Full515 mean.  This is an
    # evaluation-gated control, not a replacement for the frozen A4 mainline.
    "evivit_g8f6_r3": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 8192,
        "fine_token_budget": 6144,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
    },
    # EviViT-v5 frozen spatial-repair screen.  These three variants change one
    # allocator variable at a time relative to A4 (G2F3-R3).  They are
    # preregistered ablations and do not replace the frozen mainline.
    "evivit_v5_g3f2_r3": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 3072,
        "fine_token_budget": 2048,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
    },
    "evivit_v5_g3f3_r3": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 3072,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
    },
    "evivit_v5_g2f3_r3_expand115": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.15,
        "topology_union_factor": 1.0,
    },
    # Mechanism-only VStar diagnostic.  It preserves A4's 2K global path but
    # removes every fine view, asking whether irrelevant high-resolution
    # evidence distracts relative-position answers.  It is not eligible for
    # model selection or a paper primary result.
    "evivit_v5_g2f0_global_only_diag": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 0,
        "max_regions": 0,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "diagnostic_only": True,
    },
    # Mechanism-only selector-text control.  The answer policy still receives
    # the complete multiple-choice prompt; only PTEA receives the question
    # stem, matching the question-only supervision used to train the map.
    "evivit_v5_a4_question_stem_diag": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "selector_question_protocol": "question_stem",
        "diagnostic_only": True,
    },
    # Directional bridge controls keep the exact A4 image views and token
    # budget while masking one or both learned residual directions.  They
    # distinguish fine-token distraction from destructive Fine->Global
    # write-back before a new relay is allowed.
    "evivit_v5_a4_preserve_global_diag": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "bridge_mode": "preserve_global",
        "diagnostic_only": True,
    },
    "evivit_v5_a4_preserve_fine_diag": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "bridge_mode": "preserve_fine",
        "diagnostic_only": True,
    },
    "evivit_v5_a4_identity_bridge_diag": {
        "family": "evivit",
        "max_pixels": 16 * 1024 * 1024,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "evidence_decoder": "residual_focus",
        "region_expansion_factor": 1.0,
        "topology_union_factor": 1.0,
        "bridge_mode": "identity",
        "diagnostic_only": True,
    },
}

ALLOWED_QA_FIELDS = {"id", "image", "question", "answer", "source_dataset"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_qa_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            extra = set(row) - ALLOWED_QA_FIELDS
            missing = {"id", "image", "question", "answer"} - set(row)
            if extra or missing:
                raise ValueError(
                    f"{path}:{line_number}: forbidden={sorted(extra)}, "
                    f"missing={sorted(missing)}"
                )
            row_id = str(row["id"])
            if row_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {row_id}")
            if not str(row["question"]).strip() or not str(row["answer"]).strip():
                raise ValueError(f"{path}:{line_number}: empty question or answer")
            seen.add(row_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"empty QA manifest: {path}")
    return rows


def clean_question(value: str) -> str:
    return " ".join(str(value).replace("<image>", " ").split())


def answer_messages(question: str, answer: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": clean_question(question)},
            ],
        },
    ]
    if answer is not None:
        target = str(answer).strip()
        if not target:
            raise ValueError("answer must be non-empty")
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": target}]}
        )
    return messages


def render_answer_texts(processor: Any, question: str, answer: str) -> tuple[str, str, tuple[int, int]]:
    """Render one shared Base/EviViT prompt and locate only the answer text.

    The assistant end marker is intentionally outside the supervised character
    span.  This prevents formatting tokens from dominating short QA targets.
    """

    target = str(answer).strip()
    prompt = processor.apply_chat_template(
        answer_messages(question), tokenize=False, add_generation_prompt=True
    )
    full = processor.apply_chat_template(
        answer_messages(question, target), tokenize=False, add_generation_prompt=False
    )
    if not full.startswith(prompt):
        raise ValueError("assistant target is not a strict prompt continuation")
    if not full[len(prompt) :].startswith(target):
        raise ValueError("rendered assistant text does not start with the target answer")
    return prompt, full, (len(prompt), len(prompt) + len(target))


def labels_from_offsets(
    token_ids: Sequence[int],
    offsets: Sequence[Sequence[int]],
    answer_span: tuple[int, int],
) -> list[int]:
    """Mask every raw chat token except tokens overlapping the answer text."""

    if len(token_ids) != len(offsets):
        raise ValueError("token_ids and offsets must have equal length")
    start, end = answer_span
    if not 0 <= start < end:
        raise ValueError("invalid answer character span")
    labels = [
        int(token_id) if max(int(left), start) < min(int(right), end) else -100
        for token_id, (left, right) in zip(token_ids, offsets)
    ]
    if all(value == -100 for value in labels):
        raise ValueError("answer span produced no supervised token")
    return labels


def expand_single_image_labels(
    raw_ids: Sequence[int],
    raw_labels: Sequence[int],
    expanded_ids: Sequence[int],
    image_token_id: int,
) -> tuple[list[int], int, int]:
    """Expand one raw image placeholder and prove tokenization parity.

    Qwen's processor replaces one image placeholder with one token per merged
    visual patch.  This function mirrors that replacement for labels and
    rejects any processor/template mismatch instead of silently shifting the
    answer target.
    """

    if len(raw_ids) != len(raw_labels):
        raise ValueError("raw ids and labels must have equal length")
    positions = [index for index, value in enumerate(raw_ids) if value == image_token_id]
    if len(positions) != 1:
        raise ValueError(f"expected one raw image placeholder, found {len(positions)}")
    visual_tokens = len(expanded_ids) - len(raw_ids) + 1
    if visual_tokens <= 0:
        raise ValueError("expanded sequence contains no visual span")
    visual_start = positions[0]
    visual_end = visual_start + visual_tokens
    expected_ids = (
        list(raw_ids[:visual_start])
        + [int(image_token_id)] * visual_tokens
        + list(raw_ids[visual_start + 1 :])
    )
    if expected_ids != list(expanded_ids):
        raise ValueError("processor ids do not match one-placeholder image expansion")
    labels = (
        list(raw_labels[:visual_start])
        + [-100] * visual_tokens
        + list(raw_labels[visual_start + 1 :])
    )
    return labels, visual_start, visual_end


def micro_batches_in_optimizer_step(
    *,
    micro_step_before: int,
    target_micro_steps: int,
    gradient_accumulation_steps: int,
    selected_rows: int,
) -> int:
    remaining_total = target_micro_steps - micro_step_before
    if remaining_total <= 0:
        raise ValueError("micro_step_before must be smaller than target_micro_steps")
    if selected_rows <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("selected_rows and gradient accumulation must be positive")
    offset = micro_step_before % selected_rows
    remaining_epoch = selected_rows - offset if offset else selected_rows
    return min(gradient_accumulation_steps, remaining_total, remaining_epoch)


def resolve_training_schedule(
    *, selected_rows: int, num_epochs: int, gradient_accumulation_steps: int
) -> tuple[int, int]:
    if selected_rows <= 0 or num_epochs <= 0 or gradient_accumulation_steps <= 0:
        raise ValueError("rows, epochs, and gradient accumulation must be positive")
    target_micro_steps = selected_rows * num_epochs
    optimizer_steps = 0
    cursor = 0
    while cursor < target_micro_steps:
        cursor += micro_batches_in_optimizer_step(
            micro_step_before=cursor,
            target_micro_steps=target_micro_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            selected_rows=selected_rows,
        )
        optimizer_steps += 1
    return target_micro_steps, optimizer_steps


def warmup_steps(total_optimizer_steps: int, warmup_ratio: float) -> int:
    if total_optimizer_steps <= 0 or not 0 <= warmup_ratio < 1:
        raise ValueError("invalid scheduler configuration")
    return math.floor(total_optimizer_steps * warmup_ratio)


def parameter_report(model: Any) -> dict[str, Any]:
    total = 0
    trainable = 0
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_ratio": trainable / total if total else 0.0,
        "trainable_tensor_count": len(trainable_names),
        "trainable_names": trainable_names,
    }


def assert_language_lora_only(report: dict[str, Any]) -> None:
    names = list(report.get("trainable_names", []))
    if not names or int(report.get("trainable_parameters", 0)) <= 0:
        raise ValueError("no trainable LoRA parameters")
    forbidden = [
        name
        for name in names
        if "language_model.layers." not in name
        or not any(f".{projection}." in name for projection in ("q_proj", "k_proj", "v_proj", "o_proj"))
        or "lora_" not in name
    ]
    if forbidden:
        raise ValueError(f"non-language-LoRA parameters are trainable: {forbidden[:8]}")
    visual = [name for name in names if ".visual." in name or "vision_model" in name]
    if visual:
        raise ValueError(f"visual parameters are trainable: {visual[:8]}")
