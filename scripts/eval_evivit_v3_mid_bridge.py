#!/usr/bin/env python3
"""Evaluate EviViT-v3 with online Block-16 Mid-PTEA and a frozen bridge.

The evaluator deliberately mirrors ``train_evivit_v3_bridge.py``.  The global
image is encoded once to the frozen insertion block, Mid-PTEA predicts a
question-conditioned map from that state, high-resolution regions are sampled
from the original image, and global/fine tokens exchange sparse messages before
the remaining native Qwen-ViT blocks.  The Qwen LLM and all Qwen-ViT weights stay
frozen.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.eval_evivit_unified_tokens import (  # noqa: E402
    build_position_ids,
    greedy_generate_from_visual_span,
    greedy_generate_from_visual_spans,
)
from scripts.eval_qwen_original_qa import (  # noqa: E402
    normalized_answer,
    parse_answer,
    parse_muir_choice,
    parse_blink_choice,
    qa_messages,
    relaxed_answer,
    relaxed_answer_match,
    summarize,
)
from scripts.extract_qwen_dense_features import build_projection  # noqa: E402
from scripts.predict_patch_text_topk_boxes import load_selector  # noqa: E402
from scripts.train_evivit_v3_bridge import (  # noqa: E402
    build_chat_text,
    build_multi_chat_text,
    native_global_visual_coordinates,
    read_jsonl,
    resize_to_token_grid,
    visual_coordinates,
)
from evivit_core.evivit_posttraining import expand_single_image_labels  # noqa: E402
from evivit_core.evivit_recovery_fastdev import (  # noqa: E402
    recovery_row_metadata,
    validate_language_lora_adapter,
)
from evivit_core.evivit import RegionBudget, grid_for_token_budget  # noqa: E402
from evivit_core.evivit_evidence_bridge import SparseGlobalLocalEvidenceBridge  # noqa: E402
from evivit_core.evivit_adaptive_box import (  # noqa: E402
    apply_residual,
    load_adaptive_box_head,
    residual_ensemble_uncertainty,
    union_boxes,
)
from evivit_core.evivit_adaptive_box_features import (  # noqa: E402
    adaptive_box_feature_vectors,
)
from evivit_core.evivit_trace_refine import (  # noqa: E402
    load_trace_refine_policy,
    select_counterfactual_portfolio,
)
from evivit_core.evivit_trace_calibrated_bridge import (  # noqa: E402
    TraceCalibratedBridgeResidual,
)
from evivit_core.evislot import EvidenceSlotCompressor  # noqa: E402
from evivit_core.evisplit_context_need import load_context_need_head  # noqa: E402
from evivit_core.eviblend import SafeJointResidualMixer  # noqa: E402
from evivit_core.evirelay import EviRelayBlock, EviRelayBridgeAdapter  # noqa: E402
from evivit_core.evitree import (  # noqa: E402
    continuous_fine_token_budget,
    continuous_native_global_token_budget,
    load_hierarchical_split_policy_head,
    load_scalar_evidence_need_head,
    plan_native_need_spread_budget,
    plan_native_ratio_budget,
    scalar_need_feature_vector,
)
from evivit_core.evivit_v3_online import (  # noqa: E402
    allocate_control_evidence,
    allocate_evitree_hierarchical_evidence,
    allocate_mid_ptea_evidence,
    allocate_reliability_calibrated_evidence,
    qwen_projected_question_tokens,
    reallocate_probability_evidence,
)
from evivit_core.evivit_v10_reliability import ReliabilityGate  # noqa: E402
from evivit_core.evivit_native_scale import (  # noqa: E402
    plan_continuous_native_global_view,
    plan_native_pixel_view,
)
from evivit_core.evivit_patch_exchange import (  # noqa: E402
    continuous_soft_floor_tokens,
    plan_patch_exchange_fine_views,
    plan_patch_exchange_global_view,
)
from evivit_core.stage_d_v27b_safe import plan_stage_d_v27b_safe_caps  # noqa: E402
from evivit_core.geometry import policy_to_pixels  # noqa: E402
from evivit_core.gpu_safety import check_gpu  # noqa: E402
from evivit_core.qwen_family import (  # noqa: E402
    language_model_visual_forward,
    load_qwen_vlm,
    vision_probe_text,
)
from evivit_core.latency import summarize_latency  # noqa: E402
from evivit_core.qwen3vl_evivit_v3_mid_encoder import (  # noqa: E402
    MidViTFineViewInput,
    Qwen3VLEviViTV3MidEncoder,
)
from evivit_core.qwen3vl_evivit import qwen_grid_centers_in_original  # noqa: E402
from evivit_core.qwen3vl_evislot_encoder import Qwen3VLEviSlotEncoder  # noqa: E402
from evivit_core.simple_evivit import (  # noqa: E402
    plan_anchor_fine_floors,
    plan_continuous_fine_floor,
    plan_native_token_cap,
)
from evivit_core.tracefovea import (  # noqa: E402
    TraceFoveaBridgeAdapter,
    TraceFoveationBlock,
)
from evivit_core.tracescale import (  # noqa: E402
    load_tracescale_head,
    predict_region_geometry,
)
from evivit_core.anchored_evidence_growth import (  # noqa: E402
    load_anchored_growth_calibrator,
)
from evivit_core.evicontour_utility import load_evicontour_utility_head  # noqa: E402


def deterministic_deranged_question_controls(
    rows: list[dict[str, Any]], *, seed: int
) -> dict[str, dict[str, str]]:
    """Assign every sample another sample's selector question deterministically.

    The answer prompt remains untouched. This control therefore removes only
    the question--evidence alignment seen by PTEA while preserving the image,
    token budget, region decoder, bridge, and downstream answer question.
    """

    if len(rows) < 2:
        raise ValueError("question-shuffled PTEA control requires at least two rows")
    identifiers = [str(row["id"]) for row in rows]
    questions = [str(row.get("selector_question") or row["question"]) for row in rows]
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(rows))
    for _ in range(256):
        rng.shuffle(permutation)
        if np.all(permutation != np.arange(len(rows))):
            break
    else:
        shift = 1 + int(seed % (len(rows) - 1))
        permutation = np.roll(np.arange(len(rows)), shift)
    return {
        identifiers[index]: {
            "question": questions[int(source_index)],
            "source_id": identifiers[int(source_index)],
        }
        for index, source_index in enumerate(permutation)
    }


def relocate_region_budgets(
    budgets: list[RegionBudget],
    *,
    mode: str,
    row_id: str,
    seed: int,
) -> tuple[list[RegionBudget], list[dict[str, Any]]]:
    """Relocate final Fine boxes without changing shape or token budget.

    The control runs after learned geometry refinements and before PatchExchange,
    so region count, width/height, role, score and requested tokens stay fixed.
    Only the spatial location changes.
    """

    if mode == "matched":
        return list(budgets), []
    if mode not in {"random_relocate", "spread_relocate"}:
        raise ValueError(f"unknown region-location control: {mode}")
    digest = hashlib.sha256(f"{seed}:{row_id}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    base_anchors = [(0.18, 0.18), (0.82, 0.82), (0.82, 0.18), (0.18, 0.82)]
    rotation = int(rng.integers(0, len(base_anchors)))
    anchors = base_anchors[rotation:] + base_anchors[:rotation]
    relocated: list[RegionBudget] = []
    plans: list[dict[str, Any]] = []

    def overlap_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if intersection == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return intersection / max(area_a + area_b - intersection, 1)

    for index, budget in enumerate(budgets):
        width = int(budget.bbox[2] - budget.bbox[0])
        height = int(budget.bbox[3] - budget.bbox[1])
        if mode == "spread_relocate":
            center_x, center_y = anchors[index % len(anchors)]
            x1 = int(round(center_x * 1000.0 - width / 2.0))
            y1 = int(round(center_y * 1000.0 - height / 2.0))
            x1 = min(max(x1, 0), 1000 - width)
            y1 = min(max(y1, 0), 1000 - height)
            candidate = (x1, y1, x1 + width, y1 + height)
        else:
            candidates: list[tuple[float, tuple[int, int, int, int]]] = []
            for _ in range(64):
                x1 = int(rng.integers(0, max(1000 - width, 0) + 1))
                y1 = int(rng.integers(0, max(1000 - height, 0) + 1))
                proposal = (x1, y1, x1 + width, y1 + height)
                maximum_overlap = max(
                    (overlap_iou(proposal, item.bbox) for item in relocated),
                    default=0.0,
                )
                candidates.append((maximum_overlap, proposal))
                if maximum_overlap <= 0.15:
                    break
            candidate = min(candidates, key=lambda item: item[0])[1]
        relocated.append(
            RegionBudget(
                bbox=candidate,
                token_budget=budget.token_budget,
                score=budget.score,
                role=budget.role,
                source_rank=budget.source_rank,
            )
        )
        plans.append(
            {
                "original_bbox": list(budget.bbox),
                "controlled_bbox": list(candidate),
                "width": width,
                "height": height,
                "token_budget": int(budget.token_budget),
                "role": budget.role,
            }
        )
    return relocated, plans


def configure_deterministic_inference(attention_backend: str = "math") -> None:
    """Lock one SDPA backend under PyTorch's deterministic-algorithm guard."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if attention_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    elif attention_backend == "mem_efficient":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    elif attention_backend == "flash":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
    else:
        raise ValueError(f"unsupported deterministic SDPA backend: {attention_backend}")


def deterministic_backend_name(
    attention_backend: str, *, ptea_auxiliary_attention: bool = True
) -> str:
    if attention_backend == "flash" and ptea_auxiliary_attention:
        return "sdpa_flash_backbone_math_ptea_deterministic"
    return f"sdpa_{attention_backend}_deterministic"


def resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _forced_policy_box(value: Any, *, field: str, row_id: str) -> tuple[int, int, int, int]:
    """Validate a manifest-supplied box in the shared 0--1000 policy frame."""

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{row_id}: {field} must contain four coordinates")
    values = tuple(int(round(float(item))) for item in value)
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise ValueError(f"{row_id}: invalid {field} policy box {values}")
    return values


def replace_trace_split_context_box(
    allocation: Any,
    bbox: tuple[int, int, int, int],
) -> bool:
    """Replace only trace_split's context anchor while preserving its budget.

    Candidate utility experiments use this hook before H-Safe, PatchExchange,
    native rereading and Sparse Bridge.  Thus every candidate is evaluated by
    the deployed EviViT path while the two decisive regions, token budget and
    all downstream modules remain matched.
    """

    budgets = list(allocation.region_budgets)
    if len(budgets) > 3:
        raise ValueError(
            "forced context evaluation supports at most three trace_split regions"
        )
    context_indexes = [
        index for index, budget in enumerate(budgets) if "context" in budget.role
    ]
    if len(context_indexes) > 1:
        raise ValueError(f"ambiguous trace_split context roles: {context_indexes}")
    # Redundancy suppression can leave one decisive region plus context.
    # Find the existing context by role, not by its ordinal position. Never
    # add a region if the deployed allocation actually omitted context.
    if not context_indexes and len(budgets) < 3:
        return False
    index = context_indexes[0] if context_indexes else 2
    previous = budgets[index]
    budgets[index] = RegionBudget(
        bbox=bbox,
        token_budget=previous.token_budget,
        score=previous.score,
        role=previous.role,
        source_rank=previous.source_rank,
    )
    allocation.region_budgets = budgets
    return True


def reference_answer_nll_from_visual_span(
    model: Any,
    processor: Any,
    *,
    question: str,
    answer: str,
    image_embeds: torch.Tensor,
    deepstack_embeds: list[torch.Tensor],
    visual_coordinates_array: np.ndarray,
    device: str,
) -> tuple[float, float, int, str]:
    """Score only GT answer-value tokens under the formal compact-JSON prompt."""

    target_json = json.dumps(
        {"answer": str(answer).strip()},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    rendered_value = json.dumps(str(answer).strip(), ensure_ascii=False)
    full_text = build_chat_text(
        processor,
        question,
        answer,
        answer_protocol="compact_json",
    )
    target_start = full_text.rfind(target_json)
    value_offset = target_json.find(rendered_value)
    if target_start < 0 or value_offset < 0:
        raise RuntimeError("could not locate compact-JSON answer value")
    value_start = target_start + value_offset
    value_end = value_start + len(rendered_value)
    if rendered_value.startswith('"') and rendered_value.endswith('"'):
        value_start += 1
        value_end -= 1
    encoded = processor.tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    raw_ids = [int(value) for value in encoded.input_ids]
    raw_labels = [
        int(token_id)
        if max(int(left), value_start) < min(int(right), value_end)
        else -100
        for token_id, (left, right) in zip(raw_ids, encoded.offset_mapping)
    ]
    if all(value == -100 for value in raw_labels):
        raise RuntimeError("reference answer produced no supervised value token")
    image_token_id = int(model.config.image_token_id)
    positions = [
        index for index, value in enumerate(raw_ids) if value == image_token_id
    ]
    if len(positions) != 1:
        raise RuntimeError(f"expected one image placeholder, found {len(positions)}")
    visual_start = positions[0]
    visual_tokens = int(image_embeds.shape[0])
    expanded_ids = (
        raw_ids[:visual_start]
        + [image_token_id] * visual_tokens
        + raw_ids[visual_start + 1 :]
    )
    labels_list, checked_start, visual_end = expand_single_image_labels(
        raw_ids,
        raw_labels,
        expanded_ids,
        image_token_id,
    )
    if checked_start != visual_start:
        raise RuntimeError("visual start changed while expanding reference labels")
    input_ids = torch.tensor([expanded_ids], dtype=torch.long, device=device)
    labels = torch.tensor([labels_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    visual_mask = input_ids.eq(image_token_id)
    inputs_embeds = inputs_embeds.masked_scatter(
        visual_mask.unsqueeze(-1), image_embeds.to(inputs_embeds.dtype)
    )
    position_ids, _ = build_position_ids(
        input_ids.shape[1],
        visual_start,
        visual_end,
        visual_coordinates_array,
        device=device,
    )
    outputs = language_model_visual_forward(
        model,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_mask=visual_mask,
        deepstack_visual_embeds=[
            value.to(inputs_embeds.dtype) for value in deepstack_embeds
        ],
        cache_position=torch.arange(input_ids.shape[1], device=device),
        use_cache=False,
    )
    shifted_targets = labels[:, 1:]
    mask = shifted_targets.ne(-100)
    answer_tokens = int(mask.sum().item())
    if answer_tokens <= 0:
        raise RuntimeError("reference answer contains no causal target token")
    selected_hidden = outputs.last_hidden_state[:, :-1][mask]
    targets = shifted_targets[mask]
    logits = model.lm_head(selected_hidden)
    loss = F.cross_entropy(logits.float(), targets)
    accuracy = logits.argmax(dim=-1).eq(targets).float().mean()
    return (
        float(loss.detach().cpu()),
        float(accuracy.detach().cpu()),
        answer_tokens,
        hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
    )


def bridge_from_checkpoint(
    checkpoint_path: Path | None,
    *,
    model: Any,
    device: str,
    default_insertion_block: int,
    default_bridge_dim: int,
    default_bridge_heads: int,
    default_neighborhood_radius: int,
    default_max_relative_residual: float | None,
    coverage_masked_parent_writeback: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Construct the exact bridge topology saved by training, or a zero control."""

    checkpoint: dict[str, Any] | None = None
    config: dict[str, Any] = {}
    checkpoint_version: str | None = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_version = str(checkpoint.get("version"))
        if checkpoint_version not in {
            "evivit_v3_sparse_mid_bridge_v1",
            "tracefovea_fixed5k_v1",
            "tracefovea_human_gate_fixed5k_v1",
            "tracefovea_trace_residual_fixed5k_v1",
            "eviweave_human_gate_fixed5k_v1",
            "eviweave_normalized_fixed5k_v2",
            "evirelay_fixed5k_v1",
            "eviblend_safe_joint_fixed5k_v1",
            "evislot_midvit_fixedslots_v1",
            "trace_calibrated_bridge_fixed5k_v1",
            "trace_calibrated_bridge_centered_fixed5k_v2",
            "trace_calibrated_bridge_centered_fp32importance_fixed5k_v3",
            "evitree_v2_parent_fusion_v1",
            "evitree_v2_parent_fusion_human_gate_v1",
        }:
            raise RuntimeError(
                "not an EviViT-v3/TraceFovea/EviRelay fusion checkpoint: "
                f"{checkpoint_version}"
            )
        config = dict(checkpoint.get("config", {}))

    def configured(name: str, default: Any) -> Any:
        value = config.get(name, default)
        return default if value is None else value

    residual_value = configured(
        "max_relative_residual", default_max_relative_residual
    )
    residual_bound = (
        float(residual_value)
        if residual_value is not None and float(residual_value) > 0
        else None
    )
    if checkpoint_version == "evislot_midvit_fixedslots_v1":
        bridge = EvidenceSlotCompressor(
            int(model.config.vision_config.hidden_size),
            slot_dim=int(configured("evislot_dim", 256)),
            heads=int(configured("evislot_heads", 4)),
            slots_per_region=int(
                configured("evislot_slots_per_region", 4)
            ),
            spatial_merge_size=int(
                model.config.vision_config.spatial_merge_size
            ),
            maximum_residual_scale=float(
                configured("evislot_maximum_residual_scale", 0.25)
            ),
            evidence_bias=float(configured("evislot_evidence_bias", 0.5)),
            enable_spatial_anchor_bias=bool(
                configured("evislot_spatial_anchor_bias", False)
            ),
            spatial_anchor_strength=float(
                configured("evislot_spatial_anchor_strength", 2.0)
            ),
            spatial_anchor_sigma=float(
                configured("evislot_spatial_anchor_sigma", 0.30)
            ),
        ).to(device=device, dtype=torch.bfloat16)
    elif checkpoint_version in {
        "tracefovea_fixed5k_v1",
        "tracefovea_human_gate_fixed5k_v1",
        "tracefovea_trace_residual_fixed5k_v1",
        "eviweave_human_gate_fixed5k_v1",
        "eviweave_normalized_fixed5k_v2",
        "evitree_v2_parent_fusion_v1",
        "evitree_v2_parent_fusion_human_gate_v1",
    }:
        if residual_bound is None:
            raise RuntimeError("TraceFovea checkpoint requires a residual bound")
        bridge: torch.nn.Module = TraceFoveaBridgeAdapter(
            TraceFoveationBlock(
                int(model.config.vision_config.hidden_size),
                fovea_dim=int(configured("fovea_dim", 256)),
                spatial_merge_size=int(model.config.vision_config.spatial_merge_size),
                max_relative_residual=float(residual_bound),
                use_child_gate=(
                    checkpoint_version
                    in {
                        "tracefovea_human_gate_fixed5k_v1",
                        "tracefovea_trace_residual_fixed5k_v1",
                        "eviweave_human_gate_fixed5k_v1",
                        "eviweave_normalized_fixed5k_v2",
                        "evitree_v2_parent_fusion_human_gate_v1",
                    }
                ),
                modulate_child_residual=(
                    checkpoint_version
                    in {
                        "tracefovea_trace_residual_fixed5k_v1",
                        "eviweave_normalized_fixed5k_v2",
                    }
                ),
                normalized_child_routing=(
                    checkpoint_version
                    == "eviweave_normalized_fixed5k_v2"
                ),
                preserve_float32_importance=(
                    checkpoint_version
                    == "eviweave_normalized_fixed5k_v2"
                ),
                minimum_child_importance=float(
                    configured("minimum_child_importance", 0.25)
                ),
                maximum_child_importance=float(
                    configured("maximum_child_importance", 3.0)
                ),
                coverage_masked_parent_writeback=(
                    coverage_masked_parent_writeback
                ),
            )
        ).to(device=device, dtype=torch.bfloat16)
    elif checkpoint_version == "evirelay_fixed5k_v1":
        if residual_bound is None:
            raise RuntimeError("EviRelay checkpoint requires a residual bound")
        bridge = EviRelayBridgeAdapter(
            EviRelayBlock(
                int(model.config.vision_config.hidden_size),
                relay_dim=int(configured("relay_dim", 256)),
                relay_tokens=int(configured("relay_tokens", 8)),
                max_relative_residual=float(residual_bound),
            )
        ).to(device=device, dtype=torch.bfloat16)
    else:
        bridge = SparseGlobalLocalEvidenceBridge(
            int(model.config.vision_config.hidden_size),
            bridge_dim=int(configured("bridge_dim", default_bridge_dim)),
            heads=int(configured("bridge_heads", default_bridge_heads)),
            neighborhood_radius=int(
                configured("neighborhood_radius", default_neighborhood_radius)
            ),
            spatial_merge_size=int(model.config.vision_config.spatial_merge_size),
            max_relative_residual=residual_bound,
        ).to(device=device, dtype=torch.bfloat16)
    if checkpoint is not None:
        bridge.load_state_dict(checkpoint["bridge"], strict=True)
    bridge.eval()
    safe_joint_residual_mixer = None
    if checkpoint_version == "eviblend_safe_joint_fixed5k_v1":
        safe_joint_residual_mixer = SafeJointResidualMixer(
            int(model.config.vision_config.hidden_size),
            blend_dim=int(configured("safe_joint_blend_dim", 256)),
            maximum_gate=float(
                configured("safe_joint_maximum_gate", 0.50)
            ),
            minimum_importance=float(
                configured("safe_joint_minimum_importance", 0.25)
            ),
            maximum_importance=float(
                configured("safe_joint_maximum_importance", 3.0)
            ),
        ).to(device=device, dtype=torch.bfloat16)
        safe_joint_residual_mixer.load_state_dict(
            checkpoint["safe_joint_residual_mixer"],
            strict=True,
        )
        safe_joint_residual_mixer.eval()
    trace_calibrated_bridge_residual = None
    if checkpoint_version in {
        "trace_calibrated_bridge_fixed5k_v1",
        "trace_calibrated_bridge_centered_fixed5k_v2",
        "trace_calibrated_bridge_centered_fp32importance_fixed5k_v3",
    }:
        trace_calibrated_bridge_residual = TraceCalibratedBridgeResidual(
            int(model.config.vision_config.hidden_size),
            calibration_dim=int(
                configured("trace_calibration_dim", 256)
            ),
            maximum_calibration=float(
                configured("trace_calibration_maximum", 0.50)
            ),
            minimum_importance=float(
                configured(
                    "trace_calibration_minimum_importance", 0.25
                )
            ),
            maximum_importance=float(
                configured(
                    "trace_calibration_maximum_importance", 3.0
                )
            ),
            amplitude_mode=str(
                configured(
                    "trace_calibration_amplitude_mode",
                    (
                        "positive_centered"
                        if checkpoint_version
                        in {
                            "trace_calibrated_bridge_centered_fixed5k_v2",
                            (
                                "trace_calibrated_bridge_centered_"
                                "fp32importance_fixed5k_v3"
                            ),
                        }
                        else "signed_zero"
                    ),
                )
            ),
            preserve_float32_importance=bool(
                configured(
                    "trace_calibration_preserve_fp32_importance",
                    checkpoint_version
                    == (
                        "trace_calibrated_bridge_centered_"
                        "fp32importance_fixed5k_v3"
                    ),
                )
            ),
        ).to(device=device, dtype=torch.bfloat16)
        trace_calibrated_bridge_residual.load_state_dict(
            checkpoint["trace_calibrated_bridge_residual"],
            strict=True,
        )
        trace_calibrated_bridge_residual.eval()
    metadata = {
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_step": int(checkpoint.get("step", 0)) if checkpoint else 0,
        "checkpoint_version": checkpoint_version,
        "fusion": (
            {
                "tracefovea_fixed5k_v1": "tracefovea",
                "tracefovea_human_gate_fixed5k_v1": (
                    "tracefovea_human_gate"
                ),
                "tracefovea_trace_residual_fixed5k_v1": (
                    "tracefovea_trace_residual"
                ),
                "eviweave_human_gate_fixed5k_v1": "eviweave_human_gate",
                "eviweave_normalized_fixed5k_v2": (
                    "eviweave_normalized"
                ),
                "evitree_v2_parent_fusion_v1": "evitree_v2_parent_fusion",
                "evitree_v2_parent_fusion_human_gate_v1": (
                    "evitree_v2_parent_fusion_human_gate"
                ),
                "evirelay_fixed5k_v1": "evirelay",
                "eviblend_safe_joint_fixed5k_v1": "eviblend_safe_joint",
                "evislot_midvit_fixedslots_v1": "evislot",
                "trace_calibrated_bridge_fixed5k_v1": (
                    "trace_calibrated_bridge"
                ),
                "trace_calibrated_bridge_centered_fixed5k_v2": (
                    "trace_calibrated_bridge"
                ),
                "trace_calibrated_bridge_centered_fp32importance_fixed5k_v3": (
                    "trace_calibrated_bridge"
                ),
            }.get(checkpoint_version, "sparse_bridge")
        ),
        "checkpoint_config": config,
        "insertion_block": int(
            configured("insertion_block", default_insertion_block)
        ),
        "bridge_parameters": sum(parameter.numel() for parameter in bridge.parameters()),
        "zero_bridge": checkpoint is None,
        "safe_joint_residual_mixer": safe_joint_residual_mixer,
        "trace_calibrated_bridge_residual": (
            trace_calibrated_bridge_residual
        ),
    }
    return bridge, metadata


def collect_fusion_diagnostics(
    bridge: torch.nn.Module,
    encoder: Qwen3VLEviViTV3MidEncoder,
    vision: Any,
) -> dict[str, Any]:
    """Return compact architecture-specific diagnostics for paired analysis."""

    result: dict[str, Any] = {}
    if isinstance(bridge, TraceFoveaBridgeAdapter):
        if bridge.last_trace_diagnostics is not None:
            result["tracefovea"] = bridge.last_trace_diagnostics.as_dict()
        if encoder.last_bridge_gate_logits_by_block:
            result["child_gate_by_block"] = {
                str(block): {
                    "mean": float(logits.detach().float().sigmoid().mean().cpu()),
                    "std": float(
                        logits.detach().float().sigmoid().std(unbiased=False).cpu()
                    ),
                    "minimum": float(
                        logits.detach().float().sigmoid().min().cpu()
                    ),
                    "maximum": float(
                        logits.detach().float().sigmoid().max().cpu()
                    ),
                }
                for block, logits in encoder.last_bridge_gate_logits_by_block.items()
            }
    elif isinstance(bridge, EviRelayBridgeAdapter):
        if bridge.last_relay_diagnostics is not None:
            relay = bridge.last_relay_diagnostics.as_dict()
            source_tokens = int(relay["global_tokens"]) + int(relay["fine_tokens"])
            relay["gather_entropy_normalized"] = (
                float(relay["gather_entropy_mean"])
                / math.log(max(source_tokens, 2))
            )
            result["evirelay"] = relay
    if getattr(vision, "progressive_bridge_diagnostics", None):
        result["progressive_bridge_by_block"] = {
            str(block): diagnostics.as_dict()
            for block, diagnostics in vision.progressive_bridge_diagnostics.items()
        }
    if getattr(vision, "global_frame_joint_diagnostics", None):
        result["global_frame_joint_by_block"] = {
            str(block): diagnostics.as_dict()
            for block, diagnostics in vision.global_frame_joint_diagnostics.items()
        }
    if getattr(vision, "safe_joint_residual_diagnostics", None):
        result["eviblend_by_block"] = {
            str(block): diagnostics.as_dict()
            for block, diagnostics in vision.safe_joint_residual_diagnostics.items()
        }
    if getattr(vision, "trace_calibrated_bridge_diagnostics", None):
        result["trace_calibrated_bridge"] = (
            vision.trace_calibrated_bridge_diagnostics.as_dict()
        )
    if getattr(vision, "slot_diagnostics", None):
        result["evislot"] = vision.slot_diagnostics.as_dict()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path)
    parser.add_argument("--bridge-checkpoint", type=Path)
    parser.add_argument(
        "--adaptive-box-head",
        type=Path,
        help=(
            "EviViT-v6 zero-initialized residual box head. It refines the "
            "frozen trace_split anchors before PatchExchange Fine rereading."
        ),
    )
    parser.add_argument(
        "--adaptive-box-safety-mode",
        choices=(
            "replace",
            "anchor_union",
            "uncertainty_gate",
            "uncertainty_union",
            "residual_read",
        ),
        default="replace",
        help=(
            "v7 safety contract: preserve the original anchor, reject "
            "high-disagreement refinements, or combine both safeguards."
        ),
    )
    parser.add_argument(
        "--adaptive-box-ensemble-heads",
        type=Path,
        nargs="+",
        help="Fivefold AdaptiveBox heads used only for uncertainty estimation.",
    )
    parser.add_argument(
        "--adaptive-box-uncertainty-threshold",
        type=float,
        default=math.inf,
        help="Reject a refinement when its ensemble RMS disagreement exceeds this value.",
    )
    parser.add_argument(
        "--adaptive-box-residual-read-tokens",
        type=int,
        default=256,
        help=(
            "v7 residual_read: extra merged tokens for each refined proposal; "
            "the original v5 anchor and its PatchExchange budget remain intact."
        ),
    )
    parser.add_argument(
        "--record-eviguard-features",
        action="store_true",
        help=(
            "Store the fixed 272-D deployment feature vector used by EviGuard-v7. "
            "This records no answer, gold label, NLL or Judge-derived feature."
        ),
    )
    parser.add_argument("--eviguard-projection-dim", type=int, default=64)
    parser.add_argument("--eviguard-projection-seed", type=int, default=20260812)
    parser.add_argument(
        "--trace-refine-policy",
        type=Path,
        help=(
            "EviViT-v7 TraceRefine policy. It observes a low-cost 3x3 "
            "Block-16 preview around each v6 H-Safe box and selects one "
            "matched-prompt counterfactual portfolio before Fine rereading."
        ),
    )
    parser.add_argument(
        "--trace-refine-preview-expansion",
        type=float,
        default=1.5,
        help="Expansion of each H-Safe box used only by the policy preview.",
    )
    parser.add_argument(
        "--trace-refine-preview-tokens",
        type=int,
        default=256,
        help="Maximum merged visual tokens per TraceRefine context preview.",
    )
    parser.add_argument(
        "--fine-patch-evidence-bias",
        action="store_true",
        help=(
            "v7 PatchBias: sample the PTEA map at every pre-merger Fine patch "
            "and use it as a Bridge weight without adding views or tokens."
        ),
    )
    parser.add_argument(
        "--fine-readout-mode",
        choices=("native", "evidence_residual"),
        default="native",
        help=(
            "How merged Fine tokens are exposed to the LLM. evidence_residual "
            "keeps high-evidence Fine features while continuously falling back "
            "to the coordinate-matched Global parent for weak patches."
        ),
    )
    parser.add_argument(
        "--fine-patch-evidence-floor",
        type=float,
        default=0.5,
        help="Minimum retained Bridge weight for a low-evidence Fine patch.",
    )
    parser.add_argument(
        "--fine-patch-evidence-exponent",
        type=float,
        default=1.0,
        help="Exponent applied to normalized patch-level PTEA probabilities.",
    )
    parser.add_argument(
        "--tracescale-head",
        type=Path,
        help=(
            "Optional human-trace-supervised continuous region-scale head. "
            "It changes only PTEA box geometry; PTEA scores and Fine budgets "
            "remain unchanged."
        ),
    )
    parser.add_argument(
        "--tracescale-positive-only",
        action="store_true",
        help=(
            "TraceScale-Safe inference: clamp negative scale residuals to "
            "zero so learned geometry can never shrink a frozen PTEA box."
        ),
    )
    parser.add_argument(
        "--tracescale-strength",
        type=float,
        default=1.0,
        help="Non-negative multiplier for the learned log-scale residual.",
    )
    parser.add_argument(
        "--anchored-growth-calibrator",
        type=Path,
        help=(
            "Optional Stage-E E2 head. It predicts only the stop beta and an "
            "enable gate along the frozen E1 anchored-growth curve."
        ),
    )
    parser.add_argument(
        "--evicontour-utility-checkpoint",
        type=Path,
        help=(
            "Stage-G full-Trace1144 dual-head checkpoint. Required only by "
            "--evidence-decoder evicontour_utility."
        ),
    )
    parser.add_argument("--evicontour-csr-sufficiency-threshold", type=float, default=0.65)
    parser.add_argument("--evicontour-csr-sufficiency-tolerance", type=float, default=0.05)
    parser.add_argument("--evicontour-csr-core-relevance-tolerance", type=float, default=0.05)
    parser.add_argument("--evicontour-csr-context-ring-mass-ratio", type=float, default=0.20)
    parser.add_argument("--evicontour-csr-maximum-scope-hops", type=int, default=3)
    parser.add_argument("--evicontour-csr-minimum-parent-sufficiency-gain", type=float, default=0.03)
    parser.add_argument("--evicontour-csr-fine-floor-native-ratio", type=float, default=0.10)
    parser.add_argument("--evicontour-csr-minimum-fine-total", type=int, default=256)
    parser.add_argument("--evicontour-csr-maximum-fine-total", type=int, default=3072)
    parser.add_argument(
        "--anchored-growth-gate-threshold",
        type=float,
        default=0.65,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--project-root", type=Path, default=Path(".")
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen3-VL-4B-Instruct"),
    )
    parser.add_argument(
        "--language-lora-adapter",
        type=Path,
        help=(
            "Optional recovery-SFT language LoRA. Its base model, exact target "
            "modules and safetensors bundle are audited before model loading."
        ),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--memory-budget-mib",
        type=int,
        default=45000,
        help=(
            "Optional per-process GPU preflight/fraction cap in MiB; set 0 to "
            "use CUDA normally without an artificial project-side memory cap."
        ),
    )
    parser.add_argument("--insertion-block", type=int, default=16)
    parser.add_argument("--global-token-budget", type=int, default=1024)
    parser.add_argument("--fine-token-budget", type=int, default=3072)
    parser.add_argument(
        "--multi-shared-fine-plan",
        type=Path,
        help=(
            "Deterministic per-question cross-image plan. Every image keeps "
            "its normal Global path; only the PatchExchange Fine ceiling is "
            "overridden by the plan's per_image_fine_budgets."
        ),
    )
    parser.add_argument(
        "--processor-min-pixels",
        type=int,
        default=4096,
        help="Qwen image processor minimum pixels used by every visual path.",
    )
    parser.add_argument(
        "--processor-max-pixels",
        type=int,
        default=16 * 1024 * 1024,
        help="Qwen image processor maximum pixels used by every visual path.",
    )
    parser.add_argument(
        "--patch-exchange",
        action="store_true",
        help=(
            "Enable EviViT-v5 PatchExchange. Derive a per-image native B16 "
            "ceiling, coarsen Global by --patch-exchange-global-scale, and "
            "reread selected raw-image crops at --patch-exchange-local-scale."
        ),
    )
    parser.add_argument(
        "--patch-exchange-tree",
        action="store_true",
        help=(
            "EviViT-v8: keep the T2 Evidence-Need/tree region topology and "
            "Parent-Child checkpoint, but realize its Global/Fine views with "
            "the per-image PatchExchange pixel/token contract."
        ),
    )
    parser.add_argument(
        "--patch-exchange-global-scale",
        type=float,
        default=2.0,
        help="2.0 gives effective p32; 1.5 gives the preregistered p24 control.",
    )
    parser.add_argument(
        "--patch-exchange-local-scale",
        type=float,
        default=2.0,
        help="2.0 upsamples each raw crop by area-equivalent 4x for effective p8.",
    )
    parser.add_argument(
        "--patch-exchange-total-cap-ratio",
        type=float,
        default=1.0,
        help="Per-image total token ceiling relative to native B16 demand.",
    )
    parser.add_argument(
        "--patch-exchange-soft-floor-tokens",
        type=int,
        default=0,
        help=(
            "Continuous small-image token floor: the total-cap basis is "
            "sqrt(N_native^2 + floor^2). Zero preserves strict Stage A."
        ),
    )
    parser.add_argument(
        "--patch-exchange-balanced-soft-floor",
        action="store_true",
        help=(
            "Turn the continuous floor into actual Global/Fine allocations, "
            "rather than only relaxing the total ceiling."
        ),
    )
    parser.add_argument(
        "--patch-exchange-preserve-native-global",
        action="store_true",
        help=(
            "EviViT-v7-P2: continuously preserve the native B16 Global grid "
            "for low-native-token samples while retaining Global/Fine "
            "exchange for high-native-token samples. Requires balanced "
            "soft-floor mode and a positive soft floor."
        ),
    )
    parser.add_argument(
        "--patch-exchange-continuous-soft-floor",
        action="store_true",
        help=(
            "EviViT-v7-P3: treat --patch-exchange-soft-floor-tokens as the "
            "minimum and continuously increase it with per-image native "
            "Qwen token demand."
        ),
    )
    parser.add_argument(
        "--patch-exchange-continuous-soft-floor-max-tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--patch-exchange-continuous-soft-floor-ramp-start-tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--patch-exchange-continuous-soft-floor-ramp-end-tokens",
        type=int,
        default=12288,
    )
    parser.add_argument(
        "--patch-exchange-native-global-anchor",
        action="store_true",
        help=(
            "EviViT-v7-P4: keep the exact B16 native Global grid and make "
            "Fine evidence a bounded additive branch instead of exchanging "
            "away Global tokens."
        ),
    )
    parser.add_argument(
        "--patch-exchange-additive-fine-ratio",
        type=float,
        default=0.5,
        help="P4 additive Fine capacity as a fraction of native B16 tokens.",
    )
    parser.add_argument(
        "--patch-exchange-additive-fine-max-tokens",
        type=int,
        default=2048,
        help="P4 per-image maximum additive Fine capacity.",
    )
    parser.add_argument(
        "--patch-exchange-view-min-pixels",
        type=int,
        default=4096,
        help=(
            "Minimum legal Fine-view grid. It remains 4K in the paired 65K "
            "processor control so every branch is not independently upsampled."
        ),
    )
    parser.add_argument(
        "--native-pixel-contract",
        action="store_true",
        help=(
            "Enable EviViT-NativeScale pixel semantics. Global and every "
            "Fine crop use Qwen smart-resize under separate pixel maxima; "
            "the resulting token grids are never resized toward token targets."
        ),
    )
    parser.add_argument(
        "--global-processor-max-pixels",
        type=int,
        default=0,
        help=(
            "Global smart-resize maximum for NativeScale. Zero inherits "
            "--processor-max-pixels."
        ),
    )
    parser.add_argument(
        "--fine-processor-max-pixels",
        type=int,
        default=0,
        help=(
            "Per-crop smart-resize maximum for NativeScale. Zero inherits "
            "--processor-max-pixels."
        ),
    )
    parser.add_argument(
        "--continuous-native-global",
        action="store_true",
        help=(
            "Use Stage-F continuous Native Global scaling. Small images keep "
            "their natural Qwen grid; large-image pixel ceilings grow by a "
            "continuous power law between the base and native maxima."
        ),
    )
    parser.add_argument(
        "--continuous-global-base-pixels",
        type=int,
        default=4 * 1024 * 1024,
    )
    parser.add_argument(
        "--continuous-global-max-pixels",
        type=int,
        default=16 * 1024 * 1024,
    )
    parser.add_argument(
        "--continuous-global-exponent",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--visual-output-mode",
        choices=("append", "global_only"),
        default="append",
        help=(
            "append serializes Global and Fine tokens to the LLM; global_only "
            "uses Fine tokens inside the ViT/Bridge but emits only the fused "
            "Global stream to the LLM (Stage-F evidence injection)."
        ),
    )
    parser.add_argument(
        "--native-global-mrope",
        action="store_true",
        help=(
            "For Global-only evidence writeback, preserve Qwen's native "
            "row/column MRoPE grid exactly."
        ),
    )
    parser.add_argument(
        "--native-fuse",
        action="store_true",
        help=(
            "EviViT-v7 NativeFuse contract: preserve the exact native B16 "
            "Global canvas and MRoPE, use AdaptiveBox only to construct "
            "high-resolution Fine reads, fuse Fine evidence into Global at "
            "Block16, and emit Global tokens only."
        ),
    )
    parser.add_argument(
        "--native-fuse-strength",
        type=float,
        default=1.0,
        help=(
            "Multiply NativeFuse Fine evidence weights before bounded "
            "Fine-to-Global writeback. One is the direct frozen bridge; "
            "values in (0,1) are conservative residual ablations."
        ),
    )
    parser.add_argument(
        "--answer-protocol",
        choices=(
            "compact_json",
            "direct_concise",
            "vero_native",
            "muir_mcq",
            "blink_mcq",
            "vision_opd_official",
        ),
        default="compact_json",
        help=(
            "Actor prompt/output contract. v10 uses direct_concise so its "
            "identity lane is prompt-matched to the current B16 baseline."
        ),
    )
    parser.add_argument(
        "--bridge-mode",
        choices=(
            "bidirectional",
            "preserve_global",
            "evidence_weighted_writeback",
            "preserve_fine",
            "identity",
        ),
        default=None,
        help=(
            "Fusion direction override. By default the evaluator restores the "
            "training-time bridge_mode saved in the bridge checkpoint; legacy "
            "checkpoints without this field use bidirectional."
        ),
    )
    parser.add_argument(
        "--native-ratio-budget",
        action="store_true",
        help=(
            "Derive both Global and Fine budgets from the native Qwen token "
            "count under the same processor min/max-pixel contract."
        ),
    )
    parser.add_argument("--native-global-ratio", type=float, default=0.30)
    parser.add_argument("--native-fine-ratio", type=float, default=0.60)
    parser.add_argument(
        "--native-need-ratio-budget",
        action="store_true",
        help=(
            "Use a native-relative Global budget and interpolate the native-"
            "relative Fine ratio with the EviTree Need head."
        ),
    )
    parser.add_argument("--native-min-fine-ratio", type=float, default=0.20)
    parser.add_argument("--native-max-fine-ratio", type=float, default=0.60)
    parser.add_argument("--native-ratio-min-global-tokens", type=int, default=64)
    parser.add_argument("--native-ratio-min-fine-tokens", type=int, default=64)
    parser.add_argument(
        "--native-ratio-soft-floor-tokens",
        type=int,
        default=0,
        help=(
            "Optional continuous native-budget floor for Need-ratio allocation. "
            "The budget basis becomes sqrt(N_native^2 + floor^2), avoiding a "
            "hard small-image route while leaving large-image budgets nearly "
            "unchanged. Zero preserves the original behavior."
        ),
    )
    parser.add_argument(
        "--native-ratio-max-global-tokens",
        type=int,
        default=0,
        help="Optional aligned upper bound for the native-relative Global stream.",
    )
    parser.add_argument(
        "--native-ratio-max-fine-tokens",
        type=int,
        default=0,
        help="Optional aligned upper bound for the native-relative Fine stream.",
    )
    parser.add_argument("--native-ratio-alignment", type=int, default=64)
    parser.add_argument(
        "--native-token-caps",
        action="store_true",
        help=(
            "Enable Simple EviViT S1: interpret the configured Global and "
            "per-region Fine budgets as hard ceilings instead of targets. "
            "Small images preserve their complete native Qwen Global grid."
        ),
    )
    parser.add_argument(
        "--native-cap-scope",
        choices=("both", "global", "fine"),
        default="both",
        help=(
            "Choose whether Native-Cap applies to both visual paths, only "
            "Global, or only Fine. The default preserves the S1 protocol."
        ),
    )
    parser.add_argument(
        "--native-cap-minimum-fine-tokens",
        type=int,
        default=64,
        help=(
            "Structural minimum for each selected Fine crop in Native-Cap "
            "mode. Global has no additional floor beyond Qwen's processor."
        ),
    )
    parser.add_argument(
        "--native-cap-minimum-global-tokens",
        type=int,
        default=0,
        help=(
            "S3 topology floor for the Native-Cap Global grid. Zero preserves "
            "the pure S1 natural-grid protocol."
        ),
    )
    parser.add_argument(
        "--native-cap-anchor-fine-tokens",
        type=int,
        default=0,
        help=(
            "S3 readability floor applied only to Residual-Focus region R1. "
            "R2/R3 keep --native-cap-minimum-fine-tokens."
        ),
    )
    parser.add_argument(
        "--native-cap-fine-floor-native-ratio",
        type=float,
        default=0.0,
        help=(
            "If positive, replace the fixed per-region Fine floor by one "
            "continuous total floor proportional to the source image's native "
            "Qwen visual-token demand."
        ),
    )
    parser.add_argument(
        "--native-cap-minimum-fine-total",
        type=int,
        default=128,
        help="Minimum total Fine floor for the native-relative floor protocol.",
    )
    parser.add_argument(
        "--adaptive-native-global-budget",
        action="store_true",
        help=(
            "Replace the fixed global budget by a continuous function of the "
            "source image's native Qwen token demand."
        ),
    )
    parser.add_argument(
        "--adaptive-global-native-fraction", type=float, default=0.1875
    )
    parser.add_argument(
        "--adaptive-global-base-token-budget",
        type=int,
        default=0,
        help=(
            "Continuous-budget safety intercept added before the native-size "
            "term. This is not an image-size tier."
        ),
    )
    parser.add_argument(
        "--adaptive-global-min-token-budget", type=int, default=512
    )
    parser.add_argument(
        "--adaptive-global-max-token-budget", type=int, default=3072
    )
    parser.add_argument(
        "--adaptive-global-maximum-pixels",
        type=int,
        default=16 * 1024 * 1024,
    )
    parser.add_argument("--adaptive-global-budget-alignment", type=int, default=64)
    parser.add_argument(
        "--evitree-need-head",
        type=Path,
        help=(
            "Enable EviTree-A: retain the configured global budget and predict "
            "a continuous per-sample fine budget from the PTEA map."
        ),
    )
    parser.add_argument(
        "--evitree-min-fine-token-budget", type=int, default=512
    )
    parser.add_argument(
        "--evitree-max-fine-token-budget", type=int, default=4096
    )
    parser.add_argument("--evitree-budget-alignment", type=int, default=64)
    parser.add_argument(
        "--evitree-tree-allocation",
        action="store_true",
        help=(
            "Replace fixed-scale residual boxes with non-overlapping "
            "hierarchical EviTree leaves."
        ),
    )
    parser.add_argument("--evitree-tree-max-depth", type=int, default=5)
    parser.add_argument("--evitree-tree-max-leaves", type=int, default=12)
    parser.add_argument("--evitree-tree-reliability", type=float, default=0.80)
    parser.add_argument("--evitree-tree-target-mass", type=float, default=0.90)
    parser.add_argument(
        "--evitree-split-policy",
        type=Path,
        help=(
            "Optional human-trace-supervised split head. When set, EviTree "
            "uses variable-depth learned refinement instead of the B0 heuristic."
        ),
    )
    parser.add_argument(
        "--evitree-split-threshold",
        type=float,
        default=-1.0,
        help=(
            "Override the split checkpoint's OOF-calibrated stop threshold. "
            "Negative values use the checkpoint threshold."
        ),
    )
    parser.add_argument(
        "--evitree-sparse-branches",
        action="store_true",
        help=(
            "Use the learned tree as an adaptive candidate generator, but "
            "re-read only a few highest-evidence leaves; low-evidence leaves "
            "remain represented by the protected global stream."
        ),
    )
    parser.add_argument("--evitree-maximum-branches", type=int, default=3)
    parser.add_argument("--evitree-minimum-branch-depth", type=int, default=0)
    parser.add_argument("--evitree-branch-density-power", type=float, default=0.20)
    parser.add_argument("--evitree-recenter-branch-tips", action="store_true")
    parser.add_argument(
        "--evitree-branch-expansion-factor", type=float, default=1.0
    )
    parser.add_argument("--evitree-maximum-branch-iou", type=float, default=1.0)
    parser.add_argument(
        "--v10-gate-mode",
        choices=("disabled", "zero", "one", "learned", "entropy_only"),
        default="disabled",
        help=(
            "Reliability-calibrated soft-evidence smoke. 'one' must reproduce "
            "the hard v4 path; 'zero' keeps only the configured minimum fine budget."
        ),
    )
    parser.add_argument(
        "--v10-reliability-report",
        type=Path,
        help="Training-only v10-P0 report containing deployment gate parameters.",
    )
    parser.add_argument(
        "--v10-min-fine-token-budget",
        type=int,
        default=2048,
        help="Fine-token floor used by learned/entropy soft allocation.",
    )
    parser.add_argument(
        "--v10-map-mode",
        choices=("soft", "hard"),
        default="soft",
        help="Use reliability-flattened or original hard PTEA probability.",
    )
    parser.add_argument(
        "--v10-budget-mode",
        choices=("dynamic", "fixed"),
        default="dynamic",
        help="Interpolate the fine budget with reliability or keep its maximum.",
    )
    parser.add_argument(
        "--v10-bridge-gate-mode",
        choices=("scaled", "fixed"),
        default="scaled",
        help="Scale bridge evidence weights by reliability or keep v4 weights.",
    )
    parser.add_argument(
        "--native-fit-total-token-budget",
        type=int,
        default=0,
        help=(
            "EviViT-v7 guard: when the native full image fits within this merged-token "
            "budget, keep the native Qwen visual path and skip evidence rereading. "
            "Zero disables the guard."
        ),
    )
    parser.add_argument("--max-regions", type=int, default=3)
    parser.add_argument("--minimum-region-tokens", type=int, default=64)
    parser.add_argument("--bridge-dim", type=int, default=256)
    parser.add_argument("--bridge-heads", type=int, default=4)
    parser.add_argument("--neighborhood-radius", type=int, default=1)
    parser.add_argument("--max-relative-residual", type=float, default=0.2)
    parser.add_argument(
        "--coverage-masked-parent-writeback",
        action="store_true",
        help=(
            "T2-SafeWrite: keep Global parents with no assigned Fine child "
            "as exact identities; covered parents retain the checkpoint's "
            "original bounded residual update."
        ),
    )
    parser.add_argument("--coordinate-bins", type=int, default=128)
    parser.add_argument("--scale-bins", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--projection-seed", type=int, default=20260715)
    parser.add_argument(
        "--evidence-policy",
        choices=("mid_ptea", "uniform", "random", "feature_norm"),
        default="mid_ptea",
    )
    parser.add_argument(
        "--evidence-decoder",
        choices=(
            "top_boxes",
            "residual_focus",
            "topology_union",
            "adaptive_density_mass",
            "adaptive_density_mass_topology",
            "adaptive_pareto",
            "adaptive_pareto_topology",
            "adaptive_contour",
            "seeded_residual_basin",
            "seeded_basin_confidence_fallback",
            "seeded_basin_role_safe_r3",
            "seeded_basin_partial_fill_three",
            "seeded_basin_relaxed_three",
            "anchored_growth_b025",
            "anchored_growth_b035",
            "anchored_growth_calibrated",
            "evicontour_utility",
            "evicontour_csr",
            "trace_split",
        ),
        default="top_boxes",
        help="Training-free decoder that converts the Mid-PTEA map into regions.",
    )
    parser.add_argument("--region-expansion-factor", type=float, default=1.0)
    parser.add_argument("--topology-union-factor", type=float, default=1.0)
    parser.add_argument("--contour-relative-threshold", type=float, default=0.40)
    parser.add_argument("--contour-context-margin", type=float, default=0.20)
    parser.add_argument(
        "--contour-minimum-residual-mass", type=float, default=0.05
    )
    parser.add_argument(
        "--seeded-basin-config",
        type=Path,
        default=None,
        help=(
            "JSON configuration frozen by the Trace1144 five-fold Stage-D "
            "audit. The file may contain the config directly or under a "
            "top-level 'config' field."
        ),
    )
    parser.add_argument(
        "--stage-d-v27b-safe-caps",
        action="store_true",
        help=(
            "For Stage-D confidence fallback under NativeScale: preserve "
            "confident three-basin rows, but cap anchor-fallback R1/R2/R3 "
            "with the v27b two-decisive-plus-one-context contract."
        ),
    )
    parser.add_argument(
        "--stage-d-v27b-context-fraction",
        type=float,
        default=0.2176,
        help="Safe R3 context ceiling fraction; must remain in [0.10, 0.25].",
    )
    parser.add_argument(
        "--trace-split-context-fraction",
        type=float,
        default=0.25,
        help=(
            "For trace_split only: fixed share of the unchanged fine-token "
            "budget reserved for one complementary human-context region."
        ),
    )
    parser.add_argument(
        "--trace-split-context-expansion-factor",
        type=float,
        default=1.0,
        help=(
            "For trace_split only: expand only the complementary context "
            "region while preserving the decisive regions and total token budget."
        ),
    )
    parser.add_argument(
        "--trace-split-adaptive-region-minimum",
        action="store_true",
        help=(
            "Preserve the configured v27b per-region token floor whenever "
            "the per-image Fine budget can afford it, but lower only that "
            "structural floor for genuinely small native-relative budgets."
        ),
    )
    parser.add_argument(
        "--trace-split-context-decoder",
        choices=("top_boxes", "adaptive_density_mass"),
        default="top_boxes",
        help=(
            "For trace_split only: top_boxes reproduces v24/v27b; "
            "adaptive_density_mass continuously grows only the protected "
            "context region while preserving both decisive focus regions."
        ),
    )
    parser.add_argument(
        "--trace-split-budget-mode",
        choices=("fixed", "competitive", "learned"),
        default="fixed",
        help=(
            "For trace_split only: fixed keeps one global context quota; "
            "competitive lets the second decisive mode and context compete "
            "through an analytic rule; learned uses a tiny Trace1144-supervised "
            "context-need head under the unchanged fine-token budget."
        ),
    )
    parser.add_argument(
        "--trace-split-context-need-checkpoint",
        type=Path,
        help=(
            "Tiny scalar context-budget head required by learned trace_split."
        ),
    )
    parser.add_argument(
        "--trace-split-min-context-fraction",
        type=float,
        default=0.10,
        help="Minimum context share in competitive trace_split mode.",
    )
    parser.add_argument(
        "--trace-split-max-context-fraction",
        type=float,
        default=0.35,
        help="Maximum context share in competitive trace_split mode.",
    )
    parser.add_argument(
        "--token-serialization",
        choices=("append", "parent_interleave"),
        default="append",
        help=(
            "Ordering of the unchanged global+fine visual-token set. "
            "parent_interleave groups fine tokens after their nearest global parent."
        ),
    )
    parser.add_argument(
        "--global-frame-joint-blocks",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Zero-based frozen Qwen-ViT block indexes that temporarily attend "
            "over one global+fine sequence in original-image coordinates."
        ),
    )
    parser.add_argument(
        "--global-frame-coordinate-bins",
        type=int,
        default=0,
        help=(
            "Legacy fixed-bin coordinate frame. Zero preserves the native "
            "global RoPE exactly and maps only fine tokens continuously into "
            "that global grid."
        ),
    )
    parser.add_argument(
        "--global-frame-joint-topology",
        choices=("all_to_all", "global_hub"),
        default="all_to_all",
        help=(
            "all_to_all lets every fine region interact directly; global_hub "
            "lets each fine region interact with the complete global stream "
            "but isolates fine regions from one another."
        ),
    )
    parser.add_argument("--control-seed", type=int, default=20260718)
    parser.add_argument(
        "--ptea-question-control",
        choices=("matched", "shuffled", "zero"),
        default="matched",
        help=(
            "Causal PTEA control. 'shuffled' supplies a deterministic other-sample "
            "question only to PTEA; 'zero' removes question information from the "
            "PTEA tokens. The answer prompt always keeps the correct question."
        ),
    )
    parser.add_argument(
        "--ptea-question-shuffle-seed",
        type=int,
        default=20260909,
    )
    parser.add_argument(
        "--region-location-control",
        choices=("matched", "random_relocate", "spread_relocate"),
        default="matched",
        help=(
            "Post-geometry causal control. Relocation preserves every final Fine "
            "box's width, height, role, score and token budget; only x/y changes."
        ),
    )
    parser.add_argument(
        "--region-location-control-seed",
        type=int,
        default=20260909,
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Pass enable_thinking=False to chat templates that support it. "
            "Required for matched Qwen3.5/Vision-OPD concise-answer evaluation."
        ),
    )
    parser.add_argument("--record-timing", action="store_true")
    parser.add_argument(
        "--record-answer-confidence",
        action="store_true",
        help=(
            "Record greedy answer-token mean/min log probability for paired "
            "Native-versus-Evidence safety diagnostics. This is read-only and "
            "does not change visual encoding or token selection."
        ),
    )
    parser.add_argument(
        "--forced-context-bbox-field",
        default="",
        help=(
            "Training-diagnostic only: replace trace_split's third/context "
            "anchor with this manifest field before H-Safe, PatchExchange and "
            "Sparse Bridge. The field must hold one 0--1000 XYXY box."
        ),
    )
    parser.add_argument(
        "--forced-context-bbox-optional",
        action="store_true",
        help=(
            "Training-diagnostic only: when the forced-context field is absent "
            "or null, retain the formal automatic context anchor. This enables "
            "matched pre/post Human-Trace state manifests without inventing a "
            "box for the initial or post-reset state."
        ),
    )
    parser.add_argument(
        "--reference-answer-nll-only",
        action="store_true",
        help=(
            "Training-diagnostic only: skip generation and record the matched "
            "compact-JSON ground-truth answer-value NLL after the complete "
            "visual path. This output is not a benchmark prediction."
        ),
    )
    parser.add_argument("--timing-warmup-samples", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--representation-probe-output-dir",
        type=Path,
        help=(
            "Optional directory for raw, coordinate-aligned visual token "
            "features used by shared-PCA and intermediate-representation "
            "audits. This does not change inference outputs."
        ),
    )
    parser.add_argument(
        "--representation-probe-blocks",
        default="",
        help=(
            "Comma-separated post-Bridge ViT block counts to capture when "
            "--representation-probe-output-dir is set, for example 18,27."
        ),
    )
    parser.add_argument(
        "--representation-variant-name",
        default="evivit",
        help="Stable method name stored in representation-probe payloads.",
    )
    parser.add_argument(
        "--export-pixeleyes-adapter-span",
        action="store_true",
        help=(
            "Include final DeepStack tensors and a fail-closed frozen-P3 "
            "configuration record in representation probes for the two-row "
            "PixelEyes paired multi-turn smoke. Disabled by default because "
            "the extra tensors are large."
        ),
    )
    parser.add_argument(
        "--deterministic-inference",
        action="store_true",
        help="Use deterministic math-SDPA kernels for reproducible evaluation.",
    )
    parser.add_argument(
        "--deterministic-attention-backend",
        choices=("math", "mem_efficient", "flash"),
        default="math",
        help=(
            "SDPA backend locked under deterministic algorithms. Math is the "
            "strict default; efficient backends require a separate repeatability audit."
        ),
    )
    return parser.parse_args()


def attach_frozen_language_lora(model: Any, adapter_path: Path) -> Any:
    """Inject a read-only PEFT adapter and expose EviViT's expected base API."""

    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        is_trainable=False,
        local_files_only=True,
    )
    peft_model.eval()
    peft_model.requires_grad_(False)
    base_model = peft_model.get_base_model()
    base_model.eval()
    base_model.requires_grad_(False)
    return base_model


def main() -> int:
    args = parse_args()
    if args.reference_answer_nll_only:
        if args.answer_protocol != "compact_json":
            raise ValueError(
                "reference-answer NLL currently requires compact_json protocol"
            )
        if args.record_timing or args.record_answer_confidence:
            raise ValueError(
                "reference-answer NLL-only is exclusive with generation timing/confidence"
            )
    if args.forced_context_bbox_field and (
        args.evidence_policy != "mid_ptea"
        or args.evidence_decoder != "trace_split"
    ):
        raise ValueError(
            "forced context boxes require Mid-PTEA with trace_split decoding"
        )
    representation_probe_blocks = tuple(
        sorted(
            {
                int(value)
                for value in args.representation_probe_blocks.split(",")
                if value.strip()
            }
        )
    )
    if bool(args.representation_probe_output_dir) != bool(
        representation_probe_blocks
    ):
        raise ValueError(
            "representation probe output directory and capture blocks must "
            "be provided together"
        )
    if args.export_pixeleyes_adapter_span:
        required_pixel_eyes_p3 = {
            "insertion_block": (args.insertion_block, 18),
            "global_token_budget": (args.global_token_budget, 2048),
            "fine_token_budget": (args.fine_token_budget, 3072),
            "max_regions": (args.max_regions, 3),
            "minimum_region_tokens": (args.minimum_region_tokens, 64),
            "evidence_policy": (args.evidence_policy, "mid_ptea"),
            "evidence_decoder": (args.evidence_decoder, "trace_split"),
            "trace_split_budget_mode": (
                args.trace_split_budget_mode,
                "learned",
            ),
            "trace_split_min_context_fraction": (
                args.trace_split_min_context_fraction,
                0.10,
            ),
            "trace_split_max_context_fraction": (
                args.trace_split_max_context_fraction,
                0.25,
            ),
            "trace_split_context_expansion_factor": (
                args.trace_split_context_expansion_factor,
                1.0,
            ),
            "trace_split_context_decoder": (
                args.trace_split_context_decoder,
                "top_boxes",
            ),
            "adaptive_box_safety_mode": (
                args.adaptive_box_safety_mode,
                "replace",
            ),
            "processor_min_pixels": (args.processor_min_pixels, 4096),
            "processor_max_pixels": (
                args.processor_max_pixels,
                16 * 1024 * 1024,
            ),
            "patch_exchange_global_scale": (
                args.patch_exchange_global_scale,
                1.5,
            ),
            "patch_exchange_local_scale": (
                args.patch_exchange_local_scale,
                2.0,
            ),
            "patch_exchange_total_cap_ratio": (
                args.patch_exchange_total_cap_ratio,
                1.0,
            ),
            "patch_exchange_soft_floor_tokens": (
                args.patch_exchange_soft_floor_tokens,
                2048,
            ),
            "patch_exchange_continuous_soft_floor_max_tokens": (
                args.patch_exchange_continuous_soft_floor_max_tokens,
                4096,
            ),
            "patch_exchange_continuous_soft_floor_ramp_start_tokens": (
                args.patch_exchange_continuous_soft_floor_ramp_start_tokens,
                4096,
            ),
            "patch_exchange_continuous_soft_floor_ramp_end_tokens": (
                args.patch_exchange_continuous_soft_floor_ramp_end_tokens,
                12288,
            ),
            "patch_exchange_view_min_pixels": (
                args.patch_exchange_view_min_pixels,
                4096,
            ),
        }
        mismatches = {
            key: {"actual": actual, "expected": expected}
            for key, (actual, expected) in required_pixel_eyes_p3.items()
            if actual != expected
        }
        required_flags = {
            "patch_exchange": args.patch_exchange,
            "patch_exchange_balanced_soft_floor": (
                args.patch_exchange_balanced_soft_floor
            ),
            "patch_exchange_preserve_native_global": (
                args.patch_exchange_preserve_native_global
            ),
            "patch_exchange_continuous_soft_floor": (
                args.patch_exchange_continuous_soft_floor
            ),
        }
        disabled_flags = [
            name for name, enabled in required_flags.items() if not enabled
        ]
        required_paths = {
            "selector_checkpoint": args.selector_checkpoint,
            "bridge_checkpoint": args.bridge_checkpoint,
            "trace_split_context_need_checkpoint": (
                args.trace_split_context_need_checkpoint
            ),
            "adaptive_box_head": args.adaptive_box_head,
        }
        missing_paths = [
            name for name, path in required_paths.items() if path is None
        ]
        if mismatches or disabled_flags or missing_paths:
            raise ValueError(
                "PixelEyes span export requires the frozen 8B P3 contract: "
                f"mismatches={mismatches}, disabled_flags={disabled_flags}, "
                f"missing_paths={missing_paths}"
            )
    seeded_basin_config: dict[str, Any] | None = None
    if args.seeded_basin_config is not None:
        payload = json.loads(args.seeded_basin_config.read_text())
        seeded_basin_config = payload.get("config", payload)
        if not isinstance(seeded_basin_config, dict):
            raise ValueError("seeded basin configuration must be a JSON object")
    if (
        args.evidence_decoder.startswith("seeded_")
        and seeded_basin_config is None
    ):
        raise ValueError(
            "Stage-D seeded decoders require --seeded-basin-config so the "
            "Trace1144-frozen protocol is explicit and reproducible"
        )
    if (
        not args.evidence_decoder.startswith("seeded_")
        and seeded_basin_config is not None
    ):
        raise ValueError(
            "--seeded-basin-config is only valid with a Stage-D seeded decoder"
        )
    global_processor_max_pixels = (
        args.global_processor_max_pixels or args.processor_max_pixels
    )
    fine_processor_max_pixels = (
        args.fine_processor_max_pixels or args.processor_max_pixels
    )
    native_global_cap = args.native_token_caps and args.native_cap_scope in {
        "both",
        "global",
    }
    native_fine_cap = args.native_token_caps and args.native_cap_scope in {
        "both",
        "fine",
    }
    if args.global_token_budget <= 0 or args.fine_token_budget <= 0:
        raise ValueError("global and fine token budgets must be positive")
    if args.multi_shared_fine_plan is not None and not args.patch_exchange:
        raise ValueError("multi-image shared Fine planning requires PatchExchange")
    if args.tracescale_strength < 0:
        raise ValueError("TraceScale strength must be non-negative")
    if args.tracescale_head is not None and args.evidence_policy != "mid_ptea":
        raise ValueError("TraceScale requires --evidence-policy mid_ptea")
    if args.adaptive_box_head is not None:
        if args.tracescale_head is not None:
            raise ValueError("AdaptiveBox and TraceScale cannot refine the same regions")
        if not args.patch_exchange and not args.native_fuse:
            raise ValueError(
                "AdaptiveBox requires --patch-exchange or --native-fuse"
            )
        if args.patch_exchange_tree:
            if (
                args.evidence_policy != "mid_ptea"
                or args.evidence_decoder != "residual_focus"
                or not args.evitree_tree_allocation
            ):
                raise ValueError(
                    "AdaptiveBox v9 requires frozen Mid-PTEA/T2 tree anchors"
                )
        elif (
            args.evidence_policy != "mid_ptea"
            or args.evidence_decoder != "trace_split"
        ):
            raise ValueError(
                "AdaptiveBox v6/v7 requires frozen Mid-PTEA trace_split anchors"
            )
    if args.adaptive_box_safety_mode != "replace" and args.adaptive_box_head is None:
        raise ValueError("adaptive box safety modes require --adaptive-box-head")
    if args.adaptive_box_residual_read_tokens <= 0:
        raise ValueError("residual-read token count must be positive")
    if args.trace_refine_policy is not None:
        if args.adaptive_box_head is None:
            raise ValueError("TraceRefine requires the v6 H-Safe adaptive box head")
        if args.adaptive_box_safety_mode != "replace":
            raise ValueError("TraceRefine currently requires H-Safe replace semantics")
        if args.evidence_policy != "mid_ptea" or args.evidence_decoder != "trace_split":
            raise ValueError("TraceRefine requires frozen Mid-PTEA trace_split anchors")
        if args.trace_refine_preview_expansion <= 1.0:
            raise ValueError("TraceRefine preview expansion must be greater than one")
        if args.trace_refine_preview_tokens < 16:
            raise ValueError("TraceRefine preview token budget must be at least 16")
    if not 0 <= args.fine_patch_evidence_floor <= 1:
        raise ValueError("Fine patch evidence floor must lie in [0, 1]")
    if args.fine_patch_evidence_exponent <= 0:
        raise ValueError("Fine patch evidence exponent must be positive")
    if args.fine_patch_evidence_bias and args.evidence_policy != "mid_ptea":
        raise ValueError("Fine patch evidence bias requires Mid-PTEA")
    if args.adaptive_box_safety_mode.startswith("uncertainty"):
        if not args.adaptive_box_ensemble_heads or len(args.adaptive_box_ensemble_heads) < 2:
            raise ValueError("uncertainty safety requires at least two ensemble heads")
        if (
            not math.isfinite(args.adaptive_box_uncertainty_threshold)
            or args.adaptive_box_uncertainty_threshold < 0
        ):
            raise ValueError("uncertainty threshold must be finite and non-negative")
    if not 0 < args.anchored_growth_gate_threshold < 1:
        raise ValueError("anchored growth gate threshold must lie in (0, 1)")
    if (
        args.evidence_decoder == "anchored_growth_calibrated"
    ) != (args.anchored_growth_calibrator is not None):
        raise ValueError(
            "anchored_growth_calibrated and --anchored-growth-calibrator "
            "must be enabled together"
        )
    if (args.evidence_decoder in {"evicontour_utility", "evicontour_csr"}) != (
        args.evicontour_utility_checkpoint is not None
    ):
        raise ValueError(
            "EviContour decoder and --evicontour-utility-checkpoint "
            "must be enabled together"
        )
    if args.evidence_decoder == "evicontour_csr":
        if not 0.0 <= args.evicontour_csr_sufficiency_threshold <= 1.0:
            raise ValueError("invalid EviContour-CSR sufficiency threshold")
        if min(
            args.evicontour_csr_sufficiency_tolerance,
            args.evicontour_csr_core_relevance_tolerance,
            args.evicontour_csr_context_ring_mass_ratio,
            args.evicontour_csr_fine_floor_native_ratio,
            args.evicontour_csr_minimum_parent_sufficiency_gain,
        ) < 0:
            raise ValueError("EviContour-CSR tolerances/ratios must be non-negative")
        if args.evicontour_csr_maximum_scope_hops < 0:
            raise ValueError("EviContour-CSR maximum scope hops must be non-negative")
        if not (
            0 < args.evicontour_csr_minimum_fine_total
            <= args.evicontour_csr_maximum_fine_total
            <= args.fine_token_budget
        ):
            raise ValueError("invalid EviContour-CSR Fine floor interval")
    if (
        args.processor_min_pixels <= 0
        or args.processor_max_pixels < args.processor_min_pixels
    ):
        raise ValueError("invalid processor min/max-pixel interval")
    if (
        global_processor_max_pixels < args.processor_min_pixels
        or fine_processor_max_pixels < args.processor_min_pixels
    ):
        raise ValueError("invalid NativeScale Global/Fine pixel interval")
    if args.patch_exchange_tree and not args.patch_exchange:
        raise ValueError("--patch-exchange-tree requires --patch-exchange")
    if args.patch_exchange:
        tree_patch_exchange = bool(args.patch_exchange_tree)
        incompatible = (
            args.native_pixel_contract
            or args.native_token_caps
            or args.native_ratio_budget
            or (args.native_need_ratio_budget and not tree_patch_exchange)
            or args.adaptive_native_global_budget
            or args.native_fit_total_token_budget > 0
            or (args.evitree_need_head is not None and not tree_patch_exchange)
            or args.v10_gate_mode != "disabled"
            or args.continuous_native_global
            or args.stage_d_v27b_safe_caps
        )
        if incompatible:
            raise ValueError(
                "PatchExchange is a single-variable sampling protocol and "
                "cannot be combined with another Global/Fine budget mode"
            )
        if args.global_processor_max_pixels or args.fine_processor_max_pixels:
            raise ValueError(
                "PatchExchange derives both streams from processor min/max; "
                "separate NativeScale maxima are not allowed"
            )
        if tree_patch_exchange:
            if (
                args.evidence_policy != "mid_ptea"
                or args.evidence_decoder != "residual_focus"
                or not args.native_need_ratio_budget
                or not args.evitree_tree_allocation
                or args.evitree_need_head is None
                or args.evitree_split_policy is None
            ):
                raise ValueError(
                    "EviViT-v8 Tree-PatchExchange requires Mid-PTEA, "
                    "residual_focus, Native-Need budgeting, a Need head, "
                    "tree allocation and a learned split policy"
                )
        elif (
            args.evidence_policy == "mid_ptea"
            and args.evidence_decoder != "trace_split"
        ):
            raise ValueError(
                "EviViT-v5 PatchExchange requires trace_split for Mid-PTEA; "
                "explicit equal-budget control policies use their own decoder"
            )
        if args.max_regions != 3:
            raise ValueError("PatchExchange v5-A requires max_regions=3")
        if (
            not math.isfinite(args.patch_exchange_global_scale)
            or args.patch_exchange_global_scale < 1.0
            or not math.isfinite(args.patch_exchange_local_scale)
            or args.patch_exchange_local_scale < 1.0
            or not math.isfinite(args.patch_exchange_total_cap_ratio)
            or args.patch_exchange_total_cap_ratio <= 0
            or args.patch_exchange_soft_floor_tokens < 0
        ):
            raise ValueError("invalid PatchExchange scale or total-cap ratio")
        if (
            args.patch_exchange_balanced_soft_floor
            and args.patch_exchange_soft_floor_tokens <= 0
        ):
            raise ValueError("balanced PatchExchange requires a positive soft floor")
        if args.patch_exchange_preserve_native_global and (
            not args.patch_exchange_balanced_soft_floor
            or args.patch_exchange_soft_floor_tokens <= 0
        ):
            raise ValueError(
                "native-Global preservation requires balanced PatchExchange "
                "with a positive soft floor"
            )
        if args.patch_exchange_continuous_soft_floor and (
            not args.patch_exchange_balanced_soft_floor
            or args.patch_exchange_soft_floor_tokens <= 0
            or args.patch_exchange_continuous_soft_floor_max_tokens
            < args.patch_exchange_soft_floor_tokens
            or args.patch_exchange_continuous_soft_floor_ramp_start_tokens < 0
            or args.patch_exchange_continuous_soft_floor_ramp_end_tokens
            <= args.patch_exchange_continuous_soft_floor_ramp_start_tokens
        ):
            raise ValueError(
                "continuous soft floor requires balanced mode, a valid "
                "minimum/maximum interval, and an increasing native-token ramp"
            )
        if args.patch_exchange_native_global_anchor and (
            args.patch_exchange_balanced_soft_floor
            or args.patch_exchange_preserve_native_global
            or args.patch_exchange_continuous_soft_floor
            or not math.isfinite(args.patch_exchange_additive_fine_ratio)
            or args.patch_exchange_additive_fine_ratio < 0
            or args.patch_exchange_additive_fine_max_tokens < 0
        ):
            raise ValueError(
                "P4 NativeAnchor is exclusive with P2/P3 soft-floor modes "
                "and requires non-negative additive Fine limits"
            )
        if not (
            0
            < args.patch_exchange_view_min_pixels
            <= args.processor_max_pixels
        ):
            raise ValueError("invalid PatchExchange Fine-view pixel minimum")
    if args.native_pixel_contract:
        incompatible = (
            args.native_token_caps
            or args.native_ratio_budget
            or args.native_need_ratio_budget
            or args.adaptive_native_global_budget
            or args.native_fit_total_token_budget > 0
            or args.evitree_need_head is not None
            or args.v10_gate_mode != "disabled"
        )
        if incompatible:
            raise ValueError(
                "NativeScale pixel semantics cannot be combined with any "
                "token-target, ratio, Need, adaptive, bypass, or gate budget"
            )
    elif args.global_processor_max_pixels or args.fine_processor_max_pixels:
        raise ValueError(
            "separate Global/Fine pixel maxima require --native-pixel-contract"
        )
    if args.native_global_mrope and (
        args.visual_output_mode != "global_only"
        or not args.native_pixel_contract
    ):
        raise ValueError(
            "--native-global-mrope requires --visual-output-mode=global_only "
            "and --native-pixel-contract"
        )
    if args.native_global_mrope and (
        global_processor_max_pixels != args.processor_max_pixels
    ):
        raise ValueError(
            "native Global identity requires equal processor/global pixel "
            "maxima so the untouched source follows the B16 processor"
        )
    if args.native_fuse:
        if not 0.0 < args.native_fuse_strength <= 1.0:
            raise ValueError("NativeFuse strength must lie in (0, 1]")
        if args.patch_exchange:
            raise ValueError(
                "NativeFuse keeps the native B16 Global path and cannot use "
                "PatchExchange Global resampling"
            )
        if not args.native_pixel_contract:
            raise ValueError("NativeFuse requires --native-pixel-contract")
        if not args.native_global_mrope:
            raise ValueError("NativeFuse requires --native-global-mrope")
        if args.visual_output_mode != "global_only":
            raise ValueError(
                "NativeFuse requires --visual-output-mode=global_only"
            )
        if args.adaptive_box_head is None:
            raise ValueError("NativeFuse requires the frozen v6 H-Safe box head")
        if args.evidence_policy != "mid_ptea" or args.evidence_decoder != "trace_split":
            raise ValueError(
                "NativeFuse requires the frozen Mid-PTEA trace_split region protocol"
            )
        if args.global_processor_max_pixels not in {None, args.processor_max_pixels}:
            raise ValueError(
                "NativeFuse Global maximum must equal the B16 processor maximum"
            )
    if args.continuous_native_global:
        if not args.native_pixel_contract:
            raise ValueError(
                "continuous Native Global requires --native-pixel-contract"
            )
        if not (
            args.processor_min_pixels
            <= args.continuous_global_base_pixels
            <= args.continuous_global_max_pixels
        ):
            raise ValueError(
                "continuous Global pixels must satisfy processor minimum <= "
                "base <= maximum"
            )
        if not 0.0 <= args.continuous_global_exponent <= 1.0:
            raise ValueError(
                "continuous Global exponent must lie in [0, 1]"
            )
    if args.stage_d_v27b_safe_caps:
        if not args.native_pixel_contract:
            raise ValueError("Stage-D v27b-safe caps require NativeScale pixels")
        if args.evidence_decoder != "seeded_basin_confidence_fallback":
            raise ValueError(
                "Stage-D v27b-safe caps require confidence-fallback regions"
            )
        if not 0.10 <= args.stage_d_v27b_context_fraction <= 0.25:
            raise ValueError("Stage-D context fraction must lie in [0.10, 0.25]")
    if args.native_ratio_budget and args.native_need_ratio_budget:
        raise ValueError("fixed and Need-conditioned native ratios are exclusive")
    if args.native_token_caps:
        incompatible = (
            args.native_ratio_budget
            or args.native_need_ratio_budget
            or args.adaptive_native_global_budget
            or args.native_fit_total_token_budget > 0
            or args.evitree_need_head is not None
            or args.v10_gate_mode != "disabled"
        )
        if incompatible:
            raise ValueError(
                "Native-Cap is a single-variable budget protocol and cannot "
                "be combined with ratio/adaptive/Need/gated budget modes"
            )
        if not (
            0
            < args.native_cap_minimum_fine_tokens
            <= args.minimum_region_tokens
            <= args.fine_token_budget
        ):
            raise ValueError(
                "Native-Cap Fine floor must be positive, no larger than the "
                "existing minimum-region-tokens, and within the Fine cap"
            )
        if not (
            0
            <= args.native_cap_minimum_global_tokens
            <= args.global_token_budget
        ):
            raise ValueError(
                "Native-Cap Global floor must lie in [0, global-token-budget]"
            )
        if not (
            0
            <= args.native_cap_anchor_fine_tokens
            <= args.fine_token_budget
        ):
            raise ValueError(
                "Native-Cap anchor Fine floor must lie in [0, fine-token-budget]"
            )
        if (
            args.native_cap_fine_floor_native_ratio < 0
            or args.native_cap_minimum_fine_total < 0
            or args.native_cap_minimum_fine_total > args.fine_token_budget
        ):
            raise ValueError("invalid continuous Native-Cap Fine floor")
        if args.native_cap_fine_floor_native_ratio and not native_fine_cap:
            raise ValueError("continuous Fine floor requires Fine Native-Cap")
    elif args.native_cap_fine_floor_native_ratio:
        raise ValueError("continuous Fine floor requires --native-token-caps")
    if args.native_ratio_budget:
        if args.adaptive_native_global_budget:
            raise ValueError(
                "native-ratio and adaptive-global budgets are mutually exclusive"
            )
        if args.native_fit_total_token_budget:
            raise ValueError(
                "native-ratio budget cannot be combined with native-fit bypass"
            )
        if args.evitree_need_head is not None:
            raise ValueError(
                "native-ratio Fine and learned Need Fine cannot both control budget"
            )
        if args.v10_gate_mode != "disabled":
            raise ValueError(
                "native-ratio budget cannot be combined with v10 budget gating"
            )
        if not 0 < args.native_global_ratio <= 1:
            raise ValueError("native global ratio must lie in (0, 1]")
        if not 0 < args.native_fine_ratio <= 1:
            raise ValueError("native fine ratio must lie in (0, 1]")
        if (
            args.native_ratio_min_global_tokens <= 0
            or args.native_ratio_min_fine_tokens <= 0
            or args.native_ratio_alignment <= 0
        ):
            raise ValueError("native-ratio floors/alignment must be positive")
    if args.native_need_ratio_budget:
        if args.adaptive_native_global_budget or args.native_fit_total_token_budget:
            raise ValueError(
                "Need-conditioned native ratios cannot use adaptive-global "
                "or native-fit bypass"
            )
        if args.evitree_need_head is None:
            raise ValueError(
                "Need-conditioned native ratios require --evitree-need-head"
            )
        if not 0 < args.native_global_ratio <= 1:
            raise ValueError("native global ratio must lie in (0, 1]")
        if not (
            0 < args.native_min_fine_ratio
            <= args.native_max_fine_ratio
            <= 1
        ):
            raise ValueError("invalid native Fine ratio interval")
        if (
            args.native_ratio_soft_floor_tokens < 0
            or
            args.native_ratio_max_global_tokens < 0
            or args.native_ratio_max_fine_tokens < 0
            or (
                args.native_ratio_max_global_tokens
                and args.native_ratio_max_global_tokens
                < args.native_ratio_min_global_tokens
            )
            or (
                args.native_ratio_max_fine_tokens
                and args.native_ratio_max_fine_tokens
                < args.native_ratio_min_fine_tokens
            )
        ):
            raise ValueError("invalid native Need-ratio stream ceilings")
    if args.adaptive_native_global_budget:
        if args.adaptive_global_base_token_budget < 0:
            raise ValueError("adaptive global base token budget must be non-negative")
        if not 0 < args.adaptive_global_native_fraction <= 1:
            raise ValueError("adaptive global native fraction must lie in (0, 1]")
        if not (
            0
            < args.adaptive_global_min_token_budget
            <= args.adaptive_global_max_token_budget
        ):
            raise ValueError("invalid adaptive global-token interval")
        if (
            args.adaptive_global_maximum_pixels <= 0
            or args.adaptive_global_budget_alignment <= 0
        ):
            raise ValueError("adaptive global pixel/alignment values must be positive")
    if not 0.1 <= args.trace_split_context_fraction <= 0.5:
        raise ValueError(
            "trace-split-context-fraction must be in [0.1, 0.5]"
        )
    if args.trace_split_context_expansion_factor < 1.0:
        raise ValueError(
            "trace-split-context-expansion-factor must be at least 1.0"
        )
    if not (
        0.1
        <= args.trace_split_min_context_fraction
        <= args.trace_split_max_context_fraction
        <= 0.5
    ):
        raise ValueError(
            "trace-split competitive context range must lie in [0.1, 0.5]"
        )
    if (
        args.trace_split_budget_mode == "learned"
        and args.trace_split_context_need_checkpoint is None
    ):
        raise ValueError(
            "learned trace_split requires "
            "--trace-split-context-need-checkpoint"
        )
    if not 0 <= args.v10_min_fine_token_budget <= args.fine_token_budget:
        raise ValueError("v10 fine-token floor must lie in [0, fine-token-budget]")
    if args.v10_gate_mode != "disabled" and args.evidence_policy != "mid_ptea":
        raise ValueError("v10 reliability gating requires --evidence-policy mid_ptea")
    if args.evitree_need_head is not None:
        if args.evidence_policy != "mid_ptea":
            raise ValueError("EviTree-A requires --evidence-policy mid_ptea")
        if args.v10_gate_mode != "disabled":
            raise ValueError("EviTree-A and v10 gating cannot be enabled together")
        if not (
            0
            < args.evitree_min_fine_token_budget
            <= args.evitree_max_fine_token_budget
        ):
            raise ValueError("invalid EviTree fine-token interval")
        if args.evitree_budget_alignment <= 0:
            raise ValueError("EviTree budget alignment must be positive")
    if args.evitree_tree_allocation:
        if args.evidence_policy != "mid_ptea":
            raise ValueError("EviTree hierarchy requires --evidence-policy mid_ptea")
        if args.v10_gate_mode != "disabled":
            raise ValueError("EviTree hierarchy and v10 gating cannot be combined")
        if not 1 <= args.evitree_tree_max_depth <= 8:
            raise ValueError("EviTree maximum depth must lie in [1, 8]")
        if args.evitree_tree_max_leaves <= 0:
            raise ValueError("EviTree maximum leaves must be positive")
        if not 0.0 <= args.evitree_tree_reliability <= 1.0:
            raise ValueError("EviTree reliability must lie in [0, 1]")
        if not 0.0 < args.evitree_tree_target_mass <= 1.0:
            raise ValueError("EviTree target mass must lie in (0, 1]")
    if args.evitree_split_policy is not None and not args.evitree_tree_allocation:
        raise ValueError("EviTree split policy requires --evitree-tree-allocation")
    if args.evitree_sparse_branches and args.evitree_split_policy is None:
        raise ValueError("sparse EviTree branches require a learned split policy")
    if args.evitree_maximum_branches <= 0:
        raise ValueError("EviTree maximum branches must be positive")
    if args.evitree_split_threshold > 1.0:
        raise ValueError("EviTree split threshold cannot exceed 1")
    if (
        args.v10_gate_mode in {"learned", "entropy_only"}
        and args.v10_reliability_report is None
    ):
        raise ValueError("learned v10 gate modes require --v10-reliability-report")
    if args.native_fit_total_token_budget < 0:
        raise ValueError("--native-fit-total-token-budget must be non-negative")
    if args.native_fit_total_token_budget and args.record_timing:
        raise ValueError("native-fit timing requires a separate isolated profiler")
    if args.timing_warmup_samples < 0:
        raise ValueError("--timing-warmup-samples must be non-negative")
    # A resumable prediction file is a single-writer artifact. Manual
    # acceleration and an automatic watcher may otherwise append the same IDs
    # concurrently and silently corrupt a scientific result.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_lock = args.output.with_suffix(args.output.suffix + ".lock")
    lock_handle = output_lock.open("a+", encoding="utf-8")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    if args.memory_budget_mib < 0:
        raise ValueError("--memory-budget-mib must be non-negative")
    if args.memory_budget_mib:
        decision = check_gpu(
            args.gpu, args.memory_budget_mib, ROOT / "configs/gpu_safety.json"
        )
        print(
            json.dumps({"gpu_preflight": decision.as_dict()}, ensure_ascii=False),
            flush=True,
        )
        if not decision.allowed:
            return 2
    else:
        print(
            json.dumps(
                {
                    "gpu_preflight": {
                        "allowed": True,
                        "mode": "cuda_unbounded_no_project_cap",
                        "gpu": args.gpu,
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.deterministic_inference:
        configure_deterministic_inference(args.deterministic_attention_backend)
    device = "cuda:0"
    total_mib = torch.cuda.get_device_properties(0).total_memory / 1024**2
    if args.memory_budget_mib:
        torch.cuda.set_per_process_memory_fraction(
            min(0.95, args.memory_budget_mib / total_mib), device=0
        )
    torch.cuda.reset_peak_memory_stats(0)

    project_root = args.project_root
    manifest = resolve_path(args.manifest, project_root)
    model_path = resolve_path(args.model, project_root)
    language_lora_adapter_path = (
        resolve_path(args.language_lora_adapter, project_root)
        if args.language_lora_adapter is not None
        else None
    )
    language_lora_adapter_audit = (
        validate_language_lora_adapter(
            language_lora_adapter_path,
            expected_base_model=model_path,
        )
        if language_lora_adapter_path is not None
        else None
    )
    # Hugging Face checkpoints may be stored either as sharded safetensors
    # with an index, or as one model.safetensors file.  Loading is delegated
    # to ``from_pretrained`` below, so the evaluator should audit either valid
    # layout instead of rejecting single-file checkpoints such as Vision-OPD.
    model_index = model_path / "model.safetensors.index.json"
    single_model_weights = model_path / "model.safetensors"
    if model_index.is_file():
        model_weight_manifest = model_index
    elif single_model_weights.is_file():
        model_weight_manifest = single_model_weights
    else:
        raise FileNotFoundError(
            "missing model weights: expected either "
            f"{model_index} or {single_model_weights}"
        )
    model_index_sha256 = sha256_file(model_weight_manifest)
    if args.evidence_policy == "mid_ptea" and args.selector_checkpoint is None:
        raise ValueError("--selector-checkpoint is required for --evidence-policy mid_ptea")
    selector_checkpoint = (
        resolve_path(args.selector_checkpoint, project_root)
        if args.selector_checkpoint is not None
        else None
    )
    tracescale_head_path = (
        resolve_path(args.tracescale_head, project_root)
        if args.tracescale_head is not None
        else None
    )
    adaptive_box_head_path = (
        resolve_path(args.adaptive_box_head, project_root)
        if args.adaptive_box_head is not None
        else None
    )
    trace_refine_policy_path = (
        resolve_path(args.trace_refine_policy, project_root)
        if args.trace_refine_policy is not None
        else None
    )
    adaptive_box_ensemble_paths = [
        resolve_path(path, project_root)
        for path in (args.adaptive_box_ensemble_heads or [])
    ]
    anchored_growth_calibrator_path = (
        resolve_path(args.anchored_growth_calibrator, project_root)
        if args.anchored_growth_calibrator is not None
        else None
    )
    evicontour_utility_path = (
        resolve_path(args.evicontour_utility_checkpoint, project_root)
        if args.evicontour_utility_checkpoint is not None
        else None
    )
    context_need_checkpoint = (
        resolve_path(args.trace_split_context_need_checkpoint, project_root)
        if args.trace_split_context_need_checkpoint is not None
        else None
    )
    bridge_checkpoint = (
        resolve_path(args.bridge_checkpoint, project_root)
        if args.bridge_checkpoint is not None
        else None
    )
    reliability_report_path = (
        resolve_path(args.v10_reliability_report, project_root)
        if args.v10_reliability_report is not None
        else None
    )
    evitree_need_head_path = (
        resolve_path(args.evitree_need_head, project_root)
        if args.evitree_need_head is not None
        else None
    )
    evitree_split_policy_path = (
        resolve_path(args.evitree_split_policy, project_root)
        if args.evitree_split_policy is not None
        else None
    )
    rows = read_jsonl(manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    if args.ptea_question_control == "shuffled":
        if args.evidence_policy != "mid_ptea":
            raise ValueError(
                "question-shuffled PTEA control requires --evidence-policy mid_ptea"
            )
        shuffled_controls = deterministic_deranged_question_controls(
            rows, seed=args.ptea_question_shuffle_seed
        )
        for row in rows:
            control = shuffled_controls[str(row["id"])]
            row["_ptea_control_question"] = control["question"]
            row["_ptea_control_source_id"] = control["source_id"]
    multi_shared_plan_path = (
        resolve_path(args.multi_shared_fine_plan, project_root)
        if args.multi_shared_fine_plan is not None
        else None
    )
    multi_shared_plans: dict[str, dict[str, Any]] = {}
    if multi_shared_plan_path is not None:
        multi_shared_plans = {
            str(item["id"]): item for item in read_jsonl(multi_shared_plan_path)
        }
        row_ids = {str(row["id"]) for row in rows}
        missing_plans = sorted(row_ids.difference(multi_shared_plans))
        if missing_plans:
            raise ValueError(
                "multi-image shared Fine plan is missing manifest IDs: "
                f"{missing_plans[:5]}"
            )
        if any(not row.get("images") for row in rows):
            raise ValueError("shared Fine plan is valid only for multi-image rows")
    existing = read_jsonl(args.output) if args.resume and args.output.exists() else []
    expected_language_adapter = (
        str(language_lora_adapter_path)
        if language_lora_adapter_path is not None
        else None
    )
    if any(
        row.get("language_lora_adapter") != expected_language_adapter
        or (
            language_lora_adapter_audit is not None
            and row.get("language_lora_adapter_bundle_fingerprint")
            != language_lora_adapter_audit.bundle_fingerprint
        )
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced by a different recovery language LoRA"
        )
    expected_attention_backend = (
        deterministic_backend_name(
            args.deterministic_attention_backend,
            ptea_auxiliary_attention=args.evidence_policy == "mid_ptea",
        )
        if args.deterministic_inference
        else "sdpa_default"
    )
    if args.record_timing and any(not isinstance(row.get("timing"), dict) for row in existing):
        raise ValueError("cannot resume timing into output rows without timing metadata")
    if any(
        bool(row.get("deterministic_inference", False))
        != bool(args.deterministic_inference)
        for row in existing
    ):
        raise ValueError("cannot mix deterministic and default-SDPA rows while resuming")
    if any(
        row.get("attention_backend") != expected_attention_backend
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced by a different attention backend: "
            f"expected {expected_attention_backend}"
        )
    if any(
        row.get("ptea_question_control", "matched")
        != args.ptea_question_control
        or int(row.get("ptea_question_shuffle_seed") or 20260909)
        != int(args.ptea_question_shuffle_seed)
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced with a different PTEA question control"
        )
    if any(
        row.get("region_location_control", "matched")
        != args.region_location_control
        or int(row.get("region_location_control_seed") or 20260909)
        != int(args.region_location_control_seed)
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced with a different region-location control"
        )
    if any(
        bool(row.get("native_ratio_budget", False))
        != bool(args.native_ratio_budget)
        or float(row.get("native_global_ratio") or 0.0)
        != (
            float(args.native_global_ratio)
            if args.native_ratio_budget
            else 0.0
        )
        or float(row.get("native_fine_ratio") or 0.0)
        != (
            float(args.native_fine_ratio)
            if args.native_ratio_budget
            else 0.0
        )
        or bool(row.get("trace_split_adaptive_region_minimum", False))
        != bool(args.trace_split_adaptive_region_minimum)
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced with a different native-ratio or "
            "trace-split floor contract"
        )
    if any(
        bool(row.get("native_token_caps", False))
        != bool(args.native_token_caps)
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced with different Native-Cap semantics"
        )
    if any(
        bool(row.get("native_pixel_contract", False))
        != bool(args.native_pixel_contract)
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced with different NativeScale pixel semantics"
        )
    if any(
        bool(row.get("native_fuse", False)) != bool(args.native_fuse)
        or float(row.get("native_fuse_strength") or 1.0)
        != float(args.native_fuse_strength)
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced with a different NativeFuse contract"
        )
    if any(
        bool(row.get("patch_exchange", False)) != bool(args.patch_exchange)
        or bool(row.get("patch_exchange_tree", False))
        != bool(args.patch_exchange_tree)
        or float(row.get("patch_exchange_global_scale") or 2.0)
        != float(args.patch_exchange_global_scale)
        or float(row.get("patch_exchange_local_scale") or 2.0)
        != float(args.patch_exchange_local_scale)
        or float(row.get("patch_exchange_total_cap_ratio") or 1.0)
        != float(args.patch_exchange_total_cap_ratio)
        or int(row.get("patch_exchange_soft_floor_tokens") or 0)
        != int(args.patch_exchange_soft_floor_tokens)
        or bool(row.get("patch_exchange_balanced_soft_floor", False))
        != bool(args.patch_exchange_balanced_soft_floor)
        or bool(row.get("patch_exchange_preserve_native_global", False))
        != bool(args.patch_exchange_preserve_native_global)
        or bool(row.get("patch_exchange_continuous_soft_floor", False))
        != bool(args.patch_exchange_continuous_soft_floor)
        or int(
            row.get("patch_exchange_continuous_soft_floor_max_tokens") or 4096
        )
        != int(args.patch_exchange_continuous_soft_floor_max_tokens)
        or int(
            row.get("patch_exchange_continuous_soft_floor_ramp_start_tokens")
            or 4096
        )
        != int(args.patch_exchange_continuous_soft_floor_ramp_start_tokens)
        or int(
            row.get("patch_exchange_continuous_soft_floor_ramp_end_tokens")
            or 12288
        )
        != int(args.patch_exchange_continuous_soft_floor_ramp_end_tokens)
        or bool(row.get("patch_exchange_native_global_anchor", False))
        != bool(args.patch_exchange_native_global_anchor)
        or float(row.get("patch_exchange_additive_fine_ratio") or 0.5)
        != float(args.patch_exchange_additive_fine_ratio)
        or int(row.get("patch_exchange_additive_fine_max_tokens") or 2048)
        != int(args.patch_exchange_additive_fine_max_tokens)
        or int(row.get("patch_exchange_view_min_pixels") or 4096)
        != int(args.patch_exchange_view_min_pixels)
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced with a different PatchExchange contract"
        )
    if any(
        bool(row.get("continuous_native_global", False))
        != bool(args.continuous_native_global)
        or float(row.get("continuous_global_exponent") or 0.5)
        != float(args.continuous_global_exponent)
        or int(row.get("continuous_global_base_pixels") or 4 * 1024 * 1024)
        != int(args.continuous_global_base_pixels)
        or int(row.get("continuous_global_max_pixels") or 16 * 1024 * 1024)
        != int(args.continuous_global_max_pixels)
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced with a different continuous Global contract"
        )
    if any(
        row.get("visual_output_mode", "append") != args.visual_output_mode
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced with a different visual output mode"
        )
    if any(
        row.get("fine_readout_mode", "native") != args.fine_readout_mode
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced with a different Fine readout mode"
        )
    if args.native_pixel_contract and any(
        int(row.get("global_processor_max_pixels") or 0)
        != int(global_processor_max_pixels)
        or int(row.get("fine_processor_max_pixels") or 0)
        != int(fine_processor_max_pixels)
        for row in existing
    ):
        raise ValueError(
            "cannot resume NativeScale rows produced with different pixel maxima"
        )
    if any(
        bool(row.get("stage_d_v27b_safe_caps", False))
        != bool(args.stage_d_v27b_safe_caps)
        for row in existing
    ):
        raise ValueError("cannot resume rows with different Stage-D v27b caps")
    if any(
        bool(row.get("fine_patch_evidence_bias", False))
        != bool(args.fine_patch_evidence_bias)
        or float(row.get("fine_patch_evidence_floor", 0.5))
        != float(args.fine_patch_evidence_floor)
        or float(row.get("fine_patch_evidence_exponent", 1.0))
        != float(args.fine_patch_evidence_exponent)
        for row in existing
    ):
        raise ValueError("cannot resume rows with different Fine PatchBias semantics")
    expected_tracescale_checkpoint = (
        str(tracescale_head_path) if tracescale_head_path is not None else None
    )
    expected_adaptive_box_checkpoint = (
        str(adaptive_box_head_path) if adaptive_box_head_path is not None else None
    )
    expected_trace_refine_checkpoint = (
        str(trace_refine_policy_path)
        if trace_refine_policy_path is not None
        else None
    )
    if any(
        row.get("tracescale_checkpoint") != expected_tracescale_checkpoint
        or row.get("adaptive_box_checkpoint") != expected_adaptive_box_checkpoint
        or row.get("trace_refine_checkpoint") != expected_trace_refine_checkpoint
        or float(row.get("trace_refine_preview_expansion") or 1.5)
        != float(args.trace_refine_preview_expansion)
        or int(row.get("trace_refine_preview_tokens") or 256)
        != int(args.trace_refine_preview_tokens)
        or row.get("adaptive_box_safety_mode", "replace")
        != args.adaptive_box_safety_mode
        or row.get("adaptive_box_ensemble_heads", [])
        != [str(path) for path in adaptive_box_ensemble_paths]
        or (
            args.adaptive_box_safety_mode == "residual_read"
            and int(row.get("adaptive_box_residual_read_tokens") or 256)
            != int(args.adaptive_box_residual_read_tokens)
        )
        or (
            args.adaptive_box_safety_mode.startswith("uncertainty")
            and float(row.get("adaptive_box_uncertainty_threshold", math.inf))
            != float(args.adaptive_box_uncertainty_threshold)
        )
        or bool(row.get("tracescale_positive_only", False))
        != bool(args.tracescale_positive_only)
        or float(row.get("tracescale_strength", 1.0))
        != float(args.tracescale_strength)
        for row in existing
    ):
        raise ValueError("cannot resume rows produced with different TraceScale semantics")
    expected_anchored_calibrator = (
        str(anchored_growth_calibrator_path)
        if anchored_growth_calibrator_path is not None
        else None
    )
    if any(
        row.get("anchored_growth_calibrator") != expected_anchored_calibrator
        or float(row.get("anchored_growth_gate_threshold", 0.65))
        != float(args.anchored_growth_gate_threshold)
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced with different E2 calibrator semantics"
        )
    expected_evicontour_utility = (
        str(evicontour_utility_path) if evicontour_utility_path is not None else None
    )
    if any(
        row.get("evicontour_utility_checkpoint") != expected_evicontour_utility
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced with a different EviContour utility head"
        )
    expected_csr_config = (
        {
            "sufficiency_threshold": args.evicontour_csr_sufficiency_threshold,
            "sufficiency_tolerance": args.evicontour_csr_sufficiency_tolerance,
            "core_relevance_tolerance": args.evicontour_csr_core_relevance_tolerance,
            "context_ring_mass_ratio": args.evicontour_csr_context_ring_mass_ratio,
            "maximum_scope_hops": args.evicontour_csr_maximum_scope_hops,
            "minimum_parent_sufficiency_gain": (
                args.evicontour_csr_minimum_parent_sufficiency_gain
            ),
            "fine_floor_native_ratio": args.evicontour_csr_fine_floor_native_ratio,
            "minimum_fine_total": args.evicontour_csr_minimum_fine_total,
            "maximum_fine_total": args.evicontour_csr_maximum_fine_total,
        }
        if args.evidence_decoder == "evicontour_csr"
        else None
    )
    if any(row.get("evicontour_csr_config") != expected_csr_config for row in existing):
        raise ValueError("cannot resume rows with different EviContour-CSR semantics")
    expected_cap_scope = args.native_cap_scope if args.native_token_caps else None
    if any(
        (
            row.get("native_cap_scope")
            if row.get("native_cap_scope") is not None
            else ("both" if row.get("native_token_caps") else None)
        )
        != expected_cap_scope
        for row in existing
    ):
        raise ValueError("cannot resume rows produced with a different cap scope")
    if any(
        float(row.get("native_cap_fine_floor_native_ratio") or 0.0)
        != float(args.native_cap_fine_floor_native_ratio)
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced with a different continuous Fine floor"
        )
    if any(
        int(row.get("native_cap_minimum_global_tokens") or 0)
        != int(args.native_cap_minimum_global_tokens)
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced with a different Global topology floor"
        )
    if any(
        int(row.get("native_cap_anchor_fine_tokens") or 0)
        != int(args.native_cap_anchor_fine_tokens)
        for row in existing
    ):
        raise ValueError(
            "cannot resume rows produced with a different R1 anchor floor"
        )
    completed = {str(row["id"]) for row in existing}
    parent_remaining = [row for row in rows if str(row["id"]) not in completed]
    remaining: list[dict[str, Any]] = []
    for parent_row in parent_remaining:
        image_values = parent_row.get("images")
        if not image_values:
            remaining.append(parent_row)
            continue
        if parent_row.get("image") is not None:
            raise ValueError(
                f"multi-image row {parent_row['id']} contains both image and images"
            )
        shared_plan = (
            multi_shared_plans[str(parent_row["id"])]
            if multi_shared_plans
            else None
        )
        shared_budgets = (
            [int(value) for value in shared_plan["per_image_fine_budgets"]]
            if shared_plan is not None
            else None
        )
        if shared_budgets is not None and len(shared_budgets) != len(image_values):
            raise ValueError(
                f"shared Fine plan/image count mismatch for {parent_row['id']}"
            )
        for image_index, image_value in enumerate(image_values):
            child = dict(parent_row)
            child["image"] = image_value
            child["id"] = f"{parent_row['id']}::image_{image_index + 1}"
            child["_multi_parent_id"] = str(parent_row["id"])
            child["_multi_images"] = list(image_values)
            child["_multi_image_index"] = image_index
            child["_multi_image_count"] = len(image_values)
            if shared_plan is not None:
                child["_multi_shared_fine_budget"] = shared_budgets[image_index]
                child["_multi_shared_plan"] = shared_plan
            remaining.append(child)
    if not remaining and args.output.with_suffix(".summary.json").is_file():
        print(
            json.dumps(
                {
                    "resume_complete": True,
                    "rows": len(existing),
                    "output": str(args.output),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0

    from transformers import AutoProcessor

    processor_runtime_max_pixels = max(
        args.processor_max_pixels,
        global_processor_max_pixels,
        fine_processor_max_pixels,
        (
            args.continuous_global_max_pixels
            if args.continuous_native_global
            else 0
        ),
    )
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
        max_pixels=processor_runtime_max_pixels,
        min_pixels=args.processor_min_pixels,
    )
    trace_refine_preview_processor = None
    if trace_refine_policy_path is not None:
        # One merged Qwen token covers a 28x28 source-pixel cell.  This
        # processor is used only for the policy's context preview and never
        # changes the final VQA visual-token sequence.
        trace_refine_preview_processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
            min_pixels=16 * 28 * 28,
            max_pixels=int(args.trace_refine_preview_tokens * 28 * 28),
        )
    model, model_type = load_qwen_vlm(model_path, device=device)
    if language_lora_adapter_path is not None:
        # PEFT injects LoRA layers into the base object. Returning that object
        # keeps P3's exact ``model.model.visual`` access while LoRA stays active.
        model = attach_frozen_language_lora(model, language_lora_adapter_path)
    model.eval()
    model.requires_grad_(False)
    checkpoint_key_mapping_count = int(
        getattr(model, "_evivit_checkpoint_key_mapping_count", 0)
    )
    checkpoint_compatibility_base = getattr(
        model, "_evivit_checkpoint_compatibility_base", None
    )
    selector = None
    projection = None
    reliability_gate = None
    context_need_head = None
    tracescale_head = None
    tracescale_checkpoint = None
    adaptive_box_head = None
    adaptive_box_checkpoint = None
    adaptive_box_ensemble_heads = []
    trace_refine_policy = None
    trace_refine_checkpoint = None
    anchored_growth_calibrator = None
    anchored_growth_calibrator_checkpoint = None
    evicontour_utility = None
    evicontour_utility_checkpoint = None
    evitree_need_head = None
    evitree_need_checkpoint = None
    evitree_split_policy = None
    evitree_split_checkpoint = None
    evitree_split_threshold = None
    if args.evidence_policy == "mid_ptea":
        selector = load_selector(
            selector_checkpoint, int(model.config.vision_config.hidden_size), device
        ).eval()
        selector.requires_grad_(False)
        projection = build_projection(
            int(model.config.text_config.hidden_size),
            args.projection_dim,
            args.projection_seed,
            device,
        )
    if tracescale_head_path is not None:
        if not tracescale_head_path.is_file():
            raise FileNotFoundError(
                f"missing TraceScale checkpoint: {tracescale_head_path}"
            )
        tracescale_head, tracescale_checkpoint = load_tracescale_head(
            str(tracescale_head_path), device="cpu"
        )
    if adaptive_box_head_path is not None:
        if not adaptive_box_head_path.is_file():
            raise FileNotFoundError(
                f"missing AdaptiveBox checkpoint: {adaptive_box_head_path}"
            )
        adaptive_box_head, adaptive_box_checkpoint = load_adaptive_box_head(
            adaptive_box_head_path, device=device
        )
    for ensemble_path in adaptive_box_ensemble_paths:
        if not ensemble_path.is_file():
            raise FileNotFoundError(
                f"missing AdaptiveBox ensemble head: {ensemble_path}"
            )
        ensemble_head, _ = load_adaptive_box_head(ensemble_path, device=device)
        adaptive_box_ensemble_heads.append(ensemble_head)
    if trace_refine_policy_path is not None:
        if not trace_refine_policy_path.is_file():
            raise FileNotFoundError(
                f"missing TraceRefine checkpoint: {trace_refine_policy_path}"
            )
        trace_refine_policy, trace_refine_checkpoint = load_trace_refine_policy(
            trace_refine_policy_path, device=device
        )
    if anchored_growth_calibrator_path is not None:
        if not anchored_growth_calibrator_path.is_file():
            raise FileNotFoundError(
                "missing Anchored Growth calibrator checkpoint: "
                f"{anchored_growth_calibrator_path}"
            )
        (
            anchored_growth_calibrator,
            anchored_growth_calibrator_checkpoint,
        ) = load_anchored_growth_calibrator(
            str(anchored_growth_calibrator_path), device="cpu"
        )
    if evicontour_utility_path is not None:
        if not evicontour_utility_path.is_file():
            raise FileNotFoundError(
                f"missing EviContour utility checkpoint: {evicontour_utility_path}"
            )
        evicontour_utility, evicontour_utility_checkpoint = (
            load_evicontour_utility_head(
                str(evicontour_utility_path), map_location="cpu"
            )
        )
    if evitree_need_head_path is not None:
        if not evitree_need_head_path.is_file():
            raise FileNotFoundError(
                f"missing EviTree Need head: {evitree_need_head_path}"
            )
        evitree_need_head, evitree_need_checkpoint = (
            load_scalar_evidence_need_head(
                str(evitree_need_head_path), map_location=device
            )
        )
        evitree_need_head.to(device)
    if evitree_split_policy_path is not None:
        if not evitree_split_policy_path.is_file():
            raise FileNotFoundError(
                f"missing EviTree split policy: {evitree_split_policy_path}"
            )
        evitree_split_policy, evitree_split_checkpoint = (
            load_hierarchical_split_policy_head(
                str(evitree_split_policy_path), map_location=device
            )
        )
        evitree_split_policy.to(device)
        checkpoint_threshold = float(
            evitree_split_checkpoint["model_config"]["split_threshold"]
        )
        evitree_split_threshold = (
            float(args.evitree_split_threshold)
            if args.evitree_split_threshold >= 0
            else checkpoint_threshold
        )
    if context_need_checkpoint is not None:
        if not context_need_checkpoint.is_file():
            raise FileNotFoundError(
                f"missing context-need checkpoint: {context_need_checkpoint}"
            )
        context_need_head = load_context_need_head(
            context_need_checkpoint,
            device=device,
        )
    if args.v10_gate_mode in {"learned", "entropy_only"}:
        if reliability_report_path is None or not reliability_report_path.is_file():
            raise FileNotFoundError(
                f"missing v10 reliability report: {reliability_report_path}"
            )
        reliability_report = json.loads(
            reliability_report_path.read_text(encoding="utf-8")
        )
        reliability_gate = ReliabilityGate.from_report(
            reliability_report,
            artifact=(
                "multifeature"
                if args.v10_gate_mode == "learned"
                else "entropy_only"
            ),
        )
    bridge, checkpoint_metadata = bridge_from_checkpoint(
        bridge_checkpoint,
        model=model,
        device=device,
        default_insertion_block=args.insertion_block,
        default_bridge_dim=args.bridge_dim,
        default_bridge_heads=args.bridge_heads,
        default_neighborhood_radius=args.neighborhood_radius,
        default_max_relative_residual=(
            args.max_relative_residual if args.max_relative_residual > 0 else None
        ),
        coverage_masked_parent_writeback=(
            args.coverage_masked_parent_writeback
        ),
    )
    bridge_mode = str(
        args.bridge_mode
        or checkpoint_metadata["checkpoint_config"].get(
            "bridge_mode", "bidirectional"
        )
    )
    if bridge_mode not in {
        "bidirectional",
        "preserve_global",
        "evidence_weighted_writeback",
        "preserve_fine",
        "identity",
    }:
        raise ValueError(f"unknown checkpoint bridge_mode: {bridge_mode}")
    map_feature_blocks = (
        tuple(
            int(value)
            for value in getattr(selector, "required_visual_blocks", ())
        )
        if selector is not None
        else ()
    )
    if isinstance(bridge, EvidenceSlotCompressor):
        encoder: torch.nn.Module = Qwen3VLEviSlotEncoder(
            model.model.visual,
            bridge,
            insertion_block=checkpoint_metadata["insertion_block"],
            freeze_vision=True,
            map_feature_blocks=map_feature_blocks,
            token_serialization=args.token_serialization,
        ).eval()
    else:
        encoder = Qwen3VLEviViTV3MidEncoder(
            model.model.visual,
            bridge,
            insertion_block=checkpoint_metadata["insertion_block"],
            freeze_vision=True,
            map_feature_blocks=map_feature_blocks,
            token_serialization=args.token_serialization,
            progressive_bridge_blocks=tuple(
                checkpoint_metadata["checkpoint_config"].get(
                    "progressive_bridge_blocks", ()
                )
            ),
            global_frame_joint_blocks=tuple(args.global_frame_joint_blocks),
            global_frame_coordinate_bins=args.global_frame_coordinate_bins,
            global_frame_joint_topology=args.global_frame_joint_topology,
            safe_joint_residual_mixer=checkpoint_metadata[
                "safe_joint_residual_mixer"
            ],
            safe_joint_residual_blocks=(
                (
                    int(
                        checkpoint_metadata["checkpoint_config"].get(
                            "safe_joint_block", 20
                        )
                    ),
                )
                if checkpoint_metadata["safe_joint_residual_mixer"] is not None
                else ()
            ),
            trace_calibrated_bridge_residual=checkpoint_metadata[
                "trace_calibrated_bridge_residual"
            ],
        ).eval()
    merge_size = int(model.config.vision_config.spatial_merge_size)
    merged_stride = merge_size * int(model.config.vision_config.patch_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if existing and args.resume else "w"
    output_rows = list(existing)
    multi_visual_accumulator: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    with args.output.open(mode, encoding="utf-8") as handle, torch.inference_mode():
        for index, row in enumerate(remaining, 1):
            if args.record_timing:
                torch.cuda.synchronize(0)
            request_started = time.perf_counter()
            image_path = Path(str(row["image"]))
            if not image_path.is_absolute():
                image_path = project_root / image_path
            source = Image.open(image_path).convert("RGB")
            native_visual_tokens: int | None = None
            native_grid_thw: list[int] | None = None
            native_ratio_plan = None
            native_need_ratio_plan = None
            native_cap_global_plan = None
            native_pixel_global_plan = None
            continuous_global_plan = None
            patch_exchange_global_plan = None
            patch_exchange_fine_plan = None
            effective_patch_exchange_soft_floor_tokens = int(
                args.patch_exchange_soft_floor_tokens
            )
            if args.patch_exchange:
                patch_exchange_global_plan = plan_patch_exchange_global_view(
                    source.width,
                    source.height,
                    global_scale=args.patch_exchange_global_scale,
                    minimum_pixels=args.processor_min_pixels,
                    maximum_pixels=args.processor_max_pixels,
                    patch_size=int(model.config.vision_config.patch_size),
                    merge_size=merge_size,
                    soft_floor_tokens=(
                        args.patch_exchange_soft_floor_tokens
                        if args.patch_exchange_balanced_soft_floor
                        else 0
                    ),
                    soft_floor_global_share=(
                        1.0 / (args.patch_exchange_global_scale**2)
                        if args.patch_exchange_balanced_soft_floor
                        else 0.0
                    ),
                    preserve_native_global=(
                        args.patch_exchange_preserve_native_global
                    ),
                    native_global_anchor=(
                        args.patch_exchange_native_global_anchor
                    ),
                )
                if args.patch_exchange_continuous_soft_floor:
                    effective_patch_exchange_soft_floor_tokens = (
                        continuous_soft_floor_tokens(
                            patch_exchange_global_plan.native_plan.realized_tokens,
                            minimum_tokens=args.patch_exchange_soft_floor_tokens,
                            maximum_tokens=(
                                args.patch_exchange_continuous_soft_floor_max_tokens
                            ),
                            ramp_start_tokens=(
                                args.patch_exchange_continuous_soft_floor_ramp_start_tokens
                            ),
                            ramp_end_tokens=(
                                args.patch_exchange_continuous_soft_floor_ramp_end_tokens
                            ),
                        )
                    )
                    patch_exchange_global_plan = plan_patch_exchange_global_view(
                        source.width,
                        source.height,
                        global_scale=args.patch_exchange_global_scale,
                        minimum_pixels=args.processor_min_pixels,
                        maximum_pixels=args.processor_max_pixels,
                        patch_size=int(model.config.vision_config.patch_size),
                        merge_size=merge_size,
                        soft_floor_tokens=(
                            effective_patch_exchange_soft_floor_tokens
                        ),
                        soft_floor_global_share=(
                            1.0 / (args.patch_exchange_global_scale**2)
                        ),
                        preserve_native_global=(
                            args.patch_exchange_preserve_native_global
                        ),
                        native_global_anchor=(
                            args.patch_exchange_native_global_anchor
                        ),
                    )
                native_visual_tokens = (
                    patch_exchange_global_plan.native_plan.realized_tokens
                )
                native_grid_thw = [
                    1,
                    patch_exchange_global_plan.native_plan.resized_height
                    // int(model.config.vision_config.patch_size),
                    patch_exchange_global_plan.native_plan.resized_width
                    // int(model.config.vision_config.patch_size),
                ]
            elif args.native_pixel_contract:
                if args.continuous_native_global:
                    continuous_global_plan = plan_continuous_native_global_view(
                        source.width,
                        source.height,
                        minimum_pixels=args.processor_min_pixels,
                        base_pixels=args.continuous_global_base_pixels,
                        maximum_pixels=args.continuous_global_max_pixels,
                        exponent=args.continuous_global_exponent,
                        patch_size=int(model.config.vision_config.patch_size),
                        merge_size=merge_size,
                    )
                    native_pixel_global_plan = continuous_global_plan.view_plan
                else:
                    native_pixel_global_plan = plan_native_pixel_view(
                        source.width,
                        source.height,
                        minimum_pixels=args.processor_min_pixels,
                        maximum_pixels=global_processor_max_pixels,
                        patch_size=int(model.config.vision_config.patch_size),
                        merge_size=merge_size,
                    )
                native_visual_tokens = native_pixel_global_plan.realized_tokens
                native_grid_thw = [
                    1,
                    native_pixel_global_plan.resized_height
                    // int(model.config.vision_config.patch_size),
                    native_pixel_global_plan.resized_width
                    // int(model.config.vision_config.patch_size),
                ]
            elif args.native_ratio_budget:
                native_ratio_plan = plan_native_ratio_budget(
                    source.width,
                    source.height,
                    global_ratio=args.native_global_ratio,
                    fine_ratio=args.native_fine_ratio,
                    minimum_global_tokens=args.native_ratio_min_global_tokens,
                    minimum_fine_tokens=args.native_ratio_min_fine_tokens,
                    alignment=args.native_ratio_alignment,
                    minimum_pixels=args.processor_min_pixels,
                    maximum_pixels=args.processor_max_pixels,
                    patch_size=int(model.config.vision_config.patch_size),
                    merge_size=merge_size,
                )
            elif args.native_need_ratio_budget:
                native_need_ratio_plan = plan_native_need_spread_budget(
                    source.width,
                    source.height,
                    evidence_need=0.0,
                    context_need=0.0,
                    minimum_global_ratio=args.native_global_ratio,
                    maximum_global_ratio=args.native_global_ratio,
                    minimum_fine_ratio=args.native_min_fine_ratio,
                    maximum_fine_ratio=args.native_max_fine_ratio,
                    minimum_global_tokens=args.native_ratio_min_global_tokens,
                    minimum_fine_tokens=args.native_ratio_min_fine_tokens,
                    maximum_global_tokens=args.native_ratio_max_global_tokens,
                    maximum_fine_tokens=args.native_ratio_max_fine_tokens,
                    soft_floor_tokens=args.native_ratio_soft_floor_tokens,
                    alignment=args.native_ratio_alignment,
                    minimum_pixels=args.processor_min_pixels,
                    maximum_pixels=args.processor_max_pixels,
                    patch_size=int(model.config.vision_config.patch_size),
                    merge_size=merge_size,
                )
            active_native_plan = native_ratio_plan or native_need_ratio_plan
            if active_native_plan is not None:
                native_visual_tokens = active_native_plan.native_tokens
                native_grid_thw = [
                    1,
                    active_native_plan.resized_height
                    // int(model.config.vision_config.patch_size),
                    active_native_plan.resized_width
                    // int(model.config.vision_config.patch_size),
                ]
            elif native_global_cap:
                native_cap_global_plan = plan_native_token_cap(
                    source.width,
                    source.height,
                    token_cap=args.global_token_budget,
                    minimum_tokens=args.native_cap_minimum_global_tokens,
                    minimum_pixels=args.processor_min_pixels,
                    maximum_pixels=args.processor_max_pixels,
                    patch_size=int(model.config.vision_config.patch_size),
                    merge_size=merge_size,
                )
                native_visual_tokens = native_cap_global_plan.native_tokens
                native_grid_thw = [
                    1,
                    native_cap_global_plan.native_resized_height
                    // int(model.config.vision_config.patch_size),
                    native_cap_global_plan.native_resized_width
                    // int(model.config.vision_config.patch_size),
                ]
            elif (
                native_fine_cap
                and args.native_cap_fine_floor_native_ratio > 0
            ):
                native_cap_source_plan = plan_native_token_cap(
                    source.width,
                    source.height,
                    token_cap=args.global_token_budget,
                    minimum_tokens=0,
                    minimum_pixels=args.processor_min_pixels,
                    maximum_pixels=args.processor_max_pixels,
                    patch_size=int(model.config.vision_config.patch_size),
                    merge_size=merge_size,
                )
                native_visual_tokens = native_cap_source_plan.native_tokens
                native_grid_thw = [
                    1,
                    native_cap_source_plan.native_resized_height
                    // int(model.config.vision_config.patch_size),
                    native_cap_source_plan.native_resized_width
                    // int(model.config.vision_config.patch_size),
                ]
            if args.native_fit_total_token_budget:
                native_messages = qa_messages(
                    str(row["question"]), source, prompt_variant="canonical"
                )
                native_template_kwargs = {
                    "tokenize": True,
                    "add_generation_prompt": True,
                    "return_dict": True,
                    "return_tensors": "pt",
                }
                if model_type == "qwen3_5":
                    native_template_kwargs["enable_thinking"] = not args.disable_thinking
                native_inputs = processor.apply_chat_template(
                    native_messages, **native_template_kwargs
                ).to(device)
                native_grids = native_inputs["image_grid_thw"].detach().cpu().tolist()
                if len(native_grids) != 1:
                    raise RuntimeError(
                        f"native-fit expected one image grid, got {native_grids}"
                    )
                native_grid_thw = [int(value) for value in native_grids[0]]
                native_visual_tokens = (
                    native_grid_thw[0]
                    * (native_grid_thw[1] // merge_size)
                    * (native_grid_thw[2] // merge_size)
                )
                if native_visual_tokens <= args.native_fit_total_token_budget:
                    generated = model.generate(
                        **native_inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                    trimmed = [
                        output[len(input_ids) :]
                        for input_ids, output in zip(native_inputs.input_ids, generated)
                    ]
                    raw = processor.batch_decode(
                        trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]
                if args.answer_protocol == "muir_mcq":
                    prediction, extracted, parse_error = parse_muir_choice(raw)
                elif args.answer_protocol == "blink_mcq":
                    prediction, extracted, parse_error = parse_blink_choice(raw)
                else:
                    prediction, extracted, parse_error = parse_answer(raw)
                    target = str(row.get("answer", ""))
                    output = {
                        "id": row["id"],
                        "eval_tier": row.get("eval_tier"),
                        "image": row.get("image"),
                        "question": row.get("question"),
                        "answer": target,
                        "raw_prediction": raw,
                        "extracted_prediction": extracted,
                        "prediction": prediction,
                        "normalized_prediction": normalized_answer(prediction),
                        "normalized_answer": normalized_answer(target),
                        "relaxed_prediction": relaxed_answer(prediction),
                        "relaxed_answer": relaxed_answer(target),
                        "exact_correct": normalized_answer(prediction)
                        == normalized_answer(target),
                        "relaxed_correct": relaxed_answer_match(prediction, target),
                        "parse_error": parse_error,
                        "method": "evivit_v7_native_fit_base",
                        "model": str(model_path),
                        "model_index_sha256": model_index_sha256,
                        "evidence_policy": "native_fit_bypass",
                        "evidence_decoder": "native_fit_bypass",
                        "region_expansion_factor": args.region_expansion_factor,
                        "topology_union_factor": args.topology_union_factor,
                        "token_serialization": "native_qwen",
                        "serialization_moved_tokens": 0,
                        "control_seed": args.control_seed,
                        "checkpoint": checkpoint_metadata["checkpoint"],
                        "checkpoint_step": checkpoint_metadata["checkpoint_step"],
                        "insertion_block": checkpoint_metadata["insertion_block"],
                        "global_token_budget": args.global_token_budget,
                        "fine_token_budget": args.fine_token_budget,
                        "native_fit_total_token_budget": args.native_fit_total_token_budget,
                        "native_fit_bypass": True,
                        "native_visual_tokens": native_visual_tokens,
                        "native_grid_thw": native_grid_thw,
                        "realized_global_tokens": native_visual_tokens,
                        "realized_fine_tokens": 0,
                        "fine_view_token_counts": [],
                        "view_token_counts": [native_visual_tokens],
                        "realized_visual_tokens": native_visual_tokens,
                        "source_images": 1,
                        "single_visual_span": True,
                        "selector_question_protocol": "native_fit_no_selector",
                        "ptea_question_control": args.ptea_question_control,
                        "ptea_question_shuffle_seed": args.ptea_question_shuffle_seed,
                        "ptea_question_control_source_id": None,
                        "region_location_control": args.region_location_control,
                        "region_location_control_seed": args.region_location_control_seed,
                        "region_location_control_plans": [],
                        "regions": 0,
                        "region_plans": [],
                        "map_entropy": None,
                        "map_peak_probability": None,
                        "relative_residual_l2": 0.0,
                        "global_relative_residual_l2": 0.0,
                        "fine_relative_residual_l2": 0.0,
                        "deterministic_inference": args.deterministic_inference,
                        "attention_backend": (
                            deterministic_backend_name(
                                args.deterministic_attention_backend,
                                ptea_auxiliary_attention=False,
                            )
                            if args.deterministic_inference
                            else "sdpa_default"
                        ),
                    }
                    output_rows.append(output)
                    handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(
                        json.dumps(
                            {
                                "progress": index,
                                "total": len(remaining),
                                "id": output["id"],
                                "prediction": prediction,
                                "answer": target,
                                "exact": output["exact_correct"],
                                "relaxed": output["relaxed_correct"],
                                "visual_tokens": native_visual_tokens,
                                "native_fit_bypass": True,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    source.close()
                    del native_inputs, generated
                    continue
                del native_inputs
            sample_global_token_budget = args.global_token_budget
            sample_fine_token_budget = args.fine_token_budget
            multi_shared_fine_budget = (
                int(row["_multi_shared_fine_budget"])
                if row.get("_multi_shared_fine_budget") is not None
                else None
            )
            if native_ratio_plan is not None:
                sample_global_token_budget = native_ratio_plan.global_tokens
                sample_fine_token_budget = native_ratio_plan.fine_tokens
            elif native_need_ratio_plan is not None:
                sample_global_token_budget = native_need_ratio_plan.global_tokens
                sample_fine_token_budget = native_need_ratio_plan.fine_tokens
            elif native_cap_global_plan is not None:
                sample_global_token_budget = (
                    native_cap_global_plan.requested_tokens
                )
            elif args.adaptive_native_global_budget:
                sample_global_token_budget = continuous_native_global_token_budget(
                    source.width,
                    source.height,
                    base_tokens=args.adaptive_global_base_token_budget,
                    native_fraction=args.adaptive_global_native_fraction,
                    minimum_tokens=args.adaptive_global_min_token_budget,
                    maximum_tokens=args.adaptive_global_max_token_budget,
                    maximum_pixels=args.adaptive_global_maximum_pixels,
                    merged_patch_stride=merged_stride,
                    alignment=args.adaptive_global_budget_alignment,
                )
            if patch_exchange_global_plan is not None:
                global_h = patch_exchange_global_plan.grid_height
                global_w = patch_exchange_global_plan.grid_width
            elif native_pixel_global_plan is not None:
                global_h = native_pixel_global_plan.grid_height
                global_w = native_pixel_global_plan.grid_width
            elif native_cap_global_plan is not None:
                global_h = native_cap_global_plan.grid_height
                global_w = native_cap_global_plan.grid_width
            else:
                global_h, global_w = grid_for_token_budget(
                    source.width, source.height, sample_global_token_budget
                )
            global_view = (
                source
                if args.native_global_mrope
                else resize_to_token_grid(
                    source, global_h, global_w, merged_patch_stride=merged_stride
                )
            )
            global_inputs = processor(
                text=[vision_probe_text(processor, model_type)],
                images=[global_view], return_tensors="pt"
            ).to(device)
            prepared = encoder.prepare_global(
                global_inputs["pixel_values"], global_inputs["image_grid_thw"]
            )
            if native_pixel_global_plan is not None and (
                prepared.map_feature_grid.shape[0]
                * prepared.map_feature_grid.shape[1]
                != native_pixel_global_plan.realized_tokens
            ):
                raise AssertionError(
                    "evaluation Global grid differs from Native-Pixel plan"
                )
            if args.record_timing:
                torch.cuda.synchronize(0)
                global_pass_finished = time.perf_counter()
            question_tokens = None
            selector_question = str(
                row.get("_ptea_control_question")
                or row.get("selector_question")
                or row["question"]
            )
            sample_evidence_need = None
            if args.evidence_policy == "mid_ptea":
                question_tokens = qwen_projected_question_tokens(
                    model,
                    processor.tokenizer,
                    selector_question,
                    projection,
                    device=device,
                )
                if args.ptea_question_control == "zero":
                    question_tokens = torch.zeros_like(question_tokens)
                selector_backend = (
                    "math"
                    if args.deterministic_inference
                    and args.deterministic_attention_backend == "flash"
                    else None
                )
                if args.v10_gate_mode == "disabled":
                    allocation = allocate_mid_ptea_evidence(
                        selector,
                        prepared.selector_visual_input(selector),
                        question_tokens,
                        fine_token_budget=sample_fine_token_budget,
                        max_regions=args.max_regions,
                        minimum_region_tokens=args.minimum_region_tokens,
                        evidence_decoder=args.evidence_decoder,
                        region_expansion_factor=args.region_expansion_factor,
                        topology_union_factor=args.topology_union_factor,
                        selector_attention_backend=selector_backend,
                        trace_split_context_fraction=(
                            args.trace_split_context_fraction
                        ),
                        trace_split_context_expansion_factor=(
                            args.trace_split_context_expansion_factor
                        ),
                        trace_split_context_decoder=(
                            args.trace_split_context_decoder
                        ),
                        trace_split_budget_mode=args.trace_split_budget_mode,
                        trace_split_min_context_fraction=(
                            args.trace_split_min_context_fraction
                        ),
                        trace_split_max_context_fraction=(
                            args.trace_split_max_context_fraction
                        ),
                        trace_split_context_need_head=context_need_head,
                        trace_split_adaptive_minimum_region_tokens=(
                            args.trace_split_adaptive_region_minimum
                        ),
                        contour_relative_threshold=(
                            args.contour_relative_threshold
                        ),
                        contour_context_margin=args.contour_context_margin,
                        contour_minimum_residual_mass=(
                            args.contour_minimum_residual_mass
                        ),
                        seeded_basin_config=seeded_basin_config,
                        anchored_growth_calibrator=anchored_growth_calibrator,
                        anchored_growth_gate_threshold=(
                            args.anchored_growth_gate_threshold
                        ),
                        evicontour_utility=evicontour_utility,
                        evicontour_image_size=[source.width, source.height],
                        evicontour_question=selector_question,
                        evicontour_csr_sufficiency_threshold=(
                            args.evicontour_csr_sufficiency_threshold
                        ),
                        evicontour_csr_sufficiency_tolerance=(
                            args.evicontour_csr_sufficiency_tolerance
                        ),
                        evicontour_csr_core_relevance_tolerance=(
                            args.evicontour_csr_core_relevance_tolerance
                        ),
                        evicontour_csr_context_ring_mass_ratio=(
                            args.evicontour_csr_context_ring_mass_ratio
                        ),
                        evicontour_csr_maximum_scope_hops=(
                            args.evicontour_csr_maximum_scope_hops
                        ),
                        evicontour_csr_minimum_parent_sufficiency_gain=(
                            args.evicontour_csr_minimum_parent_sufficiency_gain
                        ),
                    )
                    if evitree_need_head is not None:
                        feature_values = scalar_need_feature_vector(
                            allocation.probability_map.detach()
                            .float()
                            .cpu()
                            .numpy(),
                            width=source.width,
                            height=source.height,
                            question=selector_question,
                        )
                        feature_tensor = torch.tensor(
                            feature_values,
                            dtype=torch.float32,
                            device=device,
                        ).unsqueeze(0)
                        with torch.inference_mode():
                            sample_evidence_need = float(
                                evitree_need_head(feature_tensor).item()
                            )
                        if args.native_need_ratio_budget:
                            native_need_ratio_plan = plan_native_need_spread_budget(
                                source.width,
                                source.height,
                                evidence_need=sample_evidence_need,
                                context_need=0.0,
                                minimum_global_ratio=args.native_global_ratio,
                                maximum_global_ratio=args.native_global_ratio,
                                minimum_fine_ratio=args.native_min_fine_ratio,
                                maximum_fine_ratio=args.native_max_fine_ratio,
                                minimum_global_tokens=(
                                    args.native_ratio_min_global_tokens
                                ),
                                minimum_fine_tokens=(
                                    args.native_ratio_min_fine_tokens
                                ),
                                maximum_global_tokens=(
                                    args.native_ratio_max_global_tokens
                                ),
                                maximum_fine_tokens=(
                                    args.native_ratio_max_fine_tokens
                                ),
                                soft_floor_tokens=(
                                    args.native_ratio_soft_floor_tokens
                                ),
                                alignment=args.native_ratio_alignment,
                                minimum_pixels=args.processor_min_pixels,
                                maximum_pixels=args.processor_max_pixels,
                                patch_size=int(
                                    model.config.vision_config.patch_size
                                ),
                                merge_size=merge_size,
                            )
                            sample_fine_token_budget = (
                                native_need_ratio_plan.fine_tokens
                            )
                        else:
                            sample_fine_token_budget = continuous_fine_token_budget(
                                sample_evidence_need,
                                minimum_tokens=args.evitree_min_fine_token_budget,
                                maximum_tokens=args.evitree_max_fine_token_budget,
                                alignment=args.evitree_budget_alignment,
                            )
                        if not args.evitree_tree_allocation:
                            allocation = reallocate_probability_evidence(
                                allocation.probability_map,
                                fine_token_budget=sample_fine_token_budget,
                                max_regions=args.max_regions,
                                minimum_region_tokens=args.minimum_region_tokens,
                                evidence_decoder=args.evidence_decoder,
                                region_expansion_factor=args.region_expansion_factor,
                                topology_union_factor=args.topology_union_factor,
                                seeded_basin_config=seeded_basin_config,
                                anchored_growth_calibrator=(
                                    anchored_growth_calibrator
                                ),
                                anchored_growth_gate_threshold=(
                                    args.anchored_growth_gate_threshold
                                ),
                            )
                            allocation.requested_fine_token_budget = (
                                sample_fine_token_budget
                            )
                            allocation.evitree_evidence_need = (
                                sample_evidence_need
                            )
                    if args.evitree_tree_allocation:
                        tree_fine_budget = (
                            sample_fine_token_budget
                            if evitree_need_head is not None
                            else sample_fine_token_budget
                        )
                        tree_probability = allocation.probability_map
                        allocation = allocate_evitree_hierarchical_evidence(
                            tree_probability,
                            fine_token_budget=tree_fine_budget,
                            minimum_leaf_tokens=args.minimum_region_tokens,
                            maximum_depth=args.evitree_tree_max_depth,
                            maximum_leaves=args.evitree_tree_max_leaves,
                            reliability=args.evitree_tree_reliability,
                            target_mass=args.evitree_tree_target_mass,
                            evidence_need=sample_evidence_need,
                            split_policy=evitree_split_policy,
                            split_threshold=evitree_split_threshold,
                            sparse_branches=args.evitree_sparse_branches,
                            maximum_branches=args.evitree_maximum_branches,
                            minimum_branch_depth=(
                                args.evitree_minimum_branch_depth
                            ),
                            branch_density_power=(
                                args.evitree_branch_density_power
                            ),
                            recenter_branch_tips=(
                                args.evitree_recenter_branch_tips
                            ),
                            branch_expansion_factor=(
                                args.evitree_branch_expansion_factor
                            ),
                            maximum_branch_iou=(
                                args.evitree_maximum_branch_iou
                            ),
                        )
                else:
                    if getattr(selector, "required_visual_blocks", ()):
                        raise ValueError(
                            "v10 reliability gate is calibrated for the "
                            "single-scale PTEA only"
                        )
                    allocation = allocate_reliability_calibrated_evidence(
                        selector,
                        prepared.map_feature_grid,
                        question_tokens,
                        question=selector_question,
                        gate_mode=args.v10_gate_mode,
                        reliability_gate=reliability_gate,
                        minimum_fine_token_budget=(
                            0
                            if args.v10_gate_mode == "zero"
                            else args.v10_min_fine_token_budget
                        ),
                        maximum_fine_token_budget=args.fine_token_budget,
                        max_regions=args.max_regions,
                        minimum_region_tokens=args.minimum_region_tokens,
                        evidence_decoder=args.evidence_decoder,
                        region_expansion_factor=args.region_expansion_factor,
                        topology_union_factor=args.topology_union_factor,
                        selector_attention_backend=selector_backend,
                        soften_probability=args.v10_map_mode == "soft",
                        dynamic_fine_budget=args.v10_budget_mode == "dynamic",
                    )
            else:
                digest = hashlib.sha256(str(row["id"]).encode("utf-8")).digest()
                sample_seed = (
                    args.control_seed + int.from_bytes(digest[:8], "big")
                ) % (2**63 - 1)
                allocation = allocate_control_evidence(
                    prepared.map_feature_grid,
                    policy=args.evidence_policy,
                    fine_token_budget=sample_fine_token_budget,
                    max_regions=args.max_regions,
                    minimum_region_tokens=args.minimum_region_tokens,
                    random_seed=sample_seed,
                )
            forced_context_bbox: tuple[int, int, int, int] | None = None
            forced_context_applied = False
            if args.forced_context_bbox_field:
                forced_value = row.get(args.forced_context_bbox_field)
                if (
                    args.forced_context_bbox_field not in row
                    or forced_value is None
                ) and not args.forced_context_bbox_optional:
                    raise KeyError(
                        f"{row['id']}: missing forced context field "
                        f"{args.forced_context_bbox_field!r}"
                    )
                if forced_value is not None:
                    forced_context_bbox = _forced_policy_box(
                        forced_value,
                        field=args.forced_context_bbox_field,
                        row_id=str(row["id"]),
                    )
                    forced_context_applied = replace_trace_split_context_box(
                        allocation, forced_context_bbox
                    )
            context_bbox_before_hsafe = (
                list(allocation.region_budgets[-1].bbox)
                if allocation.region_budgets
                else None
            )
            if args.record_timing:
                torch.cuda.synchronize(0)
                allocation_finished = time.perf_counter()
            adaptive_box_plans: list[dict[str, Any]] = []
            adaptive_residual_budgets: list[RegionBudget] = []
            adaptive_features: torch.Tensor | None = None
            if adaptive_box_head is not None:
                adaptive_features = adaptive_box_feature_vectors(
                    selector,
                    prepared.selector_visual_input(selector),
                    question_tokens,
                    allocation,
                    image_size=[source.width, source.height],
                    attention_backend=selector_backend,
                ).to(device)
                with torch.inference_mode():
                    adaptive_residuals = adaptive_box_head(adaptive_features)
                    anchor_boxes = torch.tensor(
                        [
                            [float(value) / 1000.0 for value in budget.bbox]
                            for budget in allocation.region_budgets
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                    adaptive_boxes = apply_residual(
                        anchor_boxes,
                        adaptive_residuals,
                        minimum_side=adaptive_box_head.config.minimum_side,
                        maximum_side=adaptive_box_head.config.maximum_side,
                    )
                    adaptive_uncertainty = None
                    adaptive_accepted = torch.ones(
                        len(anchor_boxes), dtype=torch.bool, device=device
                    )
                    if adaptive_box_ensemble_heads:
                        ensemble_predictions = torch.stack(
                            [
                                head(adaptive_features)
                                for head in adaptive_box_ensemble_heads
                            ],
                            dim=0,
                        )
                        adaptive_uncertainty = residual_ensemble_uncertainty(
                            ensemble_predictions
                        )
                        adaptive_accepted = (
                            adaptive_uncertainty
                            <= args.adaptive_box_uncertainty_threshold
                        )
                        adaptive_boxes = torch.where(
                            adaptive_accepted[:, None],
                            adaptive_boxes,
                            anchor_boxes,
                        )
                    if args.adaptive_box_safety_mode in {
                        "anchor_union",
                        "uncertainty_union",
                    }:
                        adaptive_boxes = union_boxes(anchor_boxes, adaptive_boxes)
                    adaptive_boxes = adaptive_boxes.cpu()
                    adaptive_uncertainty_cpu = (
                        adaptive_uncertainty.detach().cpu().tolist()
                        if adaptive_uncertainty is not None
                        else [None] * len(anchor_boxes)
                    )
                    adaptive_accepted_cpu = (
                        adaptive_accepted.detach().cpu().tolist()
                    )
                adjusted_budgets = []
                for budget, refined, residual, uncertainty, accepted in zip(
                    allocation.region_budgets,
                    adaptive_boxes.tolist(),
                    adaptive_residuals.detach().cpu().tolist(),
                    adaptive_uncertainty_cpu,
                    adaptive_accepted_cpu,
                ):
                    adjusted_policy_box = tuple(
                        int(round(float(value) * 1000.0)) for value in refined
                    )
                    if args.adaptive_box_safety_mode == "residual_read":
                        adjusted_budgets.append(budget)
                        if adjusted_policy_box != budget.bbox:
                            adaptive_residual_budgets.append(
                                RegionBudget(
                                    bbox=adjusted_policy_box,
                                    token_budget=args.adaptive_box_residual_read_tokens,
                                    score=budget.score,
                                    role=f"adaptive_residual_{budget.role}",
                                    source_rank=budget.source_rank,
                                )
                            )
                    else:
                        adjusted_budgets.append(
                            RegionBudget(
                                bbox=adjusted_policy_box,
                                token_budget=budget.token_budget,
                                score=budget.score,
                                role=budget.role,
                                source_rank=budget.source_rank,
                            )
                        )
                    adaptive_box_plans.append(
                        {
                            "original_policy_bbox": list(budget.bbox),
                            "adjusted_policy_bbox": list(adjusted_policy_box),
                            "residual": [float(value) for value in residual],
                            "uncertainty": (
                                float(uncertainty)
                                if uncertainty is not None
                                else None
                            ),
                            "accepted": bool(accepted),
                            "safety_mode": args.adaptive_box_safety_mode,
                            "residual_read_tokens": (
                                args.adaptive_box_residual_read_tokens
                                if args.adaptive_box_safety_mode == "residual_read"
                                else None
                            ),
                        }
                    )
                allocation.region_budgets = adjusted_budgets
            trace_refine_plan: dict[str, Any] | None = None
            if trace_refine_policy is not None:
                if adaptive_features is None or trace_refine_preview_processor is None:
                    raise RuntimeError("TraceRefine requires online H-Safe features")
                anchor_budgets = list(allocation.region_budgets)
                preview_rows: list[torch.Tensor] = []
                expanded_boxes: list[list[int]] = []
                for budget in anchor_budgets:
                    x1, y1, x2, y2 = (float(value) for value in budget.bbox)
                    width, height = x2 - x1, y2 - y1
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    expanded_width = min(
                        1000.0, width * args.trace_refine_preview_expansion
                    )
                    expanded_height = min(
                        1000.0, height * args.trace_refine_preview_expansion
                    )
                    cx = min(
                        1000.0 - expanded_width / 2.0,
                        max(expanded_width / 2.0, cx),
                    )
                    cy = min(
                        1000.0 - expanded_height / 2.0,
                        max(expanded_height / 2.0, cy),
                    )
                    expanded = [
                        int(round(cx - expanded_width / 2.0)),
                        int(round(cy - expanded_height / 2.0)),
                        int(round(cx + expanded_width / 2.0)),
                        int(round(cy + expanded_height / 2.0)),
                    ]
                    expanded[2] = max(expanded[0] + 1, expanded[2])
                    expanded[3] = max(expanded[1] + 1, expanded[3])
                    preview_crop = source.crop(
                        policy_to_pixels(expanded, source.width, source.height)
                    ).convert("RGB")
                    preview_inputs = trace_refine_preview_processor(
                        text=[vision_probe_text(processor, model_type)],
                        images=[preview_crop], return_tensors="pt"
                    ).to(device)
                    preview_prepared = encoder.prepare_global(
                        preview_inputs["pixel_values"],
                        preview_inputs["image_grid_thw"],
                    )
                    preview_grid = (
                        preview_prepared.map_feature_grid
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
                    preview_cells = F.adaptive_avg_pool2d(
                        preview_grid.float(), (3, 3)
                    )
                    preview_rows.append(
                        preview_cells.squeeze(0).permute(1, 2, 0).reshape(9, -1)
                    )
                    expanded_boxes.append(expanded)
                    preview_crop.close()
                    del preview_inputs, preview_prepared
                chosen_portfolio, trace_refine_plan = select_counterfactual_portfolio(
                    trace_refine_policy,
                    adaptive_features,
                    torch.stack(preview_rows),
                    [budget.bbox for budget in anchor_budgets],
                )
                allocation.region_budgets = [
                    RegionBudget(
                        bbox=tuple(int(value) for value in box),
                        token_budget=budget.token_budget,
                        score=budget.score,
                        role=budget.role,
                        source_rank=budget.source_rank,
                    )
                    for budget, box in zip(anchor_budgets, chosen_portfolio.boxes)
                ]
                trace_refine_plan["anchor_boxes"] = [
                    list(budget.bbox) for budget in anchor_budgets
                ]
                trace_refine_plan["expanded_preview_boxes"] = expanded_boxes
                trace_refine_plan["selected_boxes"] = [
                    list(box) for box in chosen_portfolio.boxes
                ]
            tracescale_plans: list[dict[str, Any]] = []
            if tracescale_head is not None:
                tracescale_plans = predict_region_geometry(
                    tracescale_head,
                    allocation.probability_map.detach().float().cpu().numpy(),
                    [budget.as_dict() for budget in allocation.region_budgets],
                    image_size=source.size,
                    question_length=len(selector_question.split()),
                    positive_only=args.tracescale_positive_only,
                    strength=args.tracescale_strength,
                )
                adjusted_budgets: list[RegionBudget] = []
                for budget, plan in zip(
                    allocation.region_budgets, tracescale_plans
                ):
                    adjusted_policy_box = tuple(
                        int(round(float(value) * 1000.0))
                        for value in plan["adjusted_bbox"]
                    )
                    adjusted_budgets.append(
                        RegionBudget(
                            bbox=adjusted_policy_box,
                            token_budget=budget.token_budget,
                            score=budget.score,
                            role=budget.role,
                            source_rank=budget.source_rank,
                        )
                    )
                    plan["original_policy_bbox"] = list(budget.bbox)
                    plan["adjusted_policy_bbox"] = list(adjusted_policy_box)
                allocation.region_budgets = adjusted_budgets
            region_location_control_plans: list[dict[str, Any]] = []
            allocation.region_budgets, region_location_control_plans = (
                relocate_region_budgets(
                    list(allocation.region_budgets),
                    mode=args.region_location_control,
                    row_id=str(row["id"]),
                    seed=args.region_location_control_seed,
                )
            )
            if patch_exchange_global_plan is not None:
                patch_exchange_fine_plan = plan_patch_exchange_fine_views(
                    source.width,
                    source.height,
                    [budget.bbox for budget in allocation.region_budgets],
                    global_plan=patch_exchange_global_plan,
                    priority_weights=[
                        float(budget.token_budget)
                        for budget in allocation.region_budgets
                    ],
                    local_scale=args.patch_exchange_local_scale,
                    total_cap_ratio=args.patch_exchange_total_cap_ratio,
                    soft_floor_tokens=(
                        effective_patch_exchange_soft_floor_tokens
                    ),
                    fill_soft_floor=args.patch_exchange_balanced_soft_floor,
                    minimum_view_pixels=args.patch_exchange_view_min_pixels,
                    maximum_view_pixels=args.processor_max_pixels,
                    native_global_anchor=(
                        args.patch_exchange_native_global_anchor
                    ),
                    additive_fine_ratio=(
                        args.patch_exchange_additive_fine_ratio
                    ),
                    additive_fine_max_tokens=(
                        args.patch_exchange_additive_fine_max_tokens
                    ),
                    shared_fine_token_override=multi_shared_fine_budget,
                    patch_size=int(model.config.vision_config.patch_size),
                    merge_size=merge_size,
                )
            fine_region_budgets = [
                *allocation.region_budgets,
                *adaptive_residual_budgets,
            ]
            maximum_weight = max(
                (budget.score for budget in fine_region_budgets), default=1.0
            )
            stage_d_v27b_plan = None
            if args.stage_d_v27b_safe_caps:
                stage_d_v27b_plan = plan_stage_d_v27b_safe_caps(
                    allocation.region_budgets,
                    total_fine_tokens=sample_fine_token_budget,
                    context_fraction=args.stage_d_v27b_context_fraction,
                    minimum_region_tokens=args.minimum_region_tokens,
                )
            continuous_fine_floor_plan = None
            region_fine_floors = list(
                plan_anchor_fine_floors(
                    [
                        int(budget.token_budget)
                        for budget in allocation.region_budgets
                    ],
                    base_minimum_tokens=args.native_cap_minimum_fine_tokens,
                    anchor_minimum_tokens=args.native_cap_anchor_fine_tokens,
                )
            )
            if args.evidence_decoder == "evicontour_csr" and allocation.region_budgets:
                if native_visual_tokens is None:
                    raise RuntimeError("EviContour-CSR requires the native Global token count")
                continuous_fine_floor_plan = plan_continuous_fine_floor(
                    int(native_visual_tokens),
                    [int(budget.token_budget) for budget in allocation.region_budgets],
                    native_ratio=args.evicontour_csr_fine_floor_native_ratio,
                    minimum_total_tokens=args.evicontour_csr_minimum_fine_total,
                    maximum_total_tokens=min(
                        args.evicontour_csr_maximum_fine_total,
                        sample_fine_token_budget,
                    ),
                )
                region_fine_floors = list(
                    continuous_fine_floor_plan.region_floor_tokens
                )
            elif (
                native_fine_cap
                and args.native_cap_fine_floor_native_ratio > 0
                and allocation.region_budgets
            ):
                if native_visual_tokens is None:
                    raise RuntimeError("Native-Cap source token count is missing")
                continuous_fine_floor_plan = plan_continuous_fine_floor(
                    int(native_visual_tokens),
                    [
                        int(budget.token_budget)
                        for budget in allocation.region_budgets
                    ],
                    native_ratio=args.native_cap_fine_floor_native_ratio,
                    minimum_total_tokens=args.native_cap_minimum_fine_total,
                    maximum_total_tokens=sample_fine_token_budget,
                )
                region_fine_floors = list(
                    continuous_fine_floor_plan.region_floor_tokens
                )
            fine_views: list[MidViTFineViewInput] = []
            region_plans: list[dict[str, Any]] = []
            native_cap_fine_plans: list[dict[str, Any]] = []
            native_pixel_fine_plans: list[dict[str, Any]] = []
            opened: list[Image.Image] = []
            base_region_count = len(allocation.region_budgets)
            for region_index, budget in enumerate(fine_region_budgets):
                patch_exchange_view_plan = (
                    patch_exchange_fine_plan.views[region_index]
                    if (
                        patch_exchange_fine_plan is not None
                        and region_index < len(patch_exchange_fine_plan.views)
                    )
                    else None
                )
                if (
                    patch_exchange_view_plan is not None
                    and patch_exchange_view_plan.dropped
                ):
                    continue
                pixel_box = policy_to_pixels(budget.bbox, *source.size)
                crop = source.crop(pixel_box).convert("RGB")
                native_cap_fine_plan = None
                native_pixel_fine_plan = None
                stage_d_region_token_cap = (
                    stage_d_v27b_plan.region_token_caps[region_index]
                    if stage_d_v27b_plan is not None
                    and stage_d_v27b_plan.region_token_caps is not None
                    and region_index < len(stage_d_v27b_plan.region_token_caps)
                    else None
                )
                stage_d_region_maximum_pixels = (
                    min(
                        fine_processor_max_pixels,
                        int(stage_d_region_token_cap * merged_stride**2),
                    )
                    if stage_d_region_token_cap is not None
                    else fine_processor_max_pixels
                )
                if patch_exchange_view_plan is not None:
                    fine_h = patch_exchange_view_plan.grid_height
                    fine_w = patch_exchange_view_plan.grid_width
                elif args.native_pixel_contract:
                    region_minimum_pixels = max(
                        args.processor_min_pixels,
                        int(
                            (
                                region_fine_floors[region_index]
                                if region_index < len(region_fine_floors)
                                else min(
                                    budget.token_budget,
                                    args.native_cap_minimum_fine_tokens,
                                )
                            )
                            * merged_stride**2
                        ),
                    )
                    native_pixel_fine_plan = plan_native_pixel_view(
                        crop.width,
                        crop.height,
                        minimum_pixels=region_minimum_pixels,
                        maximum_pixels=stage_d_region_maximum_pixels,
                        patch_size=int(model.config.vision_config.patch_size),
                        merge_size=merge_size,
                    )
                    fine_h = native_pixel_fine_plan.grid_height
                    fine_w = native_pixel_fine_plan.grid_width
                    native_pixel_fine_plans.append(
                        native_pixel_fine_plan.to_dict()
                    )
                elif native_fine_cap:
                    native_cap_fine_plan = plan_native_token_cap(
                        crop.width,
                        crop.height,
                        token_cap=budget.token_budget,
                        minimum_tokens=region_fine_floors[region_index],
                        minimum_pixels=args.processor_min_pixels,
                        maximum_pixels=args.processor_max_pixels,
                        patch_size=int(
                            model.config.vision_config.patch_size
                        ),
                        merge_size=merge_size,
                    )
                    fine_h = native_cap_fine_plan.grid_height
                    fine_w = native_cap_fine_plan.grid_width
                    native_cap_fine_plans.append(
                        native_cap_fine_plan.to_dict()
                    )
                else:
                    fine_h, fine_w = grid_for_token_budget(
                        crop.width, crop.height, budget.token_budget
                    )
                fine_image = resize_to_token_grid(
                    crop, fine_h, fine_w, merged_patch_stride=merged_stride
                )
                fine_inputs = processor(
                    text=[vision_probe_text(processor, model_type)],
                    images=[fine_image], return_tensors="pt"
                ).to(device)
                patch_evidence_weights = None
                if (
                    args.fine_patch_evidence_bias
                    or args.fine_readout_mode == "evidence_residual"
                ):
                    grid_t, premerge_h, premerge_w = (
                        int(value)
                        for value in fine_inputs["image_grid_thw"][0].tolist()
                    )
                    if grid_t != 1:
                        raise ValueError("Fine PatchBias currently requires a still image")
                    patch_centers, _ = qwen_grid_centers_in_original(
                        tuple(value / 1000.0 for value in budget.bbox),
                        grid_h=premerge_h,
                        grid_w=premerge_w,
                        spatial_merge_size=merge_size,
                        device=torch.device(device),
                        dtype=torch.float32,
                    )
                    sampling_grid = (
                        2.0 * patch_centers.clamp(0.0, 1.0) - 1.0
                    ).reshape(1, -1, 1, 2)
                    patch_prior = F.grid_sample(
                        allocation.probability_map.float().reshape(
                            1, 1, *allocation.probability_map.shape
                        ),
                        sampling_grid,
                        mode="bilinear",
                        padding_mode="border",
                        align_corners=False,
                    ).reshape(-1)
                    patch_prior = patch_prior.clamp_min(0.0)
                    patch_prior = patch_prior / patch_prior.max().clamp_min(1e-12)
                    patch_evidence_weights = (
                        args.fine_patch_evidence_floor
                        + (1.0 - args.fine_patch_evidence_floor)
                        * patch_prior.pow(args.fine_patch_evidence_exponent)
                    )
                fine_views.append(
                    MidViTFineViewInput(
                        pixel_values=fine_inputs["pixel_values"],
                        grid_thw=fine_inputs["image_grid_thw"],
                        bbox_xyxy=tuple(value / 1000.0 for value in budget.bbox),
                        evidence_weight=(
                            budget.score / max(maximum_weight, 1e-8)
                        )
                        * (
                            args.native_fuse_strength
                            if args.native_fuse
                            else 1.0
                        )
                        * (
                            allocation.reliability_gate
                            if allocation.reliability_gate is not None
                            and args.v10_bridge_gate_mode == "scaled"
                            else 1.0
                        ),
                        token_evidence_weights=(
                            patch_evidence_weights
                            if args.fine_patch_evidence_bias
                            else None
                        ),
                        readout_evidence_weights=(
                            patch_evidence_weights
                            if args.fine_readout_mode == "evidence_residual"
                            else None
                        ),
                    )
                )
                region_plans.append(
                    {
                        **budget.as_dict(),
                        "pixel_bbox": list(pixel_box),
                        "token_grid": [fine_h, fine_w],
                        "realized_tokens": fine_h * fine_w,
                        "native_cap_plan": (
                            native_cap_fine_plan.to_dict()
                            if native_cap_fine_plan is not None
                            else None
                        ),
                        "native_pixel_plan": (
                            native_pixel_fine_plan.to_dict()
                            if native_pixel_fine_plan is not None
                            else None
                        ),
                        "native_crop_tokens": (
                            native_cap_fine_plan.native_tokens
                            if native_cap_fine_plan is not None
                            else None
                        ),
                        "token_cap": (
                            native_cap_fine_plan.token_cap
                            if native_cap_fine_plan is not None
                            else None
                        ),
                        "unused_cap_tokens": (
                            native_cap_fine_plan.unused_cap_tokens
                            if native_cap_fine_plan is not None
                            else None
                        ),
                        "stage_d_role_token_cap": stage_d_region_token_cap,
                        "stage_d_role_maximum_pixels": (
                            stage_d_region_maximum_pixels
                            if args.stage_d_v27b_safe_caps
                            else None
                        ),
                        "patch_exchange_plan": (
                            patch_exchange_view_plan.to_dict()
                            if patch_exchange_view_plan is not None
                            else None
                        ),
                        "adaptive_residual_read": region_index >= base_region_count,
                        "fine_patch_evidence_weight_mean": (
                            float(patch_evidence_weights.mean().cpu())
                            if patch_evidence_weights is not None
                            else None
                        ),
                        "fine_patch_evidence_weight_min": (
                            float(patch_evidence_weights.min().cpu())
                            if patch_evidence_weights is not None
                            else None
                        ),
                        "fine_patch_evidence_weight_max": (
                            float(patch_evidence_weights.max().cpu())
                            if patch_evidence_weights is not None
                            else None
                        ),
                    }
                )
                opened.extend([crop, fine_image])

            vision = encoder.complete_from_prepared(
                prepared,
                fine_views,
                bridge_mode=bridge_mode,
                global_evidence_map=(
                    allocation.probability_map
                    if (
                        checkpoint_metadata[
                            "safe_joint_residual_mixer"
                        ]
                        is not None
                        or checkpoint_metadata[
                            "trace_calibrated_bridge_residual"
                        ]
                        is not None
                        or isinstance(bridge, EvidenceSlotCompressor)
                    )
                    else None
                ),
                visual_output_mode=args.visual_output_mode,
                fine_readout_mode=args.fine_readout_mode,
                capture_probe_blocks=representation_probe_blocks,
            )
            if args.representation_probe_output_dir is not None:
                args.representation_probe_output_dir.mkdir(
                    parents=True, exist_ok=True
                )
                safe_id = "".join(
                    character
                    if character.isalnum() or character in "._-"
                    else "_"
                    for character in str(row["id"])
                )
                probe_path = (
                    args.representation_probe_output_dir / f"{safe_id}.pt"
                )
                temporary_probe_path = probe_path.with_suffix(".pt.tmp")
                torch.save(
                    {
                        "format_version": "evivit_representation_probe_v1",
                        "variant": args.representation_variant_name,
                        "id": str(row["id"]),
                        "image": str(image_path),
                        "image_size": list(source.size),
                        "question": str(row["question"]),
                        "answer": row.get("answer"),
                        "eval_tier": row.get("eval_tier"),
                        "target_boxes_policy": (
                            row.get("official_target_boxes_policy")
                            or row.get("final_boxes_policy")
                            or []
                        ),
                        "ptea_probability_map": allocation.probability_map
                        .detach()
                        .to(dtype=torch.float16, device="cpu"),
                        "features_by_block": {
                            int(block): features.detach().to(
                                dtype=torch.float16, device="cpu"
                            )
                            for block, features in (
                                vision.probe_features_by_block.items()
                            )
                        },
                        "features_by_deepstack": (
                            {
                                int(layer): features.detach().to(
                                    dtype=torch.float16, device="cpu"
                                )
                                for layer, features in enumerate(
                                    vision.deepstack_features
                                )
                            }
                            if args.export_pixeleyes_adapter_span
                            else {}
                        ),
                        "merger_features": vision.image_embeds.detach().to(
                            dtype=torch.float16, device="cpu"
                        ),
                        "centers_xy": vision.token_centers_xy.detach().to(
                            dtype=torch.float16, device="cpu"
                        ),
                        "scales": vision.token_scales.detach().to(
                            dtype=torch.float16, device="cpu"
                        ),
                        "levels": vision.token_levels.detach().to(
                            dtype=torch.int8, device="cpu"
                        ),
                        "global_image_tokens": int(vision.global_image_tokens),
                        "fine_image_tokens": int(vision.fine_image_tokens),
                        "fine_view_token_counts": list(
                            vision.fine_view_token_counts
                        ),
                        "region_plans": region_plans,
                        "coordinate_bins": int(args.coordinate_bins),
                        "scale_bins": int(args.scale_bins),
                        "pixeleyes_adapter_config": (
                            {
                                "insertion_block": int(
                                    checkpoint_metadata["insertion_block"]
                                ),
                                "global_token_budget": int(
                                    args.global_token_budget
                                ),
                                "fine_token_budget": int(args.fine_token_budget),
                                "maximum_regions": int(args.max_regions),
                                "minimum_region_tokens": int(
                                    args.minimum_region_tokens
                                ),
                                "evidence_policy": str(args.evidence_policy),
                                "evidence_decoder": str(args.evidence_decoder),
                                "trace_split_budget_mode": str(
                                    args.trace_split_budget_mode
                                ),
                                "trace_split_min_context_fraction": float(
                                    args.trace_split_min_context_fraction
                                ),
                                "trace_split_max_context_fraction": float(
                                    args.trace_split_max_context_fraction
                                ),
                                "trace_split_context_expansion_factor": float(
                                    args.trace_split_context_expansion_factor
                                ),
                                "trace_split_context_decoder": str(
                                    args.trace_split_context_decoder
                                ),
                                "adaptive_box_safety_mode": str(
                                    args.adaptive_box_safety_mode
                                ),
                                "processor_min_pixels": int(
                                    args.processor_min_pixels
                                ),
                                "processor_max_pixels": int(
                                    args.processor_max_pixels
                                ),
                                "patch_exchange": bool(args.patch_exchange),
                                "patch_exchange_global_scale": float(
                                    args.patch_exchange_global_scale
                                ),
                                "patch_exchange_local_scale": float(
                                    args.patch_exchange_local_scale
                                ),
                                "patch_exchange_total_cap_ratio": float(
                                    args.patch_exchange_total_cap_ratio
                                ),
                                "patch_exchange_soft_floor_tokens": int(
                                    args.patch_exchange_soft_floor_tokens
                                ),
                                "patch_exchange_balanced_soft_floor": bool(
                                    args.patch_exchange_balanced_soft_floor
                                ),
                                "patch_exchange_preserve_native_global": bool(
                                    args.patch_exchange_preserve_native_global
                                ),
                                "patch_exchange_continuous_soft_floor": bool(
                                    args.patch_exchange_continuous_soft_floor
                                ),
                                "patch_exchange_continuous_soft_floor_max_tokens": int(
                                    args.patch_exchange_continuous_soft_floor_max_tokens
                                ),
                                "patch_exchange_continuous_soft_floor_ramp_start_tokens": int(
                                    args.patch_exchange_continuous_soft_floor_ramp_start_tokens
                                ),
                                "patch_exchange_continuous_soft_floor_ramp_end_tokens": int(
                                    args.patch_exchange_continuous_soft_floor_ramp_end_tokens
                                ),
                                "patch_exchange_view_min_pixels": int(
                                    args.patch_exchange_view_min_pixels
                                ),
                                "bridge_mode": str(bridge_mode),
                                "selector_checkpoint_sha256": sha256_file(
                                    selector_checkpoint
                                ),
                                "context_need_checkpoint_sha256": sha256_file(
                                    context_need_checkpoint
                                ),
                                "bridge_checkpoint_sha256": sha256_file(
                                    bridge_checkpoint
                                ),
                                "adaptive_box_checkpoint_sha256": sha256_file(
                                    adaptive_box_head_path
                                ),
                            }
                            if args.export_pixeleyes_adapter_span
                            else None
                        ),
                    },
                    temporary_probe_path,
                )
                temporary_probe_path.replace(probe_path)
            if native_global_cap:
                if native_cap_global_plan is None:
                    raise AssertionError("Native-Cap Global plan is missing")
                if (
                    vision.global_image_tokens
                    != native_cap_global_plan.realized_tokens
                ):
                    raise AssertionError(
                        "realized Global tokens differ from Native-Cap plan"
                    )
                if vision.global_image_tokens > args.global_token_budget:
                    raise AssertionError("Native-Cap exceeded Global ceiling")
            if native_fine_cap:
                planned_fine = [
                    int(plan["realized_tokens"])
                    for plan in native_cap_fine_plans
                ]
                if list(vision.fine_view_token_counts) != planned_fine:
                    raise AssertionError(
                        "realized Fine tokens differ from Native-Cap plans"
                    )
                if vision.fine_encoder_tokens > args.fine_token_budget:
                    raise AssertionError("Native-Cap exceeded shared Fine ceiling")
                if (
                    vision.global_image_tokens + vision.fine_encoder_tokens
                    > args.global_token_budget + args.fine_token_budget
                ):
                    raise AssertionError("Native-Cap exceeded total visual ceiling")
            if patch_exchange_fine_plan is not None:
                planned_fine = [
                    int(view.realized_tokens)
                    for view in patch_exchange_fine_plan.views
                    if not view.dropped
                ]
                if vision.global_image_tokens != patch_exchange_global_plan.realized_tokens:
                    raise AssertionError(
                        "realized Global tokens differ from PatchExchange plan"
                    )
                if (
                    list(vision.fine_view_token_counts[: len(planned_fine)])
                    != planned_fine
                ):
                    raise AssertionError(
                        "realized Fine tokens differ from PatchExchange plan"
                    )
                residual_fine_tokens = sum(
                    int(plan["realized_tokens"])
                    for plan in region_plans
                    if plan.get("adaptive_residual_read")
                )
                if (
                    vision.global_image_tokens + vision.fine_encoder_tokens
                    > patch_exchange_fine_plan.total_token_ceiling
                    + residual_fine_tokens
                ):
                    raise AssertionError(
                        "PatchExchange exceeded the per-image native-token ceiling"
                    )
            coordinates = (
                native_global_visual_coordinates(vision)
                if args.native_global_mrope
                else visual_coordinates(
                    vision,
                    coordinate_bins=args.coordinate_bins,
                    scale_bins=args.scale_bins,
                )
            )
            if args.reference_answer_nll_only:
                if row.get("_multi_parent_id") is not None:
                    raise ValueError(
                        "reference-answer NLL-only currently supports one source image"
                    )
                reference_nll, reference_accuracy, answer_tokens, prompt_sha256 = (
                    reference_answer_nll_from_visual_span(
                        model,
                        processor,
                        question=str(row["question"]),
                        answer=str(row["answer"]),
                        image_embeds=vision.image_embeds,
                        deepstack_embeds=vision.deepstack_features,
                        visual_coordinates_array=coordinates,
                        device=device,
                    )
                )
                diagnostic_output = {
                    "format_version": "evivit_forced_context_gt_nll_v1",
                    "deterministic_inference": args.deterministic_inference,
                    "attention_backend": expected_attention_backend,
                    "language_lora_adapter": expected_language_adapter,
                    "language_lora_adapter_bundle_fingerprint": (
                        language_lora_adapter_audit.bundle_fingerprint
                        if language_lora_adapter_audit is not None else None
                    ),
                    "id": str(row["id"]),
                    "image": str(row["image"]),
                    "question": str(row["question"]),
                    "answer": str(row["answer"]),
                    "reference_answer_nll": reference_nll,
                    "reference_answer_token_accuracy": reference_accuracy,
                    "reference_answer_tokens": answer_tokens,
                    "prompt_sha256": prompt_sha256,
                    "candidate_name": row.get("candidate_name"),
                    "candidate_family": row.get("candidate_family"),
                    "candidate_is_negative": bool(
                        row.get("candidate_is_negative", False)
                    ),
                    "forced_context_bbox_field": (
                        args.forced_context_bbox_field or None
                    ),
                    "forced_context_bbox": (
                        list(forced_context_bbox)
                        if forced_context_bbox is not None
                        else None
                    ),
                    "forced_context_applied": forced_context_applied,
                    "context_bbox_before_hsafe": context_bbox_before_hsafe,
                    "realized_context_bbox_after_hsafe": (
                        list(allocation.region_budgets[-1].bbox)
                        if allocation.region_budgets
                        else None
                    ),
                    "region_plans": region_plans,
                    "map_entropy": float(allocation.map_entropy),
                    "map_peak_probability": float(
                        allocation.map_peak_probability
                    ),
                    "context_map_entropy": allocation.context_map_entropy,
                    "trace_split_context_fraction": (
                        allocation.trace_split_context_fraction
                    ),
                    "native_visual_tokens": native_visual_tokens,
                    "realized_global_tokens": int(vision.global_image_tokens),
                    "realized_fine_tokens": int(vision.fine_image_tokens),
                    "realized_fine_encoder_tokens": int(
                        vision.fine_encoder_tokens
                    ),
                    "realized_visual_tokens": int(vision.image_embeds.shape[0]),
                    "fine_view_token_counts": list(
                        vision.fine_view_token_counts
                    ),
                    "patch_exchange_effective_soft_floor_tokens": (
                        effective_patch_exchange_soft_floor_tokens
                        if args.patch_exchange
                        else None
                    ),
                    "patch_exchange_fine_plan": (
                        patch_exchange_fine_plan.to_dict()
                        if patch_exchange_fine_plan is not None
                        else None
                    ),
                    "relative_residual_l2": float(
                        vision.relative_residual_l2.detach().float().cpu()
                    ),
                    "global_relative_residual_l2": float(
                        vision.global_relative_residual_l2.detach().float().cpu()
                    ),
                    "fine_relative_residual_l2": float(
                        vision.fine_relative_residual_l2.detach().float().cpu()
                    ),
                    "scope": (
                        "training utility diagnostic; not generated accuracy or Judge"
                    ),
                }
                output_rows.append(diagnostic_output)
                handle.write(
                    json.dumps(diagnostic_output, ensure_ascii=False) + "\n"
                )
                handle.flush()
                print(
                    json.dumps(
                        {
                            "progress": index,
                            "total": len(remaining),
                            "id": diagnostic_output["id"],
                            "candidate": diagnostic_output["candidate_name"],
                            "reference_answer_nll": reference_nll,
                            "visual_tokens": diagnostic_output[
                                "realized_visual_tokens"
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                closed: set[int] = set()
                for image in [*opened, global_view, source]:
                    if id(image) not in closed:
                        image.close()
                        closed.add(id(image))
                del global_inputs, prepared, question_tokens, allocation, fine_views, vision
                continue
            if args.record_timing:
                torch.cuda.synchronize(0)
                visual_finished = time.perf_counter()
            multi_entry: dict[str, Any] | None = None
            multi_parent_id = row.get("_multi_parent_id")
            if multi_parent_id is not None:
                multi_entry = multi_visual_accumulator.setdefault(
                    str(multi_parent_id),
                    {
                        "request_started": request_started,
                        "image_embeds": [],
                        "deepstack": [],
                        "coordinates": [],
                        "per_image": [],
                    },
                )
                multi_entry["image_embeds"].append(vision.image_embeds)
                multi_entry["deepstack"].append(vision.deepstack_features)
                multi_entry["coordinates"].append(coordinates)
                multi_entry["per_image"].append(
                    {
                        "index": int(row["_multi_image_index"]),
                        "image": str(row["image"]),
                        "width": source.width,
                        "height": source.height,
                        "global_tokens": int(vision.global_image_tokens),
                        "fine_tokens": int(vision.fine_image_tokens),
                        "fine_encoder_tokens": int(vision.fine_encoder_tokens),
                        "shared_fine_budget": multi_shared_fine_budget,
                        "shared_fine_selected": bool(
                            multi_shared_fine_budget is None
                            or multi_shared_fine_budget > 0
                        ),
                        "visual_tokens": int(vision.image_embeds.shape[0]),
                        "regions": len(fine_views),
                        "fine_view_token_counts": list(
                            vision.fine_view_token_counts
                        ),
                        "region_plans": region_plans,
                        "map_entropy": float(allocation.map_entropy),
                        "map_peak_probability": float(
                            allocation.map_peak_probability
                        ),
                        "global_pass_seconds": (
                            global_pass_finished - request_started
                            if args.record_timing
                            else None
                        ),
                        "allocation_seconds": (
                            allocation_finished - global_pass_finished
                            if args.record_timing
                            else None
                        ),
                        "fine_reread_and_fusion_seconds": (
                            visual_finished - allocation_finished
                            if args.record_timing
                            else None
                        ),
                    }
                )
                if int(row["_multi_image_index"]) + 1 < int(
                    row["_multi_image_count"]
                ):
                    closed: set[int] = set()
                    for image in [*opened, global_view, source]:
                        if id(image) not in closed:
                            image.close()
                            closed.add(id(image))
                    del global_inputs, prepared, question_tokens, allocation, fine_views, vision
                    continue
                request_started = float(multi_entry["request_started"])
                generation_result = greedy_generate_from_visual_spans(
                    model,
                    processor,
                    build_multi_chat_text(
                        processor,
                        str(row["question"]),
                        int(row["_multi_image_count"]),
                        None,
                        answer_protocol=args.answer_protocol,
                        enable_thinking=(
                            False
                            if model_type == "qwen3_5" and args.disable_thinking
                            else None
                        ),
                    ),
                    multi_entry["image_embeds"],
                    multi_entry["deepstack"],
                    multi_entry["coordinates"],
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    record_timing=args.record_timing,
                    record_confidence=args.record_answer_confidence,
                )
                row = dict(row)
                row["id"] = str(multi_parent_id)
                row["images"] = list(row["_multi_images"])
            else:
                generation_result = greedy_generate_from_visual_span(
                    model,
                    processor,
                    build_chat_text(
                        processor,
                        str(row["question"]),
                        None,
                        answer_protocol=args.answer_protocol,
                        enable_thinking=(
                            False
                            if model_type == "qwen3_5" and args.disable_thinking
                            else None
                        ),
                    ),
                    vision.image_embeds,
                    vision.deepstack_features,
                    coordinates,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    record_timing=args.record_timing,
                    record_confidence=args.record_answer_confidence,
                )
            if args.record_timing or args.record_answer_confidence:
                raw, generation_metadata = generation_result
                timing = generation_metadata if args.record_timing else None
            else:
                raw = generation_result
                generation_metadata = None
                timing = None
            if args.answer_protocol == "muir_mcq":
                prediction, extracted, parse_error = parse_muir_choice(raw)
            elif args.answer_protocol == "blink_mcq":
                prediction, extracted, parse_error = parse_blink_choice(raw)
            else:
                prediction, extracted, parse_error = parse_answer(raw)
            target = str(row.get("answer", ""))
            eviguard_features: list[float] | None = None
            if args.record_eviguard_features:
                if adaptive_features is None or adaptive_box_head is None:
                    raise RuntimeError(
                        "EviGuard features require an active AdaptiveBox head"
                    )
                normalized = (
                    adaptive_features.float() - adaptive_box_head.feature_mean.float()
                ) / adaptive_box_head.feature_scale.float().clamp_min(1e-6)
                generator = torch.Generator(device="cpu").manual_seed(
                    args.eviguard_projection_seed
                )
                eviguard_projection = torch.randn(
                    normalized.shape[1],
                    args.eviguard_projection_dim,
                    generator=generator,
                    dtype=torch.float32,
                ).to(device) / math.sqrt(normalized.shape[1])
                projected = normalized @ eviguard_projection
                aggregate = torch.cat(
                    (
                        projected.mean(dim=0),
                        projected.std(dim=0, unbiased=False),
                        projected.amax(dim=0),
                        projected[0],
                    )
                )
                widths = [
                    (budget.bbox[2] - budget.bbox[0]) / 1000.0
                    for budget in allocation.region_budgets
                ]
                heights = [
                    (budget.bbox[3] - budget.bbox[1]) / 1000.0
                    for budget in allocation.region_budgets
                ]
                realized_global = int(vision.global_image_tokens)
                realized_fine = int(vision.fine_encoder_tokens)
                scalar = [
                    math.log1p(source.width * source.height / 1e6),
                    math.log(max(source.width, source.height) / min(source.width, source.height)),
                    float(native_visual_tokens) / 1000.0,
                    float(realized_global + realized_fine) / 1000.0,
                    float(realized_global) / 1000.0,
                    float(realized_fine) / 1000.0,
                    float(allocation.map_entropy),
                    float(allocation.trace_split_context_fraction or 0.0),
                    len(allocation.region_budgets) / 3.0,
                    float(np.mean(widths)),
                    float(np.std(widths)),
                    float(np.mean(heights)),
                    float(np.std(heights)),
                    float(sum(w * h for w, h in zip(widths, heights))),
                    len(str(row["question"]).replace("<image>", "").split()) / 32.0,
                    len(allocation.region_budgets) / 3.0,
                ]
                eviguard_features = scalar + aggregate.detach().cpu().tolist()
                if len(eviguard_features) != 16 + 4 * args.eviguard_projection_dim:
                    raise AssertionError("unexpected EviGuard feature dimension")
            output = {
                "id": row["id"],
                "eval_tier": row.get("eval_tier"),
                "image": row.get("image"),
                "question": row.get("question"),
                "image_width": source.width,
                "image_height": source.height,
                "answer": target,
                "raw_prediction": raw,
                "extracted_prediction": extracted,
                "prediction": prediction,
                "normalized_prediction": normalized_answer(prediction),
                "normalized_answer": normalized_answer(target),
                "relaxed_prediction": relaxed_answer(prediction),
                "relaxed_answer": relaxed_answer(target),
                "exact_correct": normalized_answer(prediction)
                == normalized_answer(target),
                "relaxed_correct": relaxed_answer_match(prediction, target),
                "parse_error": parse_error,
                "answer_mean_logprob": (
                    generation_metadata.get("answer_mean_logprob")
                    if generation_metadata is not None
                    else None
                ),
                "answer_min_logprob": (
                    generation_metadata.get("answer_min_logprob")
                    if generation_metadata is not None
                    else None
                ),
                "answer_scored_tokens": (
                    generation_metadata.get("answer_scored_tokens")
                    if generation_metadata is not None
                    else None
                ),
                "method": (
                    "evivit_v7_nativefuse"
                    if args.native_fuse
                    else (
                        "evivit_v7_trace_refine"
                        if trace_refine_policy is not None
                        else (
                            (
                                (
                                    "evivit_v9_t2_hsafe"
                                    if adaptive_box_head is not None
                                    else "evivit_v8_t2_patch_exchange"
                                )
                                if args.patch_exchange_tree
                                else "evivit_v5_patch_exchange"
                            )
                            if args.patch_exchange
                            else (
                                f"evivit_v3_{args.evidence_policy}_bridge"
                                if bridge_checkpoint is not None
                                else f"evivit_v3_{args.evidence_policy}_zero_bridge"
                            )
                        )
                    )
                ),
                "model": str(model_path),
                "model_index_sha256": model_index_sha256,
                "evidence_policy": args.evidence_policy,
                "evidence_decoder": allocation.evidence_decoder,
                "region_expansion_factor": args.region_expansion_factor,
                "topology_union_factor": args.topology_union_factor,
                "token_serialization": vision.token_serialization,
                "serialization_moved_tokens": vision.serialization_moved_tokens,
                "global_frame_joint_blocks": list(
                    args.global_frame_joint_blocks
                ),
                "global_frame_coordinate_bins": (
                    args.global_frame_coordinate_bins
                ),
                "global_frame_joint_topology": (
                    args.global_frame_joint_topology
                ),
                "control_seed": args.control_seed,
                "checkpoint": checkpoint_metadata["checkpoint"],
                "checkpoint_step": checkpoint_metadata["checkpoint_step"],
                "fusion": checkpoint_metadata["fusion"],
                "insertion_block": checkpoint_metadata["insertion_block"],
                "global_token_budget": args.global_token_budget,
                "requested_global_token_budget": (
                    patch_exchange_global_plan.requested_token_ceiling
                    if args.patch_exchange_tree
                    and patch_exchange_global_plan is not None
                    else sample_global_token_budget
                ),
                "tree_policy_requested_global_tokens": (
                    native_need_ratio_plan.global_tokens
                    if args.patch_exchange_tree
                    and native_need_ratio_plan is not None
                    else None
                ),
                "native_ratio_budget": args.native_ratio_budget,
                "native_need_ratio_budget": args.native_need_ratio_budget,
                "native_global_ratio": (
                    args.native_global_ratio
                    if args.native_ratio_budget or args.native_need_ratio_budget
                    else None
                ),
                "native_fine_ratio": (
                    args.native_fine_ratio
                    if args.native_ratio_budget
                    else None
                ),
                "native_min_fine_ratio": (
                    args.native_min_fine_ratio
                    if args.native_need_ratio_budget
                    else None
                ),
                "native_max_fine_ratio": (
                    args.native_max_fine_ratio
                    if args.native_need_ratio_budget
                    else None
                ),
                "native_need_ratio_plan": (
                    native_need_ratio_plan.to_dict()
                    if native_need_ratio_plan is not None
                    else None
                ),
                "adaptive_native_global_budget": (
                    args.adaptive_native_global_budget
                ),
                "fine_token_budget": args.fine_token_budget,
                "requested_fine_token_budget": (
                    patch_exchange_fine_plan.allocated_fine_token_ceiling
                    if args.patch_exchange_tree
                    and patch_exchange_fine_plan is not None
                    else (
                        allocation.requested_fine_token_budget
                        if allocation.requested_fine_token_budget is not None
                        else args.fine_token_budget
                    )
                ),
                "tree_policy_requested_fine_tokens": (
                    allocation.requested_fine_token_budget
                    if args.patch_exchange_tree
                    else None
                ),
                "evitree_evidence_need": (
                    allocation.evitree_evidence_need
                ),
                "evitree_tree_allocation": args.evitree_tree_allocation,
                "evitree_tree_nodes": (
                    allocation.candidates
                    if args.evitree_tree_allocation
                    else None
                ),
                "evitree_tree_reliability": (
                    allocation.evitree_tree_reliability
                ),
                "evitree_split_policy": allocation.evitree_split_policy,
                "evitree_split_threshold": allocation.evitree_split_threshold,
                "native_fit_total_token_budget": args.native_fit_total_token_budget,
                "native_fit_bypass": False,
                "patch_exchange": args.patch_exchange,
                "patch_exchange_tree": args.patch_exchange_tree,
                "patch_exchange_global_scale": (
                    args.patch_exchange_global_scale
                    if args.patch_exchange
                    else None
                ),
                "patch_exchange_local_scale": (
                    args.patch_exchange_local_scale
                    if args.patch_exchange
                    else None
                ),
                "patch_exchange_total_cap_ratio": (
                    args.patch_exchange_total_cap_ratio
                    if args.patch_exchange
                    else None
                ),
                "patch_exchange_view_min_pixels": (
                    args.patch_exchange_view_min_pixels
                    if args.patch_exchange
                    else None
                ),
                "patch_exchange_soft_floor_tokens": (
                    args.patch_exchange_soft_floor_tokens
                    if args.patch_exchange
                    else None
                ),
                "patch_exchange_balanced_soft_floor": (
                    args.patch_exchange_balanced_soft_floor
                    if args.patch_exchange
                    else None
                ),
                "patch_exchange_preserve_native_global": (
                    args.patch_exchange_preserve_native_global
                    if args.patch_exchange
                    else None
                ),
                "patch_exchange_continuous_soft_floor": (
                    args.patch_exchange_continuous_soft_floor
                    if args.patch_exchange
                    else None
                ),
                "patch_exchange_continuous_soft_floor_max_tokens": (
                    args.patch_exchange_continuous_soft_floor_max_tokens
                    if args.patch_exchange_continuous_soft_floor
                    else None
                ),
                "patch_exchange_continuous_soft_floor_ramp_start_tokens": (
                    args.patch_exchange_continuous_soft_floor_ramp_start_tokens
                    if args.patch_exchange_continuous_soft_floor
                    else None
                ),
                "patch_exchange_continuous_soft_floor_ramp_end_tokens": (
                    args.patch_exchange_continuous_soft_floor_ramp_end_tokens
                    if args.patch_exchange_continuous_soft_floor
                    else None
                ),
                "patch_exchange_native_global_anchor": (
                    args.patch_exchange_native_global_anchor
                    if args.patch_exchange
                    else None
                ),
                "patch_exchange_additive_fine_ratio": (
                    args.patch_exchange_additive_fine_ratio
                    if args.patch_exchange_native_global_anchor
                    else None
                ),
                "patch_exchange_additive_fine_max_tokens": (
                    args.patch_exchange_additive_fine_max_tokens
                    if args.patch_exchange_native_global_anchor
                    else None
                ),
                "patch_exchange_effective_soft_floor_tokens": (
                    effective_patch_exchange_soft_floor_tokens
                    if args.patch_exchange
                    else None
                ),
                "patch_exchange_global_plan": (
                    patch_exchange_global_plan.to_dict()
                    if patch_exchange_global_plan is not None
                    else None
                ),
                "patch_exchange_fine_plan": (
                    patch_exchange_fine_plan.to_dict()
                    if patch_exchange_fine_plan is not None
                    else None
                ),
                "native_pixel_contract": args.native_pixel_contract,
                "native_fuse": args.native_fuse,
                "native_fuse_strength": (
                    args.native_fuse_strength if args.native_fuse else None
                ),
                "continuous_native_global": args.continuous_native_global,
                "continuous_global_base_pixels": (
                    args.continuous_global_base_pixels
                    if args.continuous_native_global
                    else None
                ),
                "continuous_global_max_pixels": (
                    args.continuous_global_max_pixels
                    if args.continuous_native_global
                    else None
                ),
                "continuous_global_exponent": (
                    args.continuous_global_exponent
                    if args.continuous_native_global
                    else None
                ),
                "continuous_global_plan": (
                    continuous_global_plan.to_dict()
                    if continuous_global_plan is not None
                    else None
                ),
                "visual_output_mode": args.visual_output_mode,
                "fine_readout_mode": args.fine_readout_mode,
                "native_global_mrope": args.native_global_mrope,
                "bridge_mode": bridge_mode,
                "answer_protocol": args.answer_protocol,
                "disable_thinking": args.disable_thinking,
                "coverage_masked_parent_writeback": (
                    args.coverage_masked_parent_writeback
                ),
                "global_processor_max_pixels": (
                    global_processor_max_pixels
                    if args.native_pixel_contract
                    else None
                ),
                "fine_processor_max_pixels": (
                    fine_processor_max_pixels
                    if args.native_pixel_contract
                    else None
                ),
                "native_pixel_global_plan": (
                    native_pixel_global_plan.to_dict()
                    if native_pixel_global_plan is not None
                    else None
                ),
                "native_pixel_fine_plans": (
                    native_pixel_fine_plans
                    if args.native_pixel_contract
                    else None
                ),
                "native_token_caps": args.native_token_caps,
                "native_cap_scope": (
                    args.native_cap_scope if args.native_token_caps else None
                ),
                "native_cap_minimum_fine_tokens": (
                    args.native_cap_minimum_fine_tokens
                    if native_fine_cap
                    else None
                ),
                "native_cap_minimum_global_tokens": (
                    args.native_cap_minimum_global_tokens
                    if native_global_cap
                    else None
                ),
                "native_cap_anchor_fine_tokens": (
                    args.native_cap_anchor_fine_tokens
                    if native_fine_cap
                    else None
                ),
                "native_cap_fine_floor_native_ratio": (
                    args.native_cap_fine_floor_native_ratio
                    if native_fine_cap
                    else None
                ),
                "native_cap_continuous_fine_floor_plan": (
                    continuous_fine_floor_plan.to_dict()
                    if continuous_fine_floor_plan is not None
                    else None
                ),
                "evicontour_csr_config": expected_csr_config,
                "evicontour_csr_fine_floor_plan": (
                    continuous_fine_floor_plan.to_dict()
                    if args.evidence_decoder == "evicontour_csr"
                    and continuous_fine_floor_plan is not None
                    else None
                ),
                "native_cap_global_plan": (
                    native_cap_global_plan.to_dict()
                    if native_cap_global_plan is not None
                    else None
                ),
                "native_cap_fine_plans": (
                    native_cap_fine_plans if native_fine_cap else None
                ),
                "native_visual_tokens": native_visual_tokens,
                "native_grid_thw": native_grid_thw,
                "realized_global_tokens": vision.global_image_tokens,
                "realized_fine_tokens": vision.fine_image_tokens,
                "realized_fine_encoder_tokens": vision.fine_encoder_tokens,
                "native_cap_global_unused_tokens": (
                    args.global_token_budget - vision.global_image_tokens
                    if native_global_cap
                    else None
                ),
                "native_cap_fine_unused_tokens": (
                    args.fine_token_budget - vision.fine_encoder_tokens
                    if native_fine_cap
                    else None
                ),
                "native_cap_global_saturated": (
                    native_cap_global_plan.cap_saturated
                    if native_cap_global_plan is not None
                    else None
                ),
                "native_cap_global_grid_preserved": (
                    native_cap_global_plan.native_grid_preserved
                    if native_cap_global_plan is not None
                    else None
                ),
                "native_cap_fine_saturated_views": (
                    sum(
                        bool(plan["cap_saturated"])
                        for plan in native_cap_fine_plans
                    )
                    if native_fine_cap
                    else None
                ),
                "native_cap_fine_native_grid_preserved_views": (
                    sum(
                        bool(plan["native_grid_preserved"])
                        for plan in native_cap_fine_plans
                    )
                    if native_fine_cap
                    else None
                ),
                "fine_view_token_counts": list(vision.fine_view_token_counts),
                "view_token_counts": [
                    vision.global_image_tokens,
                    *vision.fine_view_token_counts,
                ],
                "realized_visual_tokens": int(vision.image_embeds.shape[0]),
                "source_images": 1,
                "single_visual_span": True,
                "selector_question_protocol": (
                    "shuffled_other_sample_question"
                    if args.ptea_question_control == "shuffled"
                    else (
                        "zero_question_tokens"
                        if args.ptea_question_control == "zero"
                        else (
                            "explicit_selector_question"
                            if row.get("selector_question")
                            else "question_text_from_manifest"
                        )
                    )
                ),
                "ptea_question_control": args.ptea_question_control,
                "ptea_question_shuffle_seed": args.ptea_question_shuffle_seed,
                "ptea_question_control_source_id": row.get(
                    "_ptea_control_source_id"
                ),
                "region_location_control": args.region_location_control,
                "region_location_control_seed": args.region_location_control_seed,
                "region_location_control_plans": region_location_control_plans,
                "regions": len(fine_views),
                "region_plans": region_plans,
                "stage_d_v27b_safe_caps": args.stage_d_v27b_safe_caps,
                "stage_d_v27b_context_fraction": (
                    stage_d_v27b_plan.context_fraction
                    if stage_d_v27b_plan is not None
                    else None
                ),
                "stage_d_v27b_route": (
                    stage_d_v27b_plan.route
                    if stage_d_v27b_plan is not None
                    else None
                ),
                "stage_d_v27b_region_token_caps": (
                    list(stage_d_v27b_plan.region_token_caps)
                    if stage_d_v27b_plan is not None
                    and stage_d_v27b_plan.region_token_caps is not None
                    else None
                ),
                "tracescale_checkpoint": (
                    str(tracescale_head_path)
                    if tracescale_head_path is not None
                    else None
                ),
                "tracescale_positive_only": args.tracescale_positive_only,
                "tracescale_strength": args.tracescale_strength,
                "tracescale_plans": tracescale_plans,
                "adaptive_box_checkpoint": (
                    str(adaptive_box_head_path)
                    if adaptive_box_head_path is not None
                    else None
                ),
                "adaptive_box_safety_mode": args.adaptive_box_safety_mode,
                "adaptive_box_uncertainty_threshold": (
                    args.adaptive_box_uncertainty_threshold
                    if args.adaptive_box_safety_mode.startswith("uncertainty")
                    else None
                ),
                "adaptive_box_ensemble_heads": [
                    str(path) for path in adaptive_box_ensemble_paths
                ],
                "adaptive_box_residual_read_tokens": (
                    args.adaptive_box_residual_read_tokens
                    if args.adaptive_box_safety_mode == "residual_read"
                    else None
                ),
                "fine_patch_evidence_bias": args.fine_patch_evidence_bias,
                "fine_patch_evidence_floor": args.fine_patch_evidence_floor,
                "fine_patch_evidence_exponent": args.fine_patch_evidence_exponent,
                "adaptive_box_plans": adaptive_box_plans,
                "eviguard_features": eviguard_features,
                "eviguard_projection_dim": (
                    args.eviguard_projection_dim
                    if args.record_eviguard_features
                    else None
                ),
                "eviguard_projection_seed": (
                    args.eviguard_projection_seed
                    if args.record_eviguard_features
                    else None
                ),
                "trace_refine_checkpoint": (
                    str(trace_refine_policy_path)
                    if trace_refine_policy_path is not None
                    else None
                ),
                "trace_refine_preview_expansion": (
                    args.trace_refine_preview_expansion
                ),
                "trace_refine_preview_tokens": args.trace_refine_preview_tokens,
                "trace_refine_plan": trace_refine_plan,
                "anchored_growth_calibrator": (
                    str(anchored_growth_calibrator_path)
                    if anchored_growth_calibrator_path is not None
                    else None
                ),
                "anchored_growth_gate_threshold": (
                    args.anchored_growth_gate_threshold
                ),
                "evicontour_utility_checkpoint": (
                    str(evicontour_utility_path)
                    if evicontour_utility_path is not None
                    else None
                ),
                "map_entropy": allocation.map_entropy,
                "map_peak_probability": allocation.map_peak_probability,
                "context_map_entropy": allocation.context_map_entropy,
                "trace_split_context_fraction": (
                    allocation.trace_split_context_fraction
                ),
                "trace_split_context_expansion_factor": (
                    allocation.trace_split_context_expansion_factor
                ),
                "trace_split_context_decoder": (
                    allocation.trace_split_context_decoder
                ),
                "trace_split_budget_mode": (
                    allocation.trace_split_budget_mode
                ),
                "trace_split_secondary_decisive_share": (
                    allocation.trace_split_secondary_decisive_share
                ),
                "trace_split_min_context_fraction": (
                    allocation.trace_split_min_context_fraction
                ),
                "trace_split_max_context_fraction": (
                    allocation.trace_split_max_context_fraction
                ),
                "trace_split_adaptive_region_minimum": (
                    args.trace_split_adaptive_region_minimum
                ),
                "trace_split_effective_minimum_region_tokens": (
                    allocation.trace_split_effective_minimum_region_tokens
                ),
                "trace_split_context_need_features": (
                    allocation.trace_split_context_need_features
                ),
                "v10_gate_mode": allocation.reliability_gate_mode,
                "v10_reliability_gate": allocation.reliability_gate,
                "v10_requested_fine_token_budget": (
                    allocation.requested_fine_token_budget
                ),
                "v10_source_map_entropy": allocation.source_map_entropy,
                "v10_map_mode": args.v10_map_mode,
                "v10_budget_mode": args.v10_budget_mode,
                "v10_bridge_gate_mode": args.v10_bridge_gate_mode,
                "relative_residual_l2": float(
                    vision.relative_residual_l2.detach().cpu()
                ),
                "global_relative_residual_l2": float(
                    vision.global_relative_residual_l2.detach().cpu()
                ),
                "fine_relative_residual_l2": float(
                    vision.fine_relative_residual_l2.detach().cpu()
                ),
                "fusion_diagnostics": collect_fusion_diagnostics(
                    bridge, encoder, vision
                ),
                "deterministic_inference": args.deterministic_inference,
                "attention_backend": (
                    deterministic_backend_name(
                        args.deterministic_attention_backend,
                        ptea_auxiliary_attention=args.evidence_policy == "mid_ptea",
                    )
                    if args.deterministic_inference
                    else "sdpa_default"
                ),
            }
            output.update(recovery_row_metadata(row))
            if language_lora_adapter_audit is not None:
                output.update(
                    {
                        "language_lora_adapter": str(language_lora_adapter_path),
                        "language_lora_adapter_config_sha256": (
                            language_lora_adapter_audit.config_sha256
                        ),
                        "language_lora_adapter_weights_sha256": (
                            language_lora_adapter_audit.weights_sha256
                        ),
                        "language_lora_adapter_bundle_fingerprint": (
                            language_lora_adapter_audit.bundle_fingerprint
                        ),
                    }
                )
            if multi_entry is not None:
                per_image = list(multi_entry["per_image"])
                output.update(
                    {
                        "image": None,
                        "images": list(row["images"]),
                        "image_count": len(per_image),
                        "image_width": [int(item["width"]) for item in per_image],
                        "image_height": [int(item["height"]) for item in per_image],
                        "multi_image_protocol": (
                            "shared_fine_select_image_then_region_v1"
                            if multi_shared_plan_path is not None
                            else "independent_evivit_spans_v1"
                        ),
                        "source_images": len(per_image),
                        "single_visual_span": False,
                        "realized_global_tokens": sum(
                            int(item["global_tokens"]) for item in per_image
                        ),
                        "realized_fine_tokens": sum(
                            int(item["fine_tokens"]) for item in per_image
                        ),
                        "realized_fine_encoder_tokens": sum(
                            int(item["fine_encoder_tokens"]) for item in per_image
                        ),
                        "realized_visual_tokens": sum(
                            int(item["visual_tokens"]) for item in per_image
                        ),
                        "regions": sum(int(item["regions"]) for item in per_image),
                        "fine_view_token_counts": [
                            list(item["fine_view_token_counts"])
                            for item in per_image
                        ],
                        "view_token_counts": [
                            [
                                int(item["global_tokens"]),
                                *list(item["fine_view_token_counts"]),
                            ]
                            for item in per_image
                        ],
                        "region_plans": None,
                        "region_plans_by_image": [
                            item["region_plans"] for item in per_image
                        ],
                        "map_entropy": sum(
                            float(item["map_entropy"]) for item in per_image
                        )
                        / len(per_image),
                        "map_entropy_by_image": [
                            float(item["map_entropy"]) for item in per_image
                        ],
                        "map_peak_probability": max(
                            float(item["map_peak_probability"])
                            for item in per_image
                        ),
                        "map_peak_probability_by_image": [
                            float(item["map_peak_probability"])
                            for item in per_image
                        ],
                        "per_image_evivit": per_image,
                        "multi_shared_fine_plan": (
                            row.get("_multi_shared_plan")
                            if multi_shared_plan_path is not None
                            else None
                        ),
                        "shared_fine_budget": (
                            int(row["_multi_shared_plan"]["shared_fine_budget"])
                            if multi_shared_plan_path is not None
                            else None
                        ),
                        "shared_fine_realized_tokens": (
                            sum(int(item["fine_encoder_tokens"]) for item in per_image)
                            if multi_shared_plan_path is not None
                            else None
                        ),
                    }
                )
            if timing is not None:
                output["elapsed_seconds"] = time.perf_counter() - request_started
                output["peak_memory_mib"] = (
                    torch.cuda.max_memory_allocated(0) / 1024**2
                )
                first_token_wall = float(timing.pop("_first_token_wall_time"))
                generation_finished_wall = float(
                    timing.pop("_generation_finished_wall_time")
                )
                generation_started_wall = (
                    first_token_wall - float(timing["llm_prefill_to_first_token_seconds"])
                )
                output["timing"] = {
                    "request_to_first_token_seconds": first_token_wall - request_started,
                    "end_to_end_seconds": generation_finished_wall - request_started,
                    "global_evidence_pass_seconds": global_pass_finished
                    - request_started,
                    "evidence_allocation_seconds": allocation_finished
                    - global_pass_finished,
                    "fine_reread_and_fusion_seconds": visual_finished
                    - allocation_finished,
                    "prompt_and_embedding_seconds": generation_started_wall
                    - visual_finished,
                    "evidence_and_vision_seconds": generation_started_wall - request_started,
                    **timing,
                }
                if multi_entry is not None:
                    output["timing"].update(
                        {
                            "global_evidence_pass_seconds": sum(
                                float(item["global_pass_seconds"])
                                for item in per_image
                            ),
                            "evidence_allocation_seconds": sum(
                                float(item["allocation_seconds"])
                                for item in per_image
                            ),
                            "fine_reread_and_fusion_seconds": sum(
                                float(item["fine_reread_and_fusion_seconds"])
                                for item in per_image
                            ),
                            "evidence_and_vision_seconds": sum(
                                float(item["global_pass_seconds"])
                                + float(item["allocation_seconds"])
                                + float(item["fine_reread_and_fusion_seconds"])
                                for item in per_image
                            ),
                        }
                    )
            output_rows.append(output)
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "progress": index,
                        "total": len(remaining),
                        "id": output["id"],
                        "prediction": prediction,
                        "answer": target,
                        "exact": output["exact_correct"],
                        "relaxed": output["relaxed_correct"],
                        "visual_tokens": output["realized_visual_tokens"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            closed: set[int] = set()
            for image in [*opened, global_view, source]:
                if id(image) not in closed:
                    image.close()
                    closed.add(id(image))
            del global_inputs, prepared, question_tokens, allocation, fine_views, vision
            if multi_entry is not None:
                multi_visual_accumulator.pop(str(row["id"]), None)

    if args.reference_answer_nll_only:
        summary = {
            "format_version": "evivit_forced_context_gt_nll_summary_v1",
            "status": "complete",
            "rows": len(output_rows),
            "manifest": str(manifest),
            "output": str(args.output),
            "forced_context_bbox_field": (
                args.forced_context_bbox_field or None
            ),
            "mean_reference_answer_nll": (
                sum(float(row["reference_answer_nll"]) for row in output_rows)
                / len(output_rows)
                if output_rows
                else None
            ),
            "mean_reference_answer_token_accuracy": (
                sum(
                    float(row["reference_answer_token_accuracy"])
                    for row in output_rows
                )
                / len(output_rows)
                if output_rows
                else None
            ),
            "mean_realized_visual_tokens": (
                sum(int(row["realized_visual_tokens"]) for row in output_rows)
                / len(output_rows)
                if output_rows
                else None
            ),
            "selector_checkpoint": (
                str(selector_checkpoint) if selector_checkpoint is not None else None
            ),
            "bridge_checkpoint": (
                str(bridge_checkpoint) if bridge_checkpoint is not None else None
            ),
            "scope": "training utility diagnostic; not benchmark accuracy",
        }
        args.output.with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
        return 0

    summary = summarize(
        output_rows,
        elapsed=time.perf_counter() - started,
        peak_reserved_mib=round(torch.cuda.max_memory_reserved(0) / 1024**2, 1),
    )
    summary.update(
        {
            "version": "evivit_v3_mid_bridge_eval_v1",
            "manifest": str(manifest),
            "model": str(model_path),
            "model_index_sha256": model_index_sha256,
            "checkpoint_key_mapping_count": checkpoint_key_mapping_count,
            "checkpoint_compatibility_base": checkpoint_compatibility_base,
            **{
                key: value
                for key, value in checkpoint_metadata.items()
                if key
                not in {
                    "safe_joint_residual_mixer",
                    "trace_calibrated_bridge_residual",
                }
            },
            "selector_checkpoint": (
                str(selector_checkpoint) if selector_checkpoint is not None else None
            ),
            "tracescale_checkpoint": (
                str(tracescale_head_path)
                if tracescale_head_path is not None
                else None
            ),
            "tracescale_checkpoint_sha256": (
                sha256_file(tracescale_head_path)
                if tracescale_head_path is not None
                else None
            ),
            "tracescale_checkpoint_version": (
                tracescale_checkpoint.get("version")
                if tracescale_checkpoint is not None
                else None
            ),
            "tracescale_positive_only": args.tracescale_positive_only,
            "tracescale_strength": args.tracescale_strength,
            "adaptive_box_checkpoint": (
                str(adaptive_box_head_path)
                if adaptive_box_head_path is not None
                else None
            ),
            "adaptive_box_checkpoint_sha256": (
                sha256_file(adaptive_box_head_path)
                if adaptive_box_head_path is not None
                else None
            ),
            "adaptive_box_checkpoint_version": (
                adaptive_box_checkpoint.get("version")
                if adaptive_box_checkpoint is not None
                else None
            ),
            "adaptive_box_safety_mode": args.adaptive_box_safety_mode,
            "adaptive_box_uncertainty_threshold": (
                args.adaptive_box_uncertainty_threshold
                if args.adaptive_box_safety_mode.startswith("uncertainty")
                else None
            ),
            "adaptive_box_ensemble_heads": [
                str(path) for path in adaptive_box_ensemble_paths
            ],
            "adaptive_box_residual_read_tokens": (
                args.adaptive_box_residual_read_tokens
                if args.adaptive_box_safety_mode == "residual_read"
                else None
            ),
            "trace_refine_checkpoint": (
                str(trace_refine_policy_path)
                if trace_refine_policy_path is not None
                else None
            ),
            "trace_refine_checkpoint_sha256": (
                sha256_file(trace_refine_policy_path)
                if trace_refine_policy_path is not None
                else None
            ),
            "trace_refine_checkpoint_version": (
                trace_refine_checkpoint.get("version")
                if trace_refine_checkpoint is not None
                else None
            ),
            "trace_refine_preview_expansion": args.trace_refine_preview_expansion,
            "trace_refine_preview_tokens": args.trace_refine_preview_tokens,
            "fine_patch_evidence_bias": args.fine_patch_evidence_bias,
            "fine_patch_evidence_floor": args.fine_patch_evidence_floor,
            "fine_patch_evidence_exponent": args.fine_patch_evidence_exponent,
            "fine_readout_mode": args.fine_readout_mode,
            "anchored_growth_calibrator": (
                str(anchored_growth_calibrator_path)
                if anchored_growth_calibrator_path is not None
                else None
            ),
            "anchored_growth_calibrator_sha256": (
                sha256_file(anchored_growth_calibrator_path)
                if anchored_growth_calibrator_path is not None
                else None
            ),
            "anchored_growth_calibrator_version": (
                anchored_growth_calibrator_checkpoint.get("version")
                if anchored_growth_calibrator_checkpoint is not None
                else None
            ),
            "anchored_growth_gate_threshold": (
                args.anchored_growth_gate_threshold
            ),
            "evicontour_utility_checkpoint": (
                str(evicontour_utility_path)
                if evicontour_utility_path is not None
                else None
            ),
            "evicontour_utility_sha256": (
                sha256_file(evicontour_utility_path)
                if evicontour_utility_path is not None
                else None
            ),
            "evicontour_utility_format_version": (
                evicontour_utility_checkpoint.get("format_version")
                if evicontour_utility_checkpoint is not None
                else None
            ),
            "evidence_policy": args.evidence_policy,
            "evidence_decoder": args.evidence_decoder,
            "seeded_basin_config": seeded_basin_config,
            "seeded_basin_config_path": (
                str(args.seeded_basin_config)
                if args.seeded_basin_config is not None
                else None
            ),
            "seeded_basin_config_sha256": (
                sha256_file(args.seeded_basin_config)
                if args.seeded_basin_config is not None
                else None
            ),
            "region_expansion_factor": args.region_expansion_factor,
            "topology_union_factor": args.topology_union_factor,
            "trace_split_context_fraction": (
                args.trace_split_context_fraction
                if args.evidence_decoder == "trace_split"
                else None
            ),
            "trace_split_context_expansion_factor": (
                args.trace_split_context_expansion_factor
                if args.evidence_decoder == "trace_split"
                else None
            ),
            "trace_split_context_decoder": (
                args.trace_split_context_decoder
                if args.evidence_decoder == "trace_split"
                else None
            ),
            "trace_split_budget_mode": (
                args.trace_split_budget_mode
                if args.evidence_decoder == "trace_split"
                else None
            ),
            "trace_split_min_context_fraction": (
                args.trace_split_min_context_fraction
                if args.evidence_decoder == "trace_split"
                and args.trace_split_budget_mode
                in {"competitive", "learned"}
                else None
            ),
            "trace_split_max_context_fraction": (
                args.trace_split_max_context_fraction
                if args.evidence_decoder == "trace_split"
                and args.trace_split_budget_mode
                in {"competitive", "learned"}
                else None
            ),
            "trace_split_adaptive_region_minimum": (
                args.trace_split_adaptive_region_minimum
            ),
            "trace_split_context_need_checkpoint": (
                str(context_need_checkpoint)
                if context_need_checkpoint is not None
                else None
            ),
            "trace_split_context_need_checkpoint_sha256": (
                sha256_file(context_need_checkpoint)
                if context_need_checkpoint is not None
                else None
            ),
            "token_serialization": args.token_serialization,
            "global_frame_joint_blocks": list(
                args.global_frame_joint_blocks
            ),
            "global_frame_coordinate_bins": (
                args.global_frame_coordinate_bins
            ),
            "global_frame_joint_topology": (
                args.global_frame_joint_topology
            ),
            "control_seed": args.control_seed,
            "global_token_budget": args.global_token_budget,
            "processor_min_pixels": args.processor_min_pixels,
            "processor_max_pixels": args.processor_max_pixels,
            "native_ratio_budget": args.native_ratio_budget,
            "native_need_ratio_budget": args.native_need_ratio_budget,
            "native_global_ratio": (
                args.native_global_ratio
                if args.native_ratio_budget or args.native_need_ratio_budget
                else None
            ),
            "native_fine_ratio": (
                args.native_fine_ratio if args.native_ratio_budget else None
            ),
            "native_min_fine_ratio": (
                args.native_min_fine_ratio
                if args.native_need_ratio_budget
                else None
            ),
            "native_max_fine_ratio": (
                args.native_max_fine_ratio
                if args.native_need_ratio_budget
                else None
            ),
            "native_ratio_min_global_tokens": (
                args.native_ratio_min_global_tokens
            ),
            "native_ratio_min_fine_tokens": args.native_ratio_min_fine_tokens,
            "native_ratio_soft_floor_tokens": (
                args.native_ratio_soft_floor_tokens
            ),
            "native_ratio_alignment": args.native_ratio_alignment,
            "adaptive_native_global_budget": args.adaptive_native_global_budget,
            "adaptive_global_native_fraction": (
                args.adaptive_global_native_fraction
            ),
            "adaptive_global_base_token_budget": (
                args.adaptive_global_base_token_budget
            ),
            "adaptive_global_min_token_budget": (
                args.adaptive_global_min_token_budget
            ),
            "adaptive_global_max_token_budget": (
                args.adaptive_global_max_token_budget
            ),
            "adaptive_global_maximum_pixels": (
                args.adaptive_global_maximum_pixels
            ),
            "adaptive_global_budget_alignment": (
                args.adaptive_global_budget_alignment
            ),
            "fine_token_budget": args.fine_token_budget,
            "evitree_need_head": (
                str(evitree_need_head_path)
                if evitree_need_head_path is not None
                else None
            ),
            "evitree_need_head_sha256": (
                sha256_file(evitree_need_head_path)
                if evitree_need_head_path is not None
                else None
            ),
            "evitree_min_fine_token_budget": (
                args.evitree_min_fine_token_budget
            ),
            "evitree_max_fine_token_budget": (
                args.evitree_max_fine_token_budget
            ),
            "evitree_budget_alignment": args.evitree_budget_alignment,
            "evitree_tree_allocation": args.evitree_tree_allocation,
            "evitree_tree_max_depth": args.evitree_tree_max_depth,
            "evitree_tree_max_leaves": args.evitree_tree_max_leaves,
            "evitree_tree_reliability": args.evitree_tree_reliability,
            "evitree_tree_target_mass": args.evitree_tree_target_mass,
            "evitree_split_policy": (
                str(evitree_split_policy_path)
                if evitree_split_policy_path is not None
                else None
            ),
            "evitree_split_policy_sha256": (
                sha256_file(evitree_split_policy_path)
                if evitree_split_policy_path is not None
                else None
            ),
            "evitree_split_threshold": evitree_split_threshold,
            "evitree_sparse_branches": args.evitree_sparse_branches,
            "evitree_maximum_branches": args.evitree_maximum_branches,
            "evitree_minimum_branch_depth": (
                args.evitree_minimum_branch_depth
            ),
            "evitree_branch_density_power": (
                args.evitree_branch_density_power
            ),
            "evitree_recenter_branch_tips": (
                args.evitree_recenter_branch_tips
            ),
            "evitree_branch_expansion_factor": (
                args.evitree_branch_expansion_factor
            ),
            "evitree_maximum_branch_iou": (
                args.evitree_maximum_branch_iou
            ),
            "v10_gate_mode": args.v10_gate_mode,
            "v10_reliability_report": (
                str(reliability_report_path)
                if reliability_report_path is not None
                else None
            ),
            "v10_min_fine_token_budget": args.v10_min_fine_token_budget,
            "v10_map_mode": args.v10_map_mode,
            "v10_budget_mode": args.v10_budget_mode,
            "v10_bridge_gate_mode": args.v10_bridge_gate_mode,
            "native_fit_total_token_budget": args.native_fit_total_token_budget,
            "native_fit_bypass_rows": sum(
                bool(row.get("native_fit_bypass", False)) for row in output_rows
            ),
            "patch_exchange": args.patch_exchange,
            "patch_exchange_tree": args.patch_exchange_tree,
            "patch_exchange_global_scale": (
                args.patch_exchange_global_scale
                if args.patch_exchange
                else None
            ),
            "patch_exchange_local_scale": (
                args.patch_exchange_local_scale
                if args.patch_exchange
                else None
            ),
            "patch_exchange_total_cap_ratio": (
                args.patch_exchange_total_cap_ratio
                if args.patch_exchange
                else None
            ),
            "patch_exchange_soft_floor_tokens": (
                args.patch_exchange_soft_floor_tokens
                if args.patch_exchange
                else None
            ),
            "patch_exchange_balanced_soft_floor": (
                args.patch_exchange_balanced_soft_floor
                if args.patch_exchange
                else None
            ),
            "patch_exchange_preserve_native_global": (
                args.patch_exchange_preserve_native_global
                if args.patch_exchange
                else None
            ),
            "patch_exchange_continuous_soft_floor": (
                args.patch_exchange_continuous_soft_floor
                if args.patch_exchange
                else None
            ),
            "patch_exchange_continuous_soft_floor_max_tokens": (
                args.patch_exchange_continuous_soft_floor_max_tokens
                if args.patch_exchange_continuous_soft_floor
                else None
            ),
            "patch_exchange_continuous_soft_floor_ramp_start_tokens": (
                args.patch_exchange_continuous_soft_floor_ramp_start_tokens
                if args.patch_exchange_continuous_soft_floor
                else None
            ),
            "patch_exchange_continuous_soft_floor_ramp_end_tokens": (
                args.patch_exchange_continuous_soft_floor_ramp_end_tokens
                if args.patch_exchange_continuous_soft_floor
                else None
            ),
            "patch_exchange_native_global_anchor": (
                args.patch_exchange_native_global_anchor
                if args.patch_exchange
                else None
            ),
            "patch_exchange_additive_fine_ratio": (
                args.patch_exchange_additive_fine_ratio
                if args.patch_exchange_native_global_anchor
                else None
            ),
            "patch_exchange_additive_fine_max_tokens": (
                args.patch_exchange_additive_fine_max_tokens
                if args.patch_exchange_native_global_anchor
                else None
            ),
            "patch_exchange_mean_effective_soft_floor_tokens": (
                sum(
                    int(row["patch_exchange_effective_soft_floor_tokens"])
                    for row in output_rows
                )
                / len(output_rows)
                if args.patch_exchange and output_rows
                else None
            ),
            "patch_exchange_view_min_pixels": (
                args.patch_exchange_view_min_pixels
                if args.patch_exchange
                else None
            ),
            "patch_exchange_mean_native_tokens": (
                sum(int(row["native_visual_tokens"]) for row in output_rows)
                / len(output_rows)
                if args.patch_exchange and output_rows
                else None
            ),
            "patch_exchange_mean_unused_tokens": (
                sum(
                    int(row["patch_exchange_fine_plan"]["unused_tokens"])
                    for row in output_rows
                )
                / len(output_rows)
                if args.patch_exchange and output_rows
                else None
            ),
            "native_pixel_contract": args.native_pixel_contract,
            "native_fuse": args.native_fuse,
            "native_fuse_strength": (
                args.native_fuse_strength if args.native_fuse else None
            ),
            "continuous_native_global": args.continuous_native_global,
            "continuous_global_base_pixels": (
                args.continuous_global_base_pixels
                if args.continuous_native_global
                else None
            ),
            "continuous_global_max_pixels": (
                args.continuous_global_max_pixels
                if args.continuous_native_global
                else None
            ),
            "continuous_global_exponent": (
                args.continuous_global_exponent
                if args.continuous_native_global
                else None
            ),
            "visual_output_mode": args.visual_output_mode,
            "bridge_mode": bridge_mode,
            "coverage_masked_parent_writeback": (
                args.coverage_masked_parent_writeback
            ),
            "global_processor_max_pixels": (
                global_processor_max_pixels
                if args.native_pixel_contract
                else None
            ),
            "fine_processor_max_pixels": (
                fine_processor_max_pixels
                if args.native_pixel_contract
                else None
            ),
            "stage_d_v27b_safe_caps": args.stage_d_v27b_safe_caps,
            "stage_d_v27b_context_fraction": (
                args.stage_d_v27b_context_fraction
                if args.stage_d_v27b_safe_caps
                else None
            ),
            "native_pixel_global_grid_preservation_rate": (
                sum(
                    bool(row["native_pixel_global_plan"]["native_grid_preserved"])
                    for row in output_rows
                )
                / len(output_rows)
                if args.native_pixel_contract and output_rows
                else None
            ),
            "native_pixel_global_maximum_application_rate": (
                sum(
                    bool(row["native_pixel_global_plan"]["maximum_applied"])
                    for row in output_rows
                )
                / len(output_rows)
                if args.native_pixel_contract and output_rows
                else None
            ),
            "native_pixel_mean_fine_native_grid_preserved_views": (
                sum(
                    sum(
                        bool(plan["native_grid_preserved"])
                        for plan in row["native_pixel_fine_plans"]
                    )
                    for row in output_rows
                )
                / len(output_rows)
                if args.native_pixel_contract and output_rows
                else None
            ),
            "native_token_caps": args.native_token_caps,
            "native_cap_scope": (
                args.native_cap_scope if args.native_token_caps else None
            ),
            "native_cap_minimum_fine_tokens": (
                args.native_cap_minimum_fine_tokens
                if native_fine_cap
                else None
            ),
            "native_cap_minimum_global_tokens": (
                args.native_cap_minimum_global_tokens
                if native_global_cap
                else None
            ),
            "native_cap_anchor_fine_tokens": (
                args.native_cap_anchor_fine_tokens
                if native_fine_cap
                else None
            ),
            "native_cap_fine_floor_native_ratio": (
                args.native_cap_fine_floor_native_ratio
                if native_fine_cap
                else None
            ),
            "native_cap_minimum_fine_total": (
                args.native_cap_minimum_fine_total
                if native_fine_cap
                else None
            ),
            "evicontour_csr_config": expected_csr_config,
            "native_cap_global_ceiling": (
                args.global_token_budget if native_global_cap else None
            ),
            "native_cap_shared_fine_ceiling": (
                args.fine_token_budget if native_fine_cap else None
            ),
            "native_cap_mean_unused_global_tokens": (
                sum(
                    int(row["native_cap_global_unused_tokens"])
                    for row in output_rows
                )
                / len(output_rows)
                if native_global_cap and output_rows
                else None
            ),
            "native_cap_mean_unused_fine_tokens": (
                sum(
                    int(row["native_cap_fine_unused_tokens"])
                    for row in output_rows
                )
                / len(output_rows)
                if native_fine_cap and output_rows
                else None
            ),
            "native_cap_global_saturation_rate": (
                sum(
                    bool(row["native_cap_global_saturated"])
                    for row in output_rows
                )
                / len(output_rows)
                if native_global_cap and output_rows
                else None
            ),
            "native_cap_global_grid_preservation_rate": (
                sum(
                    bool(row["native_cap_global_grid_preserved"])
                    for row in output_rows
                )
                / len(output_rows)
                if native_global_cap and output_rows
                else None
            ),
            "native_cap_mean_fine_saturated_views": (
                sum(
                    int(row["native_cap_fine_saturated_views"])
                    for row in output_rows
                )
                / len(output_rows)
                if native_fine_cap and output_rows
                else None
            ),
            "native_cap_mean_fine_native_grid_preserved_views": (
                sum(
                    int(
                        row[
                            "native_cap_fine_native_grid_preserved_views"
                        ]
                    )
                    for row in output_rows
                )
                / len(output_rows)
                if native_fine_cap and output_rows
                else None
            ),
            "max_regions": args.max_regions,
            "deterministic_inference": args.deterministic_inference,
            "attention_backend": (
                deterministic_backend_name(
                    args.deterministic_attention_backend,
                    ptea_auxiliary_attention=args.evidence_policy == "mid_ptea",
                )
                if args.deterministic_inference
                else "sdpa_default"
            ),
            "record_timing": args.record_timing,
            "record_answer_confidence": args.record_answer_confidence,
            "latency": (
                summarize_latency(
                    output_rows, warmup_samples=args.timing_warmup_samples
                )
                if args.record_timing
                else None
            ),
            "mean_visual_tokens": (
                sum(int(row["realized_visual_tokens"]) for row in output_rows)
                / len(output_rows)
                if output_rows
                else 0.0
            ),
            "mean_requested_global_tokens": (
                sum(
                    int(
                        row.get(
                            "requested_global_token_budget",
                            row["global_token_budget"],
                        )
                    )
                    for row in output_rows
                )
                / len(output_rows)
                if output_rows
                else 0.0
            ),
            "mean_realized_global_tokens": (
                sum(int(row["realized_global_tokens"]) for row in output_rows)
                / len(output_rows)
                if output_rows
                else 0.0
            ),
            "mean_context_map_entropy": (
                sum(
                    float(row["context_map_entropy"])
                    for row in output_rows
                    if row.get("context_map_entropy") is not None
                )
                / sum(
                    row.get("context_map_entropy") is not None
                    for row in output_rows
                )
                if any(
                    row.get("context_map_entropy") is not None
                    for row in output_rows
                )
                else None
            ),
            "mean_relative_residual_l2": (
                sum(float(row["relative_residual_l2"]) for row in output_rows)
                / len(output_rows)
                if output_rows
                else 0.0
            ),
            "mean_v10_reliability_gate": (
                sum(
                    float(row["v10_reliability_gate"])
                    for row in output_rows
                    if row.get("v10_reliability_gate") is not None
                )
                / sum(
                    row.get("v10_reliability_gate") is not None
                    for row in output_rows
                )
                if any(
                    row.get("v10_reliability_gate") is not None
                    for row in output_rows
                )
                else None
            ),
            "mean_realized_fine_tokens": (
                sum(int(row["realized_fine_tokens"]) for row in output_rows)
                / len(output_rows)
                if output_rows
                else 0.0
            ),
            "mean_fine_encoder_tokens": (
                sum(
                    int(
                        row.get(
                            "realized_fine_encoder_tokens",
                            row["realized_fine_tokens"],
                        )
                    )
                    for row in output_rows
                )
                / len(output_rows)
                if output_rows
                else 0.0
            ),
        }
    )
    if language_lora_adapter_audit is not None:
        summary.update(
            {
                "language_lora_adapter": str(language_lora_adapter_path),
                "language_lora_adapter_config_sha256": (
                    language_lora_adapter_audit.config_sha256
                ),
                "language_lora_adapter_weights_sha256": (
                    language_lora_adapter_audit.weights_sha256
                ),
                "language_lora_adapter_bundle_fingerprint": (
                    language_lora_adapter_audit.bundle_fingerprint
                ),
            }
        )
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
