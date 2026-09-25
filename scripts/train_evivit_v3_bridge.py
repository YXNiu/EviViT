#!/usr/bin/env python3
"""Train only the sparse EviViT-v3 mid-ViT evidence bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.extract_qwen_dense_features import build_projection  # noqa: E402
from scripts.predict_patch_text_topk_boxes import load_selector  # noqa: E402
from evivit_core.evivit import (  # noqa: E402
    RegionBudget,
    grid_for_token_budget,
    quantize_mrope_coordinates,
)
from evivit_core.evivit_adaptive_box import (  # noqa: E402
    apply_residual,
    load_adaptive_box_head,
)
from evivit_core.evivit_adaptive_box_features import (  # noqa: E402
    adaptive_box_feature_vectors,
)
from evivit_core.evivit_evidence_bridge import SparseGlobalLocalEvidenceBridge  # noqa: E402
from evivit_core.evivit_trace_calibrated_bridge import (  # noqa: E402
    TraceCalibratedBridgeResidual,
)
from evivit_core.evislot import EvidenceSlotCompressor  # noqa: E402
from evivit_core.evivit_native_scale import (  # noqa: E402
    plan_continuous_native_global_view,
    plan_native_pixel_view,
)
from evivit_core.evivit_patch_exchange import (  # noqa: E402
    continuous_soft_floor_tokens,
    plan_patch_exchange_fine_views,
    plan_patch_exchange_global_view,
)
from evivit_core.evivit_posttraining import (  # noqa: E402
    LANGUAGE_LORA_TARGET_PATTERN,
    warmup_steps,
)
from evivit_core.evivit_grpo import (  # noqa: E402
    compare_policy_reference_weights,
    completion_log_probs,
    group_advantages,
    grpo_clipped_loss,
    load_semantic_reward_judge,
    outcome_reward,
    prepare_evivit_visual_prompt,
    sample_completions,
    semantic_reward_judge_batch,
    SEMANTIC_REWARD_JUDGE_VERSION,
    strict_relaxed_reward_match,
    synchronize_policy_reference_weights,
)
from evivit_core.final_round_reward import parse_answer, binary_reward, reliable_judge, active_group
from evivit_core.evivit_recovery_sft import (  # noqa: E402
    RECOVERY_ADAMW_BETAS,
    RECOVERY_ADAMW_EPS,
    RECOVERY_EPOCH_ORDER_VERSION,
    assert_formal_p3_contract,
    assert_recovery_trainable_scope,
    epoch_order,
    half_epoch_micro_steps,
    optimizer_groups as recovery_optimizer_groups,
    recovery_lane,
)
from evivit_core.evivit_recovery_router_sft import (  # noqa: E402
    H_SAFE_GEOMETRY_LOSS_VERSION,
    bbox_soft_map,
    h_safe_geometry_loss,
    load_aligned_router_rows,
    load_h_safe_geometry_mask,
    ptea_recovery_loss,
)
from evivit_core.eviblend import SafeJointResidualMixer  # noqa: E402
from evivit_core.evicontour_utility import (  # noqa: E402
    load_evicontour_utility_head,
)
from evivit_core.evisplit_context_need import load_context_need_head  # noqa: E402
from evivit_core.evirelay import EviRelayBlock, EviRelayBridgeAdapter  # noqa: E402
from evivit_core.evitree import (  # noqa: E402
    load_hierarchical_split_policy_head,
    load_scalar_evidence_need_head,
    plan_native_ratio_budget,
    plan_native_need_spread_budget,
    scalar_need_feature_vector,
)
from evivit_core.evivit_v3_online import (  # noqa: E402
    allocate_control_evidence,
    allocate_evitree_hierarchical_evidence,
    allocate_mid_ptea_evidence,
    allocate_trace_split_evidence,
    forward_mid_ptea_training_paths,
    qwen_projected_question_tokens,
)
from evivit_core.geometry import policy_to_pixels  # noqa: E402
from evivit_core.gpu_safety import check_gpu  # noqa: E402
from evivit_core.qwen_family import (  # noqa: E402
    language_model_visual_forward,
    load_qwen_vlm,
    vision_probe_text,
)
from evivit_core.qwen3vl_evivit_v3_mid_encoder import (  # noqa: E402
    MidViTFineViewInput,
    Qwen3VLEviViTV3MidEncoder,
)
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
from evivit_core.trace_supervision import (  # noqa: E402
    combine_trace_maps,
    sample_trace_distribution,
    soft_distribution_cross_entropy,
)


SYSTEM_PROMPT = (
    "You are a careful fine-grained visual question answering assistant. "
    "Answer using only the image evidence. Return compact JSON only."
)
DIRECT_SYSTEM_PROMPT = (
    "Answer the visual question using the image. "
    "Return only the concise final answer."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_question(value: str) -> str:
    return " ".join(str(value).replace("<image>", " ").split())


def direct_prediction(raw: str) -> tuple[str, str | None]:
    """Parse either the formal compact-JSON answer or a direct answer.

    The P3 recovery prompt uses ``{"answer": ...}``, while the matched Base
    lane emits the answer directly.  Reward both protocols by their semantic
    answer rather than by serialization punctuation.
    """

    value = str(raw).strip()
    if not value:
        return "", "empty_answer"
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return value, None
    if isinstance(payload, dict):
        if payload.get("answer") is None:
            return "", "missing_answer"
        return str(payload["answer"]).strip(), None
    if isinstance(payload, (str, int, float, bool)):
        return str(payload).strip(), None
    return value, None


def normalized_answer(value: str) -> str:
    """Normalize an answer without importing the heavyweight evaluation CLI."""

    return "".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def build_chat_text(
    processor: Any,
    question: str,
    answer: str | None,
    *,
    answer_protocol: str = "compact_json",
    enable_thinking: bool | None = None,
) -> str:
    if answer_protocol not in {
        "compact_json",
        "direct_concise",
        "vero_native",
        "blink_mcq",
        "vision_opd_official",
    }:
        raise ValueError(f"unknown answer_protocol: {answer_protocol}")
    if answer_protocol == "blink_mcq":
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": str(question).strip()},
                ],
            }
        ]
        if answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(answer).strip()}],
                }
            )
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": answer is None,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        return processor.apply_chat_template(messages, **template_kwargs)
    if answer_protocol == "vero_native":
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": clean_question(question)},
                ],
            }
        ]
        if answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"<answer>\\boxed{{{str(answer).strip()}}}</answer>",
                        }
                    ],
                }
            )
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": answer is None,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        return processor.apply_chat_template(messages, **template_kwargs)
    if answer_protocol == "vision_opd_official":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": str(question).replace("<image>", "").strip(),
                    },
                ],
            }
        ]
        if answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(answer).strip()}],
                }
            )
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": answer is None,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        return processor.apply_chat_template(messages, **template_kwargs)
    if answer_protocol == "direct_concise":
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": DIRECT_SYSTEM_PROMPT}],
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
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(answer).strip()}],
                }
            )
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": answer is None,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        return processor.apply_chat_template(messages, **template_kwargs)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f"Question: {clean_question(question)}\n"
                        'Return exactly one compact JSON object: {"answer":"..."}\n'
                        "Do not explain."
                    ),
                },
            ],
        },
    ]
    if answer is not None:
        target = json.dumps(
            {"answer": str(answer).strip()},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": target}]}
        )
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": answer is None,
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    return processor.apply_chat_template(messages, **template_kwargs)


def build_multi_chat_text(
    processor: Any,
    question: str,
    image_count: int,
    answer: str | None,
    *,
    answer_protocol: str = "compact_json",
    enable_thinking: bool | None = None,
) -> str:
    """Build the same answer protocol with ordered image placeholders.

    Images are independent observations in their official order.  No figure
    captions or contact-sheet conversion are introduced by the evaluator.
    """

    if image_count <= 0:
        raise ValueError("image_count must be positive")
    if answer_protocol == "muir_mcq":
        segments = str(question).split("<image>")
        if len(segments) - 1 != image_count:
            raise ValueError(
                "MuirBench prompt/image mismatch: "
                f"{len(segments) - 1} placeholders for {image_count} images"
            )
        content: list[dict[str, Any]] = []
        if segments[0]:
            content.append({"type": "text", "text": segments[0]})
        for segment in segments[1:]:
            content.append({"type": "image"})
            if segment:
                content.append({"type": "text", "text": segment})
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        if answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(answer).strip()}],
                }
            )
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": answer is None,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        return processor.apply_chat_template(messages, **template_kwargs)
    if answer_protocol == "blink_mcq":
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in range(image_count)],
                    {"type": "text", "text": str(question).strip()},
                ],
            }
        ]
        if answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(answer).strip()}],
                }
            )
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": answer is None,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        return processor.apply_chat_template(messages, **template_kwargs)
    if image_count == 1:
        return build_chat_text(
            processor,
            question,
            answer,
            answer_protocol=answer_protocol,
            enable_thinking=enable_thinking,
        )
    image_content = [{"type": "image"} for _ in range(image_count)]
    if answer_protocol in {"vero_native", "vision_opd_official"}:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    *image_content,
                    {
                        "type": "text",
                        "text": (
                            str(question).replace("<image>", "").strip()
                            if answer_protocol == "vision_opd_official"
                            else clean_question(question)
                        ),
                    },
                ],
            }
        ]
        if answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                str(answer).strip()
                                if answer_protocol == "vision_opd_official"
                                else f"<answer>\\boxed{{{str(answer).strip()}}}</answer>"
                            ),
                        }
                    ],
                }
            )
    elif answer_protocol == "direct_concise":
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": DIRECT_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    *image_content,
                    {"type": "text", "text": clean_question(question)},
                ],
            },
        ]
        if answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(answer).strip()}],
                }
            )
    elif answer_protocol == "compact_json":
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    *image_content,
                    {
                        "type": "text",
                        "text": (
                            f"Question: {clean_question(question)}\n"
                            'Return exactly one compact JSON object: {"answer":"..."}\n'
                            "Do not explain."
                        ),
                    },
                ],
            },
        ]
        if answer is not None:
            target = json.dumps(
                {"answer": str(answer).strip()},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": target}]}
            )
    else:
        raise ValueError(f"unknown answer_protocol: {answer_protocol}")
    template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": answer is None,
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    return processor.apply_chat_template(messages, **template_kwargs)


def expand_image_placeholder(
    token_ids: Sequence[int], image_token_id: int, visual_tokens: int
) -> tuple[list[int], int, int]:
    positions = [index for index, value in enumerate(token_ids) if value == image_token_id]
    if len(positions) != 1:
        raise RuntimeError(f"expected one image placeholder, found {len(positions)}")
    start = positions[0]
    expanded = (
        list(token_ids[:start])
        + [image_token_id] * visual_tokens
        + list(token_ids[start + 1 :])
    )
    return expanded, start, start + visual_tokens


def build_position_ids(
    sequence_length: int,
    visual_start: int,
    visual_end: int,
    visual_coordinates: np.ndarray,
    *,
    device: str,
) -> torch.Tensor:
    visual_tokens = visual_end - visual_start
    if visual_coordinates.shape != (3, visual_tokens):
        raise ValueError("visual coordinate count does not match the visual span")
    positions = torch.zeros(3, sequence_length, dtype=torch.long, device=device)
    if visual_start:
        prefix = torch.arange(visual_start, device=device)
        positions[:, :visual_start] = prefix
    coordinates = torch.from_numpy(visual_coordinates).to(
        device=device, dtype=torch.long
    )
    positions[:, visual_start:visual_end] = coordinates + visual_start
    suffix_base = int(positions[:, visual_start:visual_end].max().item()) + 1
    suffix_length = sequence_length - visual_end
    if suffix_length:
        suffix = torch.arange(suffix_base, suffix_base + suffix_length, device=device)
        positions[:, visual_end:] = suffix
    return positions.unsqueeze(1)


def resize_to_token_grid(
    image: Image.Image,
    grid_h: int,
    grid_w: int,
    *,
    merged_patch_stride: int,
) -> Image.Image:
    target = (grid_w * merged_patch_stride, grid_h * merged_patch_stride)
    if image.size == target:
        return image
    downsample = target[0] < image.width or target[1] < image.height
    return image.resize(
        target,
        Image.Resampling.LANCZOS if downsample else Image.Resampling.BICUBIC,
    )


def visual_coordinates(vision: Any, *, coordinate_bins: int, scale_bins: int) -> np.ndarray:
    metadata = {
        "centers_xy": vision.token_centers_xy.detach().float().cpu().numpy(),
        "levels": vision.token_levels.detach().cpu().numpy(),
        "scales": vision.token_scales.detach().float().cpu().numpy(),
    }
    return quantize_mrope_coordinates(
        metadata,
        coordinate_bins=coordinate_bins,
        scale_bins=scale_bins,
        mode="original_xy_scale_t",
    )


def native_global_visual_coordinates(vision: Any) -> np.ndarray:
    """Recover Qwen's native static-image MRoPE grid for Global-only output.

    Mixed-resolution EviViT variants quantize source coordinates because the
    emitted sequence contains several independently resized views. A
    parent-preserving Global-only route instead emits the complete native
    Global raster in its original row-major order. Reusing mixed 128-bin
    coordinates would silently break the exact B16 identity fallback even
    when the evidence bridge is zero initialized.
    """

    if str(getattr(vision, "visual_output_mode", "")) != "global_only":
        raise ValueError("native Global MRoPE requires visual_output_mode=global_only")
    if int(getattr(vision, "fine_image_tokens", -1)) != 0:
        raise ValueError("native Global MRoPE cannot serialize Fine tokens")
    centers = vision.token_centers_xy.detach().float().cpu().numpy()
    levels = vision.token_levels.detach().cpu().numpy()
    if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] == 0:
        raise ValueError("native Global centers must have shape [tokens, 2]")
    if np.any(levels != 0):
        raise ValueError("native Global MRoPE expects only level-zero tokens")
    # Token centers are assembled through several BF16/FP32 geometry paths.
    # Five decimal places remain far below the smallest Qwen grid spacing in
    # the supported 16M pixel contract while collapsing harmless round-off.
    grid_h = int(getattr(vision, "global_grid_h", 0))
    grid_w = int(getattr(vision, "global_grid_w", 0))
    if grid_h <= 0 or grid_w <= 0:
        unique_x = np.unique(np.round(centers[:, 0], decimals=5))
        unique_y = np.unique(np.round(centers[:, 1], decimals=5))
        grid_h = int(unique_y.size)
        grid_w = int(unique_x.size)
    if grid_h * grid_w != int(centers.shape[0]):
        raise ValueError(
            "Global tokens do not form one complete native raster: "
            f"tokens={centers.shape[0]} unique_y={grid_h} unique_x={grid_w}"
        )
    rows, columns = np.meshgrid(
        np.arange(grid_h, dtype=np.int64),
        np.arange(grid_w, dtype=np.int64),
        indexing="ij",
    )
    rows = rows.reshape(-1)
    columns = columns.reshape(-1)
    return np.stack([np.zeros_like(rows), rows, columns], axis=0)


def activate_policy_adapter(model: Any, name: str, *, trainable: bool) -> None:
    """Switch one PEFT adapter without changing the frozen host."""

    model.base_model.set_adapter(name, inference_mode=not trainable)
    if model.base_model.active_adapter != name:
        raise RuntimeError(f"failed to activate adapter {name}")


def configure_grpo_language_checkpointing(
    language_model: Any, *, enabled: bool
) -> None:
    """Toggle recomputation around sampling, which requires an active KV cache."""

    if enabled:
        language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        language_model.train()
        for module in language_model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        language_model.config.use_cache = False
    else:
        if hasattr(language_model, "gradient_checkpointing_disable"):
            language_model.gradient_checkpointing_disable()
        language_model.eval()
        language_model.config.use_cache = True


def answer_only_loss(
    model: Any,
    processor: Any,
    question: str,
    answer: str,
    vision: Any,
    *,
    device: str,
    coordinate_bins: int,
    scale_bins: int,
    answer_protocol: str = "compact_json",
    native_global_mrope: bool = False,
    enable_thinking: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    tokenizer = processor.tokenizer
    prompt_text = build_chat_text(
        processor,
        question,
        None,
        answer_protocol=answer_protocol,
        enable_thinking=enable_thinking,
    )
    full_text = build_chat_text(
        processor,
        question,
        answer,
        answer_protocol=answer_protocol,
        enable_thinking=enable_thinking,
    )
    # Some chat templates (notably Qwen3.5) end the generation prompt with a
    # newline whose tokenization changes when the completed assistant message
    # adds another newline and </think>.  The rendered text is still a strict
    # continuation, while its last prompt token is not necessarily a token-
    # level prefix.  Labels below are located from full-text offsets and do not
    # depend on prompt tokenization, so validate the actual text contract.
    if not full_text.startswith(prompt_text):
        raise RuntimeError("assistant target is not a strict text continuation")
    full_encoding = tokenizer(
        full_text, add_special_tokens=False, return_offsets_mapping=True
    )
    full_ids = full_encoding.input_ids
    if answer_protocol == "compact_json":
        target_json = json.dumps(
            {"answer": str(answer).strip()}, ensure_ascii=False, separators=(",", ":")
        )
        encoded_value = json.dumps(str(answer).strip(), ensure_ascii=False)
        target_start = full_text.rfind(target_json)
        value_offset = target_json.find(encoded_value)
        if target_start < 0 or value_offset < 0:
            raise RuntimeError("could not locate answer value in rendered chat")
        value_start = target_start + value_offset
        value_end = value_start + len(encoded_value)
        if encoded_value.startswith('"') and encoded_value.endswith('"'):
            value_start += 1
            value_end -= 1
    else:
        answer_value = str(answer).strip()
        value_start = full_text.rfind(answer_value)
        if value_start < 0:
            raise RuntimeError("could not locate direct answer in rendered chat")
        value_end = value_start + len(answer_value)
    raw_labels = [
        token_id if max(start, value_start) < min(end, value_end) else -100
        for token_id, (start, end) in zip(full_ids, full_encoding.offset_mapping)
    ]

    visual_tokens = int(vision.image_embeds.shape[0])
    expanded, visual_start, visual_end = expand_image_placeholder(
        full_ids, int(model.config.image_token_id), visual_tokens
    )
    image_position = next(
        index
        for index, value in enumerate(full_ids)
        if value == int(model.config.image_token_id)
    )
    expanded_labels = (
        raw_labels[:image_position]
        + [-100] * visual_tokens
        + raw_labels[image_position + 1 :]
    )
    input_ids = torch.tensor([expanded], dtype=torch.long, device=device)
    labels = torch.tensor([expanded_labels], dtype=torch.long, device=device)
    answer_tokens = int(labels.ne(-100).sum().item())
    if answer_tokens == 0:
        raise RuntimeError("answer contains no supervised tokens")
    attention_mask = torch.ones_like(input_ids)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    visual_mask = input_ids.eq(int(model.config.image_token_id))
    inputs_embeds = inputs_embeds.masked_scatter(
        visual_mask.unsqueeze(-1), vision.image_embeds.to(inputs_embeds.dtype)
    )
    position_ids = build_position_ids(
        input_ids.shape[1],
        visual_start,
        visual_end,
        (
            native_global_visual_coordinates(vision)
            if native_global_mrope
            else visual_coordinates(
                vision, coordinate_bins=coordinate_bins, scale_bins=scale_bins
            )
        ),
        device=device,
    )
    outputs = language_model_visual_forward(
        model,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_mask=visual_mask,
        deepstack_visual_embeds=[
            feature.to(inputs_embeds.dtype) for feature in vision.deepstack_features
        ],
        use_cache=False,
    )
    shift_labels = labels[:, 1:].contiguous()
    supervised = shift_labels.ne(-100)
    selected_hidden = outputs.last_hidden_state[:, :-1][supervised]
    selected_targets = shift_labels[supervised]
    # Project only supervised answer positions.  This is algebraically
    # identical to full-sequence CE with ignore_index=-100, but avoids a
    # [~5K visual positions × vocabulary] tensor that wastes several GB.
    selected_logits = model.lm_head(selected_hidden)
    loss = F.cross_entropy(selected_logits.float(), selected_targets)
    with torch.no_grad():
        token_accuracy = (
            selected_logits.argmax(dim=-1).eq(selected_targets).float().mean()
        )
    return loss, token_accuracy, answer_tokens


def trace_calibrated_checkpoint_version(
    amplitude_mode: str,
    *,
    preserve_fp32_importance: bool = False,
) -> str:
    if amplitude_mode == "positive_centered":
        if preserve_fp32_importance:
            return (
                "trace_calibrated_bridge_centered_"
                "fp32importance_fixed5k_v3"
            )
        return "trace_calibrated_bridge_centered_fixed5k_v2"
    if amplitude_mode == "signed_zero":
        return "trace_calibrated_bridge_fixed5k_v1"
    raise ValueError(f"unsupported trace calibration amplitude mode: {amplitude_mode}")


def save_checkpoint(
    path: Path,
    bridge: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    config: dict[str, Any],
    safe_joint_residual_mixer: SafeJointResidualMixer | None = None,
    trace_calibrated_bridge_residual: (
        TraceCalibratedBridgeResidual | None
    ) = None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fusion = str(config.get("fusion", "sparse_bridge"))
    trace_residual = bool(config.get("trace_residual_modulation", False))
    version = {
        "sparse_bridge": "evivit_v3_sparse_mid_bridge_v1",
        "tracefovea": "tracefovea_fixed5k_v1",
        "tracefovea_human_gate": (
            "tracefovea_trace_residual_fixed5k_v1"
            if trace_residual
            else "tracefovea_human_gate_fixed5k_v1"
        ),
        "eviweave_human_gate": "eviweave_human_gate_fixed5k_v1",
        "eviweave_normalized": "eviweave_normalized_fixed5k_v2",
        "evirelay": "evirelay_fixed5k_v1",
        "eviblend_safe_joint": "eviblend_safe_joint_fixed5k_v1",
        "evislot": "evislot_midvit_fixedslots_v1",
        "trace_calibrated_bridge": (
            trace_calibrated_checkpoint_version(
                str(
                    config.get(
                        "trace_calibration_amplitude_mode", "signed_zero"
                    )
                ),
                preserve_fp32_importance=bool(
                    config.get(
                        "trace_calibration_preserve_fp32_importance", False
                    )
                ),
            )
        ),
    }[fusion]
    if bool(config.get("evitree_v2", False)):
        if fusion == "tracefovea":
            version = "evitree_v2_parent_fusion_v1"
        elif fusion == "tracefovea_human_gate":
            version = "evitree_v2_parent_fusion_human_gate_v1"
        else:
            raise ValueError("unsupported EviTree-v2 fusion checkpoint")
    payload = {
            "version": version,
            "step": step,
            "bridge": bridge.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "python_rng_state": random.getstate(),
        }
    if safe_joint_residual_mixer is not None:
        payload["safe_joint_residual_mixer"] = (
            safe_joint_residual_mixer.state_dict()
        )
    if trace_calibrated_bridge_residual is not None:
        payload["trace_calibrated_bridge_residual"] = (
            trace_calibrated_bridge_residual.state_dict()
        )
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_fingerprint(value: dict[str, Any]) -> str:
    # Keep training checkpoints byte-for-byte compatible with the canonical
    # verifier used by ``evivit_core.evivit_recovery_router_sft``.  The explicit
    # separators matter: the old implementation included JSON whitespace and
    # therefore produced a different digest for the same protocol payload.
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _legacy_protocol_fingerprint(value: dict[str, Any]) -> str:
    """Fingerprint emitted before canonical compact JSON serialization."""

    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _recovery_formal_p3_fields(args: Any) -> dict[str, Any]:
    keys = {
        "insertion_block",
        "global_token_budget",
        "fine_token_budget",
        "max_regions",
        "minimum_region_tokens",
        "evidence_decoder",
        "trace_split_budget_mode",
        "trace_split_min_context_fraction",
        "trace_split_max_context_fraction",
        "trace_split_context_expansion_factor",
        "trace_split_context_decoder",
        "patch_exchange",
        "patch_exchange_global_scale",
        "patch_exchange_local_scale",
        "patch_exchange_total_cap_ratio",
        "patch_exchange_soft_floor_tokens",
        "patch_exchange_balanced_soft_floor",
        "patch_exchange_preserve_native_global",
        "patch_exchange_continuous_soft_floor",
        "patch_exchange_continuous_soft_floor_max_tokens",
        "patch_exchange_continuous_soft_floor_ramp_start_tokens",
        "patch_exchange_continuous_soft_floor_ramp_end_tokens",
        "patch_exchange_view_min_pixels",
        "answer_protocol",
        "fusion",
        "bridge_mode",
        "visual_output_mode",
    }
    return {key: value for key, value in vars(args).items() if key in keys}


def validate_stage_b_parent(
    checkpoint: Path,
    *,
    protocol: dict[str, Any],
    state: dict[str, Any],
    expected_rows: int,
    gradient_accumulation: int,
    expected_alignment: dict[str, Any],
    expected_manifest_sha256: str,
    expected_selector_sha256: str,
    expected_h_safe_sha256: str | None = None,
    expected_protocol_fields: dict[str, Any] | None = None,
) -> tuple[str, int]:
    """Validate one exact E-PLB 0.5-epoch parent for a matched Stage-B fork."""

    if state.get("format_version") != "evivit_visual_cot_recovery_sft_v1":
        raise ValueError("unsupported Stage-B parent checkpoint")
    if protocol.get("format_version") != "evivit_visual_cot_recovery_sft_v1":
        raise ValueError("unsupported Stage-B parent protocol")
    parent_payload = {
        key: value for key, value in protocol.items() if key != "protocol_fingerprint"
    }
    canonical_parent_fingerprint = _protocol_fingerprint(parent_payload)
    legacy_parent_fingerprint = _legacy_protocol_fingerprint(parent_payload)
    parent_fingerprint = protocol.get("protocol_fingerprint")
    if parent_fingerprint not in {
        canonical_parent_fingerprint,
        legacy_parent_fingerprint,
    }:
        raise ValueError("Stage-B parent protocol fingerprint is stale")
    if state.get("protocol_fingerprint") != parent_fingerprint:
        raise ValueError("Stage-B parent state/protocol fingerprint mismatch")
    if protocol.get("lane") != "E-PLB":
        raise ValueError("Stage-B parent must be the E-PLB lane")
    router = protocol.get("router")
    if not isinstance(router, dict) or router.get("stage") != "A":
        raise ValueError("Stage-B parent must be a Stage-A router checkpoint")
    if protocol.get("rows") != expected_rows:
        raise ValueError("Stage-B parent row count differs from the matched fork")
    if protocol.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("Stage-B parent manifest hash differs from the matched fork")
    if router.get("alignment") != expected_alignment:
        raise ValueError("Stage-B parent router alignment differs from the matched fork")
    if router.get("teacher_selector_sha256") != expected_selector_sha256:
        raise ValueError("Stage-B parent selector artifact differs from the matched fork")
    if expected_h_safe_sha256 is not None:
        h_safe = router.get("h_safe")
        if (
            not isinstance(h_safe, dict)
            or h_safe.get("mode") != "frozen"
            or h_safe.get("artifact_sha256") != expected_h_safe_sha256
        ):
            raise ValueError("Stage-B parent H-Safe artifact differs from the matched fork")
    for key, expected in (expected_protocol_fields or {}).items():
        if protocol.get(key) != expected:
            raise ValueError(f"Stage-B parent protocol field differs: {key}")
    boundary = half_epoch_micro_steps(expected_rows, gradient_accumulation)
    if int(state.get("micro_step", -1)) != boundary:
        raise ValueError("Stage-B parent is not the exact 0.5-epoch checkpoint")
    if int(state.get("optimizer_step", -1)) != boundary // gradient_accumulation:
        raise ValueError("Stage-B parent optimizer step is inconsistent")
    if state.get("selector_state_dict") is None:
        raise ValueError("Stage-B parent lacks its trainable PTEA student state")
    if state.get("h_safe_state_dict") is not None:
        raise ValueError("Stage-B parent unexpectedly contains trainable H-Safe state")
    for required in ("bridge_checkpoint.pt", "language_adapter"):
        if not (checkpoint / required).exists():
            raise FileNotFoundError(f"Stage-B parent lacks {required}")
    return parent_fingerprint, boundary


def save_recovery_checkpoint(
    destination: Path,
    *,
    bridge: torch.nn.Module,
    policy_model: Any | None,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    micro_step: int,
    optimizer_step: int,
    protocol_fingerprint: str,
    config: dict[str, Any],
    selector: torch.nn.Module | None = None,
    h_safe: torch.nn.Module | None = None,
    protocol: dict[str, Any] | None = None,
) -> None:
    """Atomically save a formal recovery checkpoint and exact resume state."""

    if protocol is not None:
        protocol_payload = {
            key: value
            for key, value in protocol.items()
            if key != "protocol_fingerprint"
        }
        if (
            protocol.get("protocol_fingerprint") != protocol_fingerprint
            or _protocol_fingerprint(protocol_payload) != protocol_fingerprint
        ):
            raise ValueError("recovery checkpoint protocol fingerprint is stale")
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    if policy_model is not None:
        policy_model.save_pretrained(temporary / "language_adapter")
    torch.save(
        {
            "version": "evivit_v3_sparse_mid_bridge_v1",
            "step": optimizer_step,
            "config": {
                "insertion_block": int(config["insertion_block"]),
                "bridge_dim": int(config["bridge_dim"]),
                "bridge_heads": int(config["bridge_heads"]),
                "neighborhood_radius": int(config["neighborhood_radius"]),
                "max_relative_residual": config["max_relative_residual"],
            },
            "bridge": bridge.state_dict(),
        },
        temporary / "bridge_checkpoint.pt",
    )
    state = {
        "format_version": "evivit_visual_cot_recovery_sft_v1",
        "protocol_fingerprint": protocol_fingerprint,
        "lane": protocol.get("lane") if protocol is not None else None,
        "recovery_stage": (
            protocol.get("recovery_stage") if protocol is not None else None
        ),
        "micro_step": micro_step,
        "optimizer_step": optimizer_step,
        "stage_optimizer_step": (
            optimizer_step if int(config.get('effective_updates',0))>0 else
            (micro_step - int(config.get("recovery_stage_start_micro_step", 0)))
            // int(config["gradient_accumulation"])
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state(0),
    }
    if selector is not None:
        state["selector_state_dict"] = selector.state_dict()
    if h_safe is not None:
        state["h_safe_state_dict"] = h_safe.state_dict()
    torch.save(state, temporary / "training_state.pt")
    (temporary / "training_state.json").write_text(
        json.dumps(
            {
                "format_version": state["format_version"],
                "protocol_fingerprint": protocol_fingerprint,
                "micro_step": micro_step,
                "optimizer_step": optimizer_step,
                "stage_optimizer_step": state["stage_optimizer_step"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if protocol is not None:
        (temporary / "run_protocol.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    backup = destination.parent / f".{destination.name}.old-{uuid.uuid4().hex}"
    moved_old = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved_old = True
        os.replace(temporary, destination)
    except BaseException:
        if moved_old and not destination.exists() and backup.exists():
            os.replace(backup, destination)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    if moved_old:
        shutil.rmtree(backup)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("datasets/derived/evivit_v2/train/ptea_a_1m_train_source_boxes.jsonl"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen3-VL-4B-Instruct"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--recovery-lane",
        choices=(
            "E-L",
            "E-B",
            "E-LB",
            "E-P",
            "E-PL",
            "E-PLB",
            "E-PLB-C",
            "E-PHLB",
        ),
        help=(
            "Enable the formal Visual-CoT recovery contract on the exact frozen "
            "P3/B16 EviViT path. E-L trains only language LoRA, E-B only the "
            "Sparse Bridge, and E-LB trains both. E-P* Stage-A lanes additionally "
            "train PTEA from aligned bbox/map supervision. Stage-B E-PLB-C/"
            "E-PHLB must fork from the same Stage-A E-PLB half-epoch checkpoint."
        ),
    )
    parser.add_argument("--recovery-stage", choices=("A", "B"), default="A")
    parser.add_argument(
        "--parent-checkpoint",
        type=Path,
        help="Stage-B-only E-PLB Stage-A half-epoch checkpoint; model states only.",
    )
    parser.add_argument(
        "--router-aux",
        type=Path,
        help="Exact router_aux.jsonl paired with the full recovery manifest.",
    )
    parser.add_argument("--expected-train-sha256")
    parser.add_argument("--expected-router-sha256")
    parser.add_argument("--expected-router-rows", type=int)
    parser.add_argument(
        "--h-safe-geometry-mask",
        type=Path,
        help=(
            "Hash-bound E-PHLB-only exclusion ledger. It may suppress H-Safe "
            "geometry loss but cannot alter the aligned PTEA router targets."
        ),
    )
    parser.add_argument("--expected-h-safe-geometry-mask-sha256")
    parser.add_argument(
        "--recovery-engineering-smoke",
        action="store_true",
        help="Allow at most 128 rows/one epoch while validating a recovery lane.",
    )
    parser.add_argument(
        "--recovery-stop-after-step",
        type=int,
        default=0,
        help="Smoke-only controlled interruption after atomically saving this micro-step.",
    )
    parser.add_argument(
        "--control-evidence-policy",
        choices=("uniform", "random", "feature_norm"),
        help=(
            "B1 matched visual-adapter control.  When set, the loaded PTEA "
            "checkpoint is used only to keep the code/feature contract "
            "matched; region allocation is generated without any human "
            "trace, last-zoom, final-box or attention-density supervision."
        ),
    )
    parser.add_argument(
        "--adaptive-box-head",
        type=Path,
        help=(
            "Frozen EviViT-v6 AdaptiveBox head. When supplied, Bridge "
            "training rereads the same H-Safe-adjusted regions used at "
            "inference instead of silently training on the anchor boxes."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Resume bridge, optimizer, RNG, and step from an existing checkpoint.",
    )
    parser.add_argument(
        "--initial-language-adapter",
        type=Path,
        help=(
            "Initialize a fresh language-LoRA run from this adapter without "
            "restoring optimizer or training-step state. Required by formal GRPO."
        ),
    )
    parser.add_argument(
        "--recovery-objective",
        choices=("sft", "grpo"),
        default="sft",
        help="Language-side objective for an E-L recovery lane.",
    )
    parser.add_argument("--grpo-rollouts-per-prompt", type=int, default=4)
    parser.add_argument("--grpo-num-epochs", type=int, default=3)
    parser.add_argument("--effective-updates", type=int, default=0)
    parser.add_argument("--save-effective-updates", type=int, default=125)
    parser.add_argument("--binary-reward-v2", action="store_true")
    parser.add_argument("--grpo-rollout-batch-size", type=int, default=1)
    parser.add_argument("--grpo-max-answer-tokens", type=int, default=64)
    parser.add_argument("--grpo-temperature", type=float, default=0.9)
    parser.add_argument("--grpo-top-p", type=float, default=0.95)
    parser.add_argument("--grpo-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--grpo-kl-coefficient", type=float, default=0.01)
    parser.add_argument(
        "--reward-judge-model",
        type=Path,
        help=(
            "Optional frozen Qwen3-VL semantic judge used only after exact "
            "and normalized matching fail."
        ),
    )
    parser.add_argument("--insertion-block", type=int, default=16)
    parser.add_argument("--global-token-budget", type=int, default=1024)
    parser.add_argument("--fine-token-budget", type=int, default=3072)
    parser.add_argument(
        "--native-token-caps",
        action="store_true",
        help=(
            "Train Simple EviViT under the same Native-Cap distribution used "
            "at inference: Global/Fine budgets are ceilings, not targets."
        ),
    )
    parser.add_argument(
        "--native-cap-minimum-fine-tokens",
        type=int,
        default=64,
        help=(
            "Minimum readable token count for each selected Fine crop under "
            "Native-Cap. Global keeps Qwen's natural processor minimum."
        ),
    )
    parser.add_argument(
        "--native-cap-minimum-global-tokens",
        type=int,
        default=0,
        help=(
            "S3 topology floor for the Global Native-Cap path. Zero keeps "
            "the original S1 natural-grid behavior."
        ),
    )
    parser.add_argument(
        "--native-cap-anchor-fine-tokens",
        type=int,
        default=0,
        help=(
            "S3 readability floor for only the first/highest-evidence region."
        ),
    )
    parser.add_argument(
        "--native-cap-fine-floor-native-ratio",
        type=float,
        default=0.0,
        help=(
            "Use one source-native-token-relative total Fine floor and split "
            "it over the allocated evidence regions."
        ),
    )
    parser.add_argument(
        "--native-cap-minimum-fine-total",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--native-global-total-token-budget",
        type=int,
        default=0,
        help=(
            "Use the image's native Qwen global tokenization when it fits this "
            "many merged tokens; otherwise use --global-token-budget."
        ),
    )
    parser.add_argument(
        "--visual-output-mode",
        choices=("append", "global_only"),
        default="append",
    )
    parser.add_argument(
        "--native-global-mrope",
        action="store_true",
        help=(
            "For a Global-only parent-preserving route, keep Qwen's exact "
            "native row/column MRoPE grid instead of mixed-resolution "
            "coordinate quantization."
        ),
    )
    parser.add_argument(
        "--bridge-mode",
        choices=("bidirectional", "preserve_global", "preserve_fine", "identity"),
        default="bidirectional",
    )
    parser.add_argument(
        "--answer-protocol",
        choices=("compact_json", "direct_concise"),
        default="compact_json",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Use the Qwen3.5 non-thinking chat-template contract during "
            "answer-only Bridge training."
        ),
    )
    parser.add_argument("--max-regions", type=int, default=3)
    parser.add_argument("--minimum-region-tokens", type=int, default=64)
    parser.add_argument(
        "--evitree-v2",
        action="store_true",
        help=(
            "Train the parent-preserving fusion on learned hierarchical child "
            "regions instead of fixed Residual-Focus rectangles."
        ),
    )
    parser.add_argument("--evitree-need-head", type=Path)
    parser.add_argument("--evitree-split-policy", type=Path)
    parser.add_argument("--evitree-tree-max-depth", type=int, default=5)
    parser.add_argument("--evitree-tree-max-leaves", type=int, default=10)
    parser.add_argument("--evitree-split-threshold", type=float, default=-1.0)
    parser.add_argument("--evitree-tree-reliability", type=float, default=0.80)
    parser.add_argument("--evitree-tree-target-mass", type=float, default=0.90)
    parser.add_argument(
        "--evitree-sparse-branches",
        action="store_true",
        help=(
            "Keep the complete Global stream but reread only the strongest "
            "adaptive tree branches as Fine children."
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
        "--native-ratio-budget",
        action="store_true",
        help=(
            "Allocate Global and Fine as fixed continuous fractions of each "
            "image's native Qwen visual-token demand."
        ),
    )
    parser.add_argument("--native-fine-ratio", type=float, default=0.50)
    parser.add_argument(
        "--native-need-ratio-budget",
        action="store_true",
        help=(
            "Scale Global/Fine budgets from Qwen's native token demand and "
            "let the human-supervised Need head interpolate the Fine ratio."
        ),
    )
    parser.add_argument("--native-global-ratio", type=float, default=0.40)
    parser.add_argument("--native-min-fine-ratio", type=float, default=0.20)
    parser.add_argument("--native-max-fine-ratio", type=float, default=0.60)
    parser.add_argument("--native-ratio-min-global-tokens", type=int, default=64)
    parser.add_argument("--native-ratio-min-fine-tokens", type=int, default=64)
    parser.add_argument("--native-ratio-max-global-tokens", type=int, default=0)
    parser.add_argument("--native-ratio-max-fine-tokens", type=int, default=0)
    parser.add_argument(
        "--native-ratio-soft-floor-tokens",
        type=int,
        default=0,
        help=(
            "Continuous SoftFloor used by Native-Need budgeting.  This is "
            "separate from the PatchExchange realization floor because the "
            "former controls tree-policy budgets while the latter controls "
            "the final image grids."
        ),
    )
    parser.add_argument("--native-budget-alignment", type=int, default=64)
    parser.add_argument("--processor-min-pixels", type=int, default=4096)
    parser.add_argument(
        "--processor-max-pixels", type=int, default=16 * 1024 * 1024
    )
    parser.add_argument("--native-pixel-contract", action="store_true")
    parser.add_argument(
        "--patch-exchange",
        action="store_true",
        help=(
            "Train the sparse Bridge under EviViT-v5 per-image native-budget "
            "Global/Fine PatchExchange semantics."
        ),
    )
    parser.add_argument(
        "--patch-exchange-tree",
        action="store_true",
        help=(
            "Explicitly allow PatchExchange to realize Global/Fine grids "
            "after EviTree-v2 selects hierarchical branches. This is the "
            "training counterpart of the v8 inference contract."
        ),
    )
    parser.add_argument("--patch-exchange-global-scale", type=float, default=1.5)
    parser.add_argument("--patch-exchange-local-scale", type=float, default=2.0)
    parser.add_argument(
        "--patch-exchange-total-cap-ratio", type=float, default=1.0
    )
    parser.add_argument(
        "--patch-exchange-soft-floor-tokens", type=int, default=0
    )
    parser.add_argument(
        "--patch-exchange-balanced-soft-floor", action="store_true"
    )
    parser.add_argument(
        "--patch-exchange-preserve-native-global",
        action="store_true",
        help=(
            "Keep at least the source-native Global grid under PatchExchange. "
            "This is the training counterpart of EviViT-v7 NativeFloor."
        ),
    )
    parser.add_argument(
        "--patch-exchange-continuous-soft-floor",
        action="store_true",
        help=(
            "Interpolate the PatchExchange soft floor per image from native "
            "Qwen visual-token demand."
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
        "--patch-exchange-view-min-pixels", type=int, default=4096
    )
    parser.add_argument(
        "--global-processor-max-pixels",
        type=int,
        default=4 * 1024 * 1024,
    )
    parser.add_argument(
        "--fine-processor-max-pixels",
        type=int,
        default=16 * 1024 * 1024,
    )
    parser.add_argument("--continuous-native-global", action="store_true")
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
        "--continuous-global-exponent", type=float, default=0.5
    )
    parser.add_argument(
        "--seeded-basin-config",
        type=Path,
        help="Trace1144-frozen Stage-D basin decoder configuration.",
    )
    parser.add_argument(
        "--evidence-decoder",
        choices=(
            "top_boxes",
            "residual_focus",
            "adaptive_density_mass",
            "adaptive_density_mass_topology",
            "adaptive_contour",
            "seeded_basin_confidence_fallback",
            "anchored_growth_b025",
            "anchored_growth_b035",
            "evicontour_utility",
            "evicontour_csr",
            "trace_split",
        ),
        default="top_boxes",
    )
    parser.add_argument("--contour-relative-threshold", type=float, default=0.40)
    parser.add_argument("--contour-context-margin", type=float, default=0.20)
    parser.add_argument(
        "--contour-minimum-residual-mass", type=float, default=0.05
    )
    parser.add_argument("--evicontour-utility-checkpoint", type=Path)
    parser.add_argument("--trace-split-context-fraction", type=float, default=0.25)
    parser.add_argument(
        "--trace-split-context-expansion-factor", type=float, default=1.0
    )
    parser.add_argument(
        "--trace-split-context-decoder",
        choices=("top_boxes", "adaptive_density_mass"),
        default="top_boxes",
    )
    parser.add_argument(
        "--trace-split-budget-mode",
        choices=("fixed", "competitive", "learned"),
        default="fixed",
    )
    parser.add_argument("--trace-split-context-need-checkpoint", type=Path)
    parser.add_argument("--trace-split-min-context-fraction", type=float, default=0.10)
    parser.add_argument("--trace-split-max-context-fraction", type=float, default=0.35)
    parser.add_argument("--trace-split-adaptive-region-minimum", action="store_true")
    parser.add_argument(
        "--evicontour-csr-sufficiency-threshold", type=float, default=0.65
    )
    parser.add_argument(
        "--evicontour-csr-sufficiency-tolerance", type=float, default=0.05
    )
    parser.add_argument(
        "--evicontour-csr-core-relevance-tolerance", type=float, default=0.05
    )
    parser.add_argument(
        "--evicontour-csr-context-ring-mass-ratio", type=float, default=0.20
    )
    parser.add_argument(
        "--evicontour-csr-maximum-scope-hops", type=int, default=3
    )
    parser.add_argument(
        "--evicontour-csr-minimum-parent-sufficiency-gain",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--evicontour-csr-fine-floor-native-ratio", type=float, default=0.10
    )
    parser.add_argument(
        "--evicontour-csr-minimum-fine-total", type=int, default=256
    )
    parser.add_argument(
        "--evicontour-csr-maximum-fine-total", type=int, default=3072
    )
    parser.add_argument("--bridge-dim", type=int, default=256)
    parser.add_argument("--bridge-heads", type=int, default=4)
    parser.add_argument("--neighborhood-radius", type=int, default=1)
    parser.add_argument(
        "--fusion",
        choices=(
            "sparse_bridge",
            "tracefovea",
            "tracefovea_human_gate",
            "eviweave_human_gate",
            "eviweave_normalized",
            "evirelay",
            "eviblend_safe_joint",
            "evislot",
            "trace_calibrated_bridge",
        ),
        default="sparse_bridge",
        help=(
            "Keep every other runtime variable fixed and replace only the "
            "Block16 global/local fusion mechanism."
        ),
    )
    parser.add_argument("--fovea-dim", type=int, default=256)
    parser.add_argument("--evislot-dim", type=int, default=256)
    parser.add_argument("--evislot-heads", type=int, default=4)
    parser.add_argument("--evislot-slots-per-region", type=int, default=4)
    parser.add_argument(
        "--evislot-maximum-residual-scale", type=float, default=0.25
    )
    parser.add_argument("--evislot-evidence-bias", type=float, default=0.5)
    parser.add_argument(
        "--evislot-spatial-anchor-bias",
        action="store_true",
        help=(
            "Enable a fixed non-parametric 2x2 quadrant bias on the four "
            "EviSlot attention logits. Disabled by default for checkpoint "
            "and protocol compatibility."
        ),
    )
    parser.add_argument(
        "--evislot-spatial-anchor-strength", type=float, default=2.0
    )
    parser.add_argument(
        "--evislot-spatial-anchor-sigma", type=float, default=0.30
    )
    parser.add_argument(
        "--evislot-diversity-weight", type=float, default=0.05
    )
    parser.add_argument(
        "--evislot-parent-consistency-weight", type=float, default=0.01
    )
    parser.add_argument(
        "--trace-label-root",
        type=Path,
        default=Path(
            "datasets/derived/evivit_v16/trace_labels_multiscale_v1"
        ),
    )
    parser.add_argument("--trace-supervision-weight", type=float, default=0.10)
    parser.add_argument(
        "--trace-residual-modulation",
        action="store_true",
        help=(
            "Use the human-trace child logits to redistribute, but not enlarge, "
            "the fine-token residual budget. Reserved for single-stage "
            "tracefovea_human_gate."
        ),
    )
    parser.add_argument("--minimum-child-importance", type=float, default=0.25)
    parser.add_argument("--maximum-child-importance", type=float, default=3.0)
    parser.add_argument(
        "--progressive-bridge-blocks",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Additional upper-ViT blocks for shared evidence fusion. "
            "EviWeave defaults to blocks 20 and 24."
        ),
    )
    parser.add_argument("--relay-dim", type=int, default=256)
    parser.add_argument("--relay-tokens", type=int, default=8)
    parser.add_argument(
        "--initial-bridge-checkpoint",
        type=Path,
        help=(
            "Required by eviblend_safe_joint: a frozen v4 sparse-bridge "
            "checkpoint defining the trusted native path."
        ),
    )
    parser.add_argument(
        "--warm-start-bridge-checkpoint",
        type=Path,
        help=(
            "Initialize a trainable sparse bridge from an existing v4 sparse-"
            "bridge checkpoint while resetting optimizer and training step. "
            "This is distribution adaptation, not interrupted-run resume."
        ),
    )
    parser.add_argument("--safe-joint-block", type=int, default=20)
    parser.add_argument("--safe-joint-blend-dim", type=int, default=256)
    parser.add_argument("--safe-joint-maximum-gate", type=float, default=0.50)
    parser.add_argument("--safe-joint-minimum-importance", type=float, default=0.25)
    parser.add_argument("--safe-joint-maximum-importance", type=float, default=3.0)
    parser.add_argument("--trace-calibration-dim", type=int, default=256)
    parser.add_argument(
        "--trace-calibration-maximum", type=float, default=0.50
    )
    parser.add_argument(
        "--trace-calibration-minimum-importance", type=float, default=0.25
    )
    parser.add_argument(
        "--trace-calibration-maximum-importance", type=float, default=3.0
    )
    parser.add_argument(
        "--trace-calibration-amplitude-mode",
        choices=("signed_zero", "positive_centered"),
        default="signed_zero",
        help=(
            "signed_zero reproduces the original double-zero calibrator; "
            "positive_centered starts with a positive bounded amplitude while "
            "remaining exactly v4 because the correction uses importance-1."
        ),
    )
    parser.add_argument(
        "--trace-calibration-preserve-fp32-importance",
        action="store_true",
        help=(
            "Keep normalized token importance in FP32 so early non-uniform "
            "gains are not rounded back to one by BF16."
        ),
    )
    parser.add_argument(
        "--skip-empty-context-trace",
        action="store_true",
        help=(
            "When the human context channel has zero mass, keep QA and any "
            "available fine read/core supervision but do not fabricate a "
            "uniform global-token target. Disabled by default for exact "
            "compatibility with already-running historical lanes."
        ),
    )
    parser.add_argument("--max-relative-residual", type=float, default=0.2)
    parser.add_argument("--coordinate-bins", type=int, default=128)
    parser.add_argument("--scale-bins", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--projection-seed", type=int, default=20260715)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "Recompute frozen upper-transformer activations during backward "
            "to reduce memory without changing tokens, loss, or parameters."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--language-learning-rate", type=float, default=1e-6)
    parser.add_argument("--bridge-learning-rate", type=float, default=5e-6)
    parser.add_argument("--ptea-learning-rate", type=float, default=1e-5)
    parser.add_argument("--h-safe-learning-rate", type=float, default=5e-6)
    parser.add_argument("--ptea-cosine-weight", type=float, default=0.30)
    parser.add_argument(
        "--ptea-decisive-distillation-weight", type=float, default=0.10
    )
    parser.add_argument(
        "--ptea-context-distillation-weight", type=float, default=0.25
    )
    parser.add_argument("--h-safe-giou-weight", type=float, default=2.0)
    parser.add_argument("--h-safe-coverage-weight", type=float, default=0.5)
    parser.add_argument("--h-safe-excess-area-weight", type=float, default=0.05)
    parser.add_argument("--h-safe-identity-weight", type=float, default=0.10)
    parser.add_argument("--h-safe-drift-weight", type=float, default=0.10)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--identity-weight", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--memory-budget-mib", type=int, default=45000)
    args = parser.parse_args()
    dynamic_grpo = args.effective_updates > 0
    if dynamic_grpo and (args.recovery_objective!='grpo' or not args.binary_reward_v2 or args.save_effective_updates<1):
        raise ValueError('effective-group mode requires binary-v2 GRPO and a positive save interval')

    if args.limit <= 0 or args.max_steps <= 0:
        raise ValueError("limit and max_steps must be positive")
    recovery = recovery_lane(args.recovery_lane) if args.recovery_lane else None
    if args.recovery_objective == "grpo":
        if recovery is None or recovery.name != "E-L":
            raise ValueError("formal P3 GRPO is restricted to the E-L lane")
        if args.initial_language_adapter is None:
            raise ValueError("formal P3 GRPO requires --initial-language-adapter")
        if args.resume_checkpoint is not None or args.parent_checkpoint is not None:
            raise ValueError("fresh formal P3 GRPO cannot reuse SFT optimizer state")
        if args.grpo_rollouts_per_prompt < 2:
            raise ValueError("GRPO requires at least two rollouts per prompt")
        if not 0 < args.grpo_top_p <= 1:
            raise ValueError("GRPO top-p must be in (0, 1]")
        if not 0 < args.grpo_clip_epsilon < 1 or args.grpo_kl_coefficient < 0:
            raise ValueError("invalid GRPO clipping or KL setting")
    elif args.initial_language_adapter is not None:
        raise ValueError("--initial-language-adapter is reserved for fresh GRPO")
    if recovery is None and args.recovery_stop_after_step:
        raise ValueError("--recovery-stop-after-step requires --recovery-lane")
    if recovery is not None:
        assert_formal_p3_contract(args)
        if args.recovery_stage == "A":
            if recovery.name not in {"E-L", "E-B", "E-LB", "E-P", "E-PL", "E-PLB"}:
                raise ValueError("Stage A does not train E-PLB-C or E-PHLB")
            if args.parent_checkpoint is not None:
                raise ValueError("Stage A cannot have --parent-checkpoint")
        else:
            if recovery.name not in {"E-PLB-C", "E-PHLB"}:
                raise ValueError("Stage B is restricted to E-PLB-C and E-PHLB")
            if args.parent_checkpoint is None:
                raise ValueError("Stage B requires the common E-PLB parent checkpoint")
        if recovery.requires_router_aux:
            missing_router = [
                name
                for name, value in (
                    ("--router-aux", args.router_aux),
                    ("--expected-train-sha256", args.expected_train_sha256),
                    ("--expected-router-sha256", args.expected_router_sha256),
                    ("--expected-router-rows", args.expected_router_rows),
                )
                if value is None
            ]
            if missing_router:
                raise ValueError(
                    "router-supervised recovery requires exact frozen inputs: "
                    f"{missing_router}"
                )
            if recovery.train_h_safe:
                missing_h_safe_mask = [
                    name
                    for name, value in (
                        ("--h-safe-geometry-mask", args.h_safe_geometry_mask),
                        (
                            "--expected-h-safe-geometry-mask-sha256",
                            args.expected_h_safe_geometry_mask_sha256,
                        ),
                    )
                    if value is None
                ]
                if missing_h_safe_mask:
                    raise ValueError(
                        "trainable H-Safe requires an exact geometry-label mask: "
                        f"{missing_h_safe_mask}"
                    )
            elif (
                args.h_safe_geometry_mask is not None
                or args.expected_h_safe_geometry_mask_sha256 is not None
            ):
                raise ValueError("only E-PHLB may consume an H-Safe geometry mask")
            if args.control_evidence_policy is not None:
                raise ValueError("router-supervised recovery cannot use a control policy")
            if args.weight_decay != 0.0:
                raise ValueError("router-supervised recovery requires weight decay 0")
            if (
                args.ptea_learning_rate != 1e-5
                or args.ptea_cosine_weight != 0.30
                or args.ptea_decisive_distillation_weight != 0.10
                or args.ptea_context_distillation_weight != 0.25
            ):
                raise ValueError("router-supervised recovery loss/LR contract changed")
            if (
                args.h_safe_learning_rate != 5e-6
                or args.h_safe_giou_weight != 2.0
                or args.h_safe_coverage_weight != 0.5
                or args.h_safe_excess_area_weight != 0.05
                or args.h_safe_identity_weight != 0.10
                or args.h_safe_drift_weight != 0.10
            ):
                raise ValueError("H-Safe recovery loss/LR contract changed")
        if args.recovery_stop_after_step < 0 or args.recovery_stop_after_step > args.max_steps:
            raise ValueError("invalid recovery stop step")
        if args.recovery_stop_after_step and not args.recovery_engineering_smoke:
            raise ValueError("controlled recovery stop is engineering-smoke only")
        if args.recovery_engineering_smoke:
            if args.limit > 128 or args.max_steps > args.limit:
                raise ValueError(
                    "recovery engineering smoke is limited to at most 128 rows/one epoch"
                )
        else:
            expected_accumulation = (
                4 if args.recovery_objective == "grpo" else 20
            )
            if args.gradient_accumulation != expected_accumulation:
                raise ValueError(
                    f"formal recovery {args.recovery_objective.upper()} requires "
                    f"gradient accumulation {expected_accumulation}"
                )
            expected_epochs = args.grpo_num_epochs if args.recovery_objective == "grpo" else 3
            if expected_epochs < 1 or args.max_steps != expected_epochs * args.limit:
                raise ValueError(f"formal recovery requires exactly {expected_epochs} epochs")
            expected_save = half_epoch_micro_steps(
                args.limit, args.gradient_accumulation
            )
            if args.save_steps != expected_save:
                raise ValueError(
                    "formal recovery checkpoints must be saved every half epoch: "
                    f"expected --save-steps {expected_save}"
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
        args.global_processor_max_pixels != args.processor_max_pixels
    ):
        raise ValueError(
            "native Global identity requires equal processor/global pixel "
            "maxima so the untouched source follows the B16 processor"
        )
    if args.fusion == "evislot":
        if (
            args.evislot_dim <= 0
            or args.evislot_heads <= 0
            or args.evislot_dim % args.evislot_heads
            or args.evislot_slots_per_region <= 0
        ):
            raise ValueError("invalid EviSlot dimension/head/slot configuration")
        if args.evislot_maximum_residual_scale < 0:
            raise ValueError("EviSlot residual scale must be non-negative")
        if args.evislot_spatial_anchor_strength < 0:
            raise ValueError("EviSlot spatial-anchor strength must be non-negative")
        if args.evislot_spatial_anchor_sigma <= 0:
            raise ValueError("EviSlot spatial-anchor sigma must be positive")
        if (
            args.evislot_spatial_anchor_bias
            and args.evislot_slots_per_region != 4
        ):
            raise ValueError("EviSlot 2x2 anchors require four slots per region")
        if min(
            args.evislot_diversity_weight,
            args.evislot_parent_consistency_weight,
            args.trace_supervision_weight,
            args.identity_weight,
        ) < 0:
            raise ValueError("EviSlot auxiliary loss weights must be non-negative")
        if args.visual_output_mode != "append":
            raise ValueError("EviSlot training must emit slots with append mode")
        if args.bridge_mode not in {"bidirectional", "preserve_global"}:
            raise ValueError("EviSlot training requires its Global-preserving slot path")
    if args.trace_residual_modulation and args.fusion != "tracefovea_human_gate":
        raise ValueError(
            "--trace-residual-modulation is reserved for "
            "tracefovea_human_gate"
        )
    requires_frozen_v4 = args.fusion in {
        "eviblend_safe_joint",
        "trace_calibrated_bridge",
    }
    if requires_frozen_v4 != bool(args.initial_bridge_checkpoint is not None):
        raise ValueError(
            "eviblend_safe_joint and trace_calibrated_bridge require "
            "--initial-bridge-checkpoint; that option is reserved for them"
        )
    if args.native_global_total_token_budget < 0:
        raise ValueError("native global token budget must be non-negative")
    if args.native_global_total_token_budget and (
        args.native_ratio_budget or args.native_need_ratio_budget
    ):
        raise ValueError(
            "native-global bypass and native-ratio budgets are mutually exclusive"
        )
    if args.patch_exchange:
        if args.patch_exchange_tree:
            if (
                not args.evitree_v2
                or not args.native_need_ratio_budget
                or args.evidence_decoder != "residual_focus"
                or not args.evitree_sparse_branches
                or args.max_regions != 3
            ):
                raise ValueError(
                    "Tree-PatchExchange training requires EviTree-v2, "
                    "Native-Need budgeting, Residual-Focus, and three sparse branches"
                )
            if (
            args.native_token_caps
                or args.native_pixel_contract
                or args.native_global_total_token_budget
                or args.native_ratio_budget
            ):
                raise ValueError(
                    "Tree-PatchExchange cannot mix with another pixel/token contract"
                )
        else:
            if (
                args.native_token_caps
                or args.native_pixel_contract
                or args.native_global_total_token_budget
                or args.native_ratio_budget
                or args.native_need_ratio_budget
            ):
                raise ValueError(
                    "PatchExchange training cannot mix with other dynamic budget "
                    "contracts"
                )
            if (
                args.control_evidence_policy is None
                and args.evidence_decoder != "trace_split"
            ) or args.max_regions != 3:
                raise ValueError(
                    "PatchExchange training requires trace_split (or an explicit "
                    "control evidence policy) with three regions"
                )
        if (
            args.patch_exchange_global_scale < 1.0
            or args.patch_exchange_local_scale < 1.0
            or args.patch_exchange_total_cap_ratio <= 0
            or args.patch_exchange_soft_floor_tokens < 0
            or not 0
            < args.patch_exchange_view_min_pixels
            <= args.processor_max_pixels
        ):
            raise ValueError("invalid PatchExchange training contract")
        if (
            args.patch_exchange_balanced_soft_floor
            and args.patch_exchange_soft_floor_tokens <= 0
        ):
            raise ValueError("balanced PatchExchange requires a positive soft floor")
        if args.patch_exchange_preserve_native_global and not (
            args.patch_exchange_balanced_soft_floor
        ):
            raise ValueError(
                "preserving native Global requires balanced PatchExchange"
            )
        if args.patch_exchange_continuous_soft_floor and (
            not args.patch_exchange_balanced_soft_floor
            or args.patch_exchange_continuous_soft_floor_max_tokens
            < args.patch_exchange_soft_floor_tokens
            or args.patch_exchange_continuous_soft_floor_ramp_start_tokens < 0
            or args.patch_exchange_continuous_soft_floor_ramp_end_tokens
            <= args.patch_exchange_continuous_soft_floor_ramp_start_tokens
        ):
            raise ValueError("invalid continuous PatchExchange soft-floor contract")
    elif args.patch_exchange_tree:
        raise ValueError("--patch-exchange-tree requires --patch-exchange")
    if args.native_pixel_contract:
        if (
            args.native_token_caps
            or args.native_global_total_token_budget
            or args.native_ratio_budget
            or args.native_need_ratio_budget
        ):
            raise ValueError(
                "Native-Pixel training cannot mix token-cap, native bypass, "
                "or Need-ratio budget semantics"
            )
        if (
            args.processor_min_pixels <= 0
            or args.global_processor_max_pixels < args.processor_min_pixels
            or args.fine_processor_max_pixels < args.processor_min_pixels
        ):
            raise ValueError("invalid Native-Pixel Global/Fine pixel interval")
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
            raise ValueError("invalid continuous Global pixel interval")
        if not 0.0 <= args.continuous_global_exponent <= 1.0:
            raise ValueError("continuous Global exponent must lie in [0, 1]")
    if args.evidence_decoder.startswith("seeded_") != bool(
        args.seeded_basin_config is not None
    ):
        raise ValueError(
            "seeded Stage-D decoder and --seeded-basin-config must be used together"
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
    if args.native_token_caps:
        if args.native_global_total_token_budget:
            raise ValueError(
                "Native-Cap cannot be combined with native-global bypass"
            )
        if args.native_need_ratio_budget:
            raise ValueError(
                "Native-Cap and native Need-ratio budgeting are separate protocols"
            )
        if args.native_ratio_budget:
            raise ValueError(
                "Native-Cap and fixed native-ratio budgeting are separate protocols"
            )
        if not 0 < args.native_cap_minimum_fine_tokens <= args.fine_token_budget:
            raise ValueError(
                "Native-Cap Fine minimum must lie in (0, fine-token-budget]"
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
    elif args.native_cap_fine_floor_native_ratio:
        raise ValueError("continuous Fine floor requires --native-token-caps")
    if args.native_ratio_budget and args.native_need_ratio_budget:
        raise ValueError(
            "fixed native-ratio and learned native Need-ratio budgets are mutually exclusive"
        )
    if args.native_ratio_budget:
        if not (
            0 < args.native_global_ratio <= 1
            and 0 < args.native_fine_ratio <= 1
            and args.native_ratio_min_global_tokens > 0
            and args.native_ratio_min_fine_tokens > 0
            and args.native_budget_alignment > 0
        ):
            raise ValueError("invalid fixed native-ratio budget")
    if args.native_need_ratio_budget:
        if (
            args.native_ratio_min_global_tokens <= 0
            or args.native_ratio_min_fine_tokens <= 0
            or args.native_ratio_max_global_tokens < 0
            or args.native_ratio_max_fine_tokens < 0
            or args.native_ratio_soft_floor_tokens < 0
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
            raise ValueError("invalid native Need-ratio token bounds")
    if args.evidence_decoder == "trace_split":
        if args.max_regions != 3:
            raise ValueError("trace_split requires max_regions=3")
        if not (
            0.1
            <= args.trace_split_min_context_fraction
            <= args.trace_split_max_context_fraction
            <= 0.5
        ):
            raise ValueError("invalid trace_split context interval")
        if args.trace_split_context_expansion_factor < 1.0:
            raise ValueError("trace_split context expansion must be at least 1")
        if (
            args.trace_split_budget_mode == "learned"
            and args.trace_split_context_need_checkpoint is None
        ):
            raise ValueError(
                "learned trace_split requires --trace-split-context-need-checkpoint"
            )
    elif args.trace_split_context_need_checkpoint is not None:
        raise ValueError(
            "--trace-split-context-need-checkpoint requires trace_split decoder"
        )
    if args.warm_start_bridge_checkpoint is not None:
        if args.fusion not in {
            "sparse_bridge",
            "tracefovea",
            "tracefovea_human_gate",
        }:
            raise ValueError(
                "warm-start bridge adaptation requires a compatible sparse or "
                "parent-preserving fusion"
            )
        if args.resume_checkpoint is not None and recovery is None:
            raise ValueError(
                "warm-start and resume are mutually exclusive because warm-start "
                "resets optimizer and training step"
            )
    if args.max_steps % args.gradient_accumulation:
        raise ValueError("max_steps must end on an optimizer boundary")
    if args.save_steps % args.gradient_accumulation:
        raise ValueError("save_steps must end on an optimizer boundary")
    if args.evitree_v2:
        if args.fusion not in {"tracefovea", "tracefovea_human_gate"}:
            raise ValueError("EviTree-v2 requires a parent-preserving TraceFovea fusion")
        if args.evitree_split_policy is None:
            raise ValueError("EviTree-v2 requires --evitree-split-policy")
        if args.native_need_ratio_budget and args.evitree_need_head is None:
            raise ValueError(
                "native Need-ratio budgeting requires --evitree-need-head"
            )
    if args.memory_budget_mib > 0:
        decision = check_gpu(
            args.gpu,
            args.memory_budget_mib,
            ROOT / "configs/gpu_safety.json",
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
                        "requested_mib": 0,
                        "policy": "unbounded_by_process; physical CUDA OOM remains active",
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = "cuda:0"
    total_mib = torch.cuda.get_device_properties(0).total_memory / 1024**2
    if args.memory_budget_mib > 0:
        torch.cuda.set_per_process_memory_fraction(
            min(0.95, args.memory_budget_mib / total_mib), device=0
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats(0)

    manifest = args.train_manifest
    if not manifest.is_absolute():
        manifest = args.project_root / manifest
    selector_checkpoint = args.selector_checkpoint
    if not selector_checkpoint.is_absolute():
        selector_checkpoint = args.project_root / selector_checkpoint
    seeded_basin_config = None
    seeded_basin_config_path = args.seeded_basin_config
    if seeded_basin_config_path is not None:
        if not seeded_basin_config_path.is_absolute():
            seeded_basin_config_path = (
                args.project_root / seeded_basin_config_path
            )
        payload = json.loads(seeded_basin_config_path.read_text())
        seeded_basin_config = payload.get("config", payload)
        if not isinstance(seeded_basin_config, dict):
            raise ValueError("seeded basin config must contain a JSON object")
    router_alignment = None
    router_rows_by_id: dict[str, Any] = {}
    h_safe_geometry_mask = None
    h_safe_geometry_mask_ids: frozenset[str] = frozenset()
    if recovery is not None and recovery.requires_router_aux:
        assert args.router_aux is not None
        router_aux = args.router_aux
        if not router_aux.is_absolute():
            router_aux = args.project_root / router_aux
        router_alignment = load_aligned_router_rows(
            manifest,
            router_aux,
            expected_train_sha256=args.expected_train_sha256,
            expected_router_sha256=args.expected_router_sha256,
            expected_rows=args.expected_router_rows,
        )
        if args.limit > len(router_alignment.rows):
            raise ValueError("--limit exceeds the exactly aligned router release")
        selected_router_rows = router_alignment.rows[: args.limit]
        rows = [item.qa for item in selected_router_rows]
        router_rows_by_id = {item.id: item for item in selected_router_rows}
        if recovery.train_h_safe:
            assert args.h_safe_geometry_mask is not None
            assert args.expected_h_safe_geometry_mask_sha256 is not None
            assert args.expected_router_sha256 is not None
            h_safe_geometry_mask_path = args.h_safe_geometry_mask
            if not h_safe_geometry_mask_path.is_absolute():
                h_safe_geometry_mask_path = (
                    args.project_root / h_safe_geometry_mask_path
                )
            h_safe_geometry_mask = load_h_safe_geometry_mask(
                h_safe_geometry_mask_path,
                expected_sha256=args.expected_h_safe_geometry_mask_sha256,
                expected_train_manifest_sha256=args.expected_train_sha256,
                expected_router_aux_sha256=args.expected_router_sha256,
            )
            alignment_ids = {item.id for item in router_alignment.rows}
            if not h_safe_geometry_mask.ids.issubset(alignment_ids):
                raise ValueError("H-Safe geometry mask contains IDs outside router aux")
            h_safe_geometry_mask_ids = h_safe_geometry_mask.ids
    else:
        rows = read_jsonl(manifest)[: args.limit]
    if not rows:
        raise RuntimeError("training manifest is empty")
    parent_checkpoint = args.parent_checkpoint
    if parent_checkpoint is not None and not parent_checkpoint.is_absolute():
        parent_checkpoint = args.project_root / parent_checkpoint
    parent_state = None
    parent_protocol = None
    parent_protocol_fingerprint = None
    recovery_stage_start_micro_step = 0
    if recovery is not None and args.recovery_stage == "B":
        assert parent_checkpoint is not None
        if not parent_checkpoint.is_dir():
            raise FileNotFoundError(parent_checkpoint)
        parent_state_path = parent_checkpoint / "training_state.pt"
        parent_protocol_path = parent_checkpoint / "run_protocol.json"
        if not parent_protocol_path.is_file():
            parent_protocol_path = parent_checkpoint.parent / "run_protocol.json"
        if not parent_state_path.is_file():
            raise FileNotFoundError(parent_state_path)
        if not parent_protocol_path.is_file():
            raise FileNotFoundError(
                f"Stage-B parent lacks run_protocol.json: {parent_checkpoint}"
            )
        parent_state = torch.load(
            parent_state_path, map_location="cpu", weights_only=False
        )
        parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
        if not isinstance(parent_protocol, dict):
            raise ValueError("Stage-B parent protocol must be a JSON object")
        assert router_alignment is not None
        parent_h_safe_path = args.adaptive_box_head
        assert parent_h_safe_path is not None
        if not parent_h_safe_path.is_absolute():
            parent_h_safe_path = args.project_root / parent_h_safe_path
        parent_protocol_fingerprint, recovery_stage_start_micro_step = (
            validate_stage_b_parent(
                parent_checkpoint,
                protocol=parent_protocol,
                state=parent_state,
                expected_rows=len(rows),
                gradient_accumulation=args.gradient_accumulation,
                expected_alignment=router_alignment.protocol_fields(),
                expected_manifest_sha256=_sha256_file(manifest),
                expected_selector_sha256=_sha256_file(selector_checkpoint),
                expected_h_safe_sha256=_sha256_file(parent_h_safe_path),
                expected_protocol_fields={
                    "model": str(args.model),
                    "max_steps": args.max_steps,
                    "gradient_accumulation": args.gradient_accumulation,
                    "seed": args.seed,
                    "formal_p3": _recovery_formal_p3_fields(args),
                    "lora": {
                        "r": args.lora_r,
                        "alpha": args.lora_alpha,
                        "dropout": args.lora_dropout,
                        "target_pattern": LANGUAGE_LORA_TARGET_PATTERN,
                    },
                },
            )
        )
    trace_label_manifest: dict[str, int] = {}
    trace_maps: Any = None
    if args.fusion in {
        "tracefovea_human_gate",
        "eviweave_human_gate",
        "eviweave_normalized",
        "eviblend_safe_joint",
        "trace_calibrated_bridge",
    }:
        trace_label_root = args.trace_label_root
        if not trace_label_root.is_absolute():
            trace_label_root = args.project_root / trace_label_root
        manifest_path = trace_label_root / "manifest.jsonl"
        maps_path = trace_label_root / "multiscale_trace_maps.npz"
        if not manifest_path.is_file() or not maps_path.is_file():
            raise FileNotFoundError(
                f"missing human trace supervision under {trace_label_root}"
            )
        trace_label_manifest = {
            str(row["id"]): int(row["label_index"])
            for row in read_jsonl(manifest_path)
        }
        missing_trace_ids = [
            str(row["id"])
            for row in rows
            if str(row["id"]) not in trace_label_manifest
        ]
        if missing_trace_ids:
            raise RuntimeError(
                f"{len(missing_trace_ids)} training rows lack trace labels; "
                f"first={missing_trace_ids[:3]}"
            )
        trace_maps = np.load(maps_path)
        for channel in ("context", "read", "core"):
            if channel not in trace_maps.files:
                raise RuntimeError(f"trace label archive lacks {channel}")
    resume_checkpoint = args.resume_checkpoint
    if resume_checkpoint is not None and not resume_checkpoint.is_absolute():
        resume_checkpoint = args.project_root / resume_checkpoint
    if resume_checkpoint is None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    else:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(resume_checkpoint)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    model_state_checkpoint = (
        resume_checkpoint
        if resume_checkpoint is not None
        else parent_checkpoint
        if recovery is not None and args.recovery_stage == "B"
        else None
    )
    initial_language_adapter = args.initial_language_adapter
    if initial_language_adapter is not None and not initial_language_adapter.is_absolute():
        initial_language_adapter = args.project_root / initial_language_adapter
    if initial_language_adapter is not None:
        if not (initial_language_adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(initial_language_adapter)
    reward_judge_model_path = args.reward_judge_model
    if reward_judge_model_path is not None and not reward_judge_model_path.is_absolute():
        reward_judge_model_path = args.project_root / reward_judge_model_path
    if reward_judge_model_path is not None and not reward_judge_model_path.is_dir():
        raise FileNotFoundError(reward_judge_model_path)

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(args.model),
        local_files_only=True,
        max_pixels=max(
            args.processor_max_pixels,
            args.global_processor_max_pixels,
            args.fine_processor_max_pixels,
            (
                args.continuous_global_max_pixels
                if args.continuous_native_global
                else 0
            ),
        ),
        min_pixels=args.processor_min_pixels,
    )
    model, model_type = load_qwen_vlm(args.model, device=device)
    model.eval()
    model.config.use_cache = False
    model.requires_grad_(False)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Transformers activates layer checkpointing only while the module is
        # in training mode.  All LLM parameters remain frozen and Qwen3-VL's
        # configured dropout is zero, so this changes memory scheduling only.
        model.model.language_model.train()
    policy_model = None
    if recovery is not None and recovery.train_language_lora:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        resume_adapter = (
            model_state_checkpoint / "language_adapter"
            if model_state_checkpoint is not None and model_state_checkpoint.is_dir()
            else None
        )
        if initial_language_adapter is not None:
            policy_model = PeftModel.from_pretrained(
                model,
                str(initial_language_adapter),
                adapter_name="default",
                is_trainable=True,
            )
        elif resume_adapter is not None:
            if not resume_adapter.is_dir():
                raise FileNotFoundError(
                    f"recovery resume lacks language_adapter: {resume_adapter}"
                )
            policy_model = PeftModel.from_pretrained(
                model, str(resume_adapter), is_trainable=True
            )
        else:
            # Reset immediately before adapter creation so B-L/E-L/E-LB get
            # the same LoRA initialization even if their frozen visual setup
            # constructed a different number of tensors beforehand.
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            policy_model = get_peft_model(
                model,
                LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=LANGUAGE_LORA_TARGET_PATTERN,
                ),
            )
        if args.recovery_objective == "grpo":
            assert initial_language_adapter is not None
            policy_model.load_adapter(
                str(initial_language_adapter),
                adapter_name="reference",
                is_trainable=False,
            )
            synchronize_policy_reference_weights(policy_model)
            activate_policy_adapter(policy_model, "default", trainable=True)
            policy_model.eval()
        else:
            policy_model.train()
        model = policy_model.get_base_model()
        model.model.visual.eval()
    reward_judge_tokenizer = None
    semantic_reward_judge = None
    if reward_judge_model_path is not None:
        reward_judge_tokenizer, semantic_reward_judge = load_semantic_reward_judge(
            reward_judge_model_path, device=device
        )
    selector = load_selector(
        selector_checkpoint,
        int(model.config.vision_config.hidden_size),
        device,
    )
    selector_teacher = None
    if recovery is not None and recovery.train_ptea:
        # The student and immutable teacher start from the exact same artifact.
        # Loading twice avoids shared storage or accidental teacher updates.
        selector_teacher = load_selector(
            selector_checkpoint,
            int(model.config.vision_config.hidden_size),
            device,
        )
        # Keep dropout disabled while retaining autograd. This makes the
        # student exactly equal to its teacher at initialization and preserves
        # paired discrete regions across E-P/E-PL/E-PLB.
        selector.requires_grad_(True).eval()
        selector_teacher.requires_grad_(False).eval()
        if (
            args.recovery_stage == "B"
            and resume_checkpoint is None
            and parent_state is not None
        ):
            selector.load_state_dict(parent_state["selector_state_dict"], strict=True)
    else:
        selector.requires_grad_(False).eval()
    # Parent optimizer/RNG tensors are validation-only.  Stage B deliberately
    # starts a new optimizer and must not retain or accidentally reuse them.
    if recovery is not None and args.recovery_stage == "B":
        parent_state = None
        parent_protocol = None
    context_need_head = None
    context_need_checkpoint_path = args.trace_split_context_need_checkpoint
    if context_need_checkpoint_path is not None:
        if not context_need_checkpoint_path.is_absolute():
            context_need_checkpoint_path = (
                args.project_root / context_need_checkpoint_path
            )
        context_need_head = load_context_need_head(
            context_need_checkpoint_path,
            device=device,
        )
        context_need_head.requires_grad_(False).eval()
    adaptive_box_head = None
    adaptive_box_teacher = None
    adaptive_box_checkpoint_path = args.adaptive_box_head
    if adaptive_box_checkpoint_path is not None:
        if not adaptive_box_checkpoint_path.is_absolute():
            adaptive_box_checkpoint_path = (
                args.project_root / adaptive_box_checkpoint_path
            )
        adaptive_box_head, _ = load_adaptive_box_head(
            adaptive_box_checkpoint_path,
            device=device,
        )
        if recovery is not None and recovery.train_h_safe:
            # Student and teacher are independent copies of the same frozen
            # P3 H-Safe artifact.  The teacher never enters the optimizer.
            adaptive_box_teacher, _ = load_adaptive_box_head(
                adaptive_box_checkpoint_path,
                device=device,
            )
            adaptive_box_head.requires_grad_(True).eval()
            adaptive_box_teacher.requires_grad_(False).eval()
        else:
            adaptive_box_head.requires_grad_(False).eval()
    evicontour_utility = None
    evicontour_utility_checkpoint = None
    evicontour_utility_path = args.evicontour_utility_checkpoint
    if evicontour_utility_path is not None:
        if not evicontour_utility_path.is_absolute():
            evicontour_utility_path = args.project_root / evicontour_utility_path
        evicontour_utility, evicontour_utility_checkpoint = (
            load_evicontour_utility_head(
                str(evicontour_utility_path), map_location="cpu"
            )
        )
        # The utility head is a frozen region-policy component.  Keep it on
        # CPU so matched Bridge training has the same inference contract
        # without consuming additional GPU memory.
        evicontour_utility.requires_grad_(False).eval()
    evitree_need_head = None
    evitree_need_checkpoint = None
    if args.evitree_need_head is not None:
        need_path = args.evitree_need_head
        if not need_path.is_absolute():
            need_path = args.project_root / need_path
        evitree_need_head, evitree_need_checkpoint = (
            load_scalar_evidence_need_head(str(need_path), map_location=device)
        )
        evitree_need_head.to(device).requires_grad_(False).eval()
    evitree_split_policy = None
    evitree_split_checkpoint = None
    evitree_split_threshold = None
    if args.evitree_split_policy is not None:
        split_path = args.evitree_split_policy
        if not split_path.is_absolute():
            split_path = args.project_root / split_path
        evitree_split_policy, evitree_split_checkpoint = (
            load_hierarchical_split_policy_head(
                str(split_path), map_location=device
            )
        )
        evitree_split_policy.to(device).requires_grad_(False).eval()
        calibrated = float(
            evitree_split_checkpoint["model_config"]["split_threshold"]
        )
        evitree_split_threshold = (
            float(args.evitree_split_threshold)
            if args.evitree_split_threshold >= 0
            else calibrated
        )
    projection = build_projection(
        int(model.config.text_config.hidden_size),
        args.projection_dim,
        args.projection_seed,
        device,
    )
    residual_bound = (
        args.max_relative_residual if args.max_relative_residual > 0 else None
    )
    safe_joint_residual_mixer: SafeJointResidualMixer | None = None
    trace_calibrated_bridge_residual: (
        TraceCalibratedBridgeResidual | None
    ) = None
    if args.fusion in {
        "sparse_bridge",
        "eviblend_safe_joint",
        "trace_calibrated_bridge",
    }:
        bridge: torch.nn.Module = SparseGlobalLocalEvidenceBridge(
            int(model.config.vision_config.hidden_size),
            bridge_dim=args.bridge_dim,
            heads=args.bridge_heads,
            neighborhood_radius=args.neighborhood_radius,
            spatial_merge_size=int(model.config.vision_config.spatial_merge_size),
            max_relative_residual=residual_bound,
        ).to(device=device, dtype=torch.bfloat16)
        if args.fusion in {
            "eviblend_safe_joint",
            "trace_calibrated_bridge",
        }:
            initial_bridge_checkpoint = args.initial_bridge_checkpoint
            assert initial_bridge_checkpoint is not None
            if not initial_bridge_checkpoint.is_absolute():
                initial_bridge_checkpoint = (
                    args.project_root / initial_bridge_checkpoint
                )
            initial_payload = torch.load(
                initial_bridge_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            if initial_payload.get("version") != (
                "evivit_v3_sparse_mid_bridge_v1"
            ):
                raise RuntimeError(
                    "safe v4 calibration requires an EviViT-v4 sparse "
                    "bridge checkpoint"
                )
            bridge.load_state_dict(initial_payload["bridge"], strict=True)
            bridge.requires_grad_(False)
            if args.fusion == "eviblend_safe_joint":
                safe_joint_residual_mixer = SafeJointResidualMixer(
                    int(model.config.vision_config.hidden_size),
                    blend_dim=args.safe_joint_blend_dim,
                    maximum_gate=args.safe_joint_maximum_gate,
                    minimum_importance=args.safe_joint_minimum_importance,
                    maximum_importance=args.safe_joint_maximum_importance,
                ).to(device=device, dtype=torch.bfloat16)
            else:
                trace_calibrated_bridge_residual = (
                    TraceCalibratedBridgeResidual(
                        int(model.config.vision_config.hidden_size),
                        calibration_dim=args.trace_calibration_dim,
                        maximum_calibration=args.trace_calibration_maximum,
                        minimum_importance=(
                            args.trace_calibration_minimum_importance
                        ),
                        maximum_importance=(
                            args.trace_calibration_maximum_importance
                        ),
                        amplitude_mode=(
                            args.trace_calibration_amplitude_mode
                        ),
                        preserve_float32_importance=(
                            args.trace_calibration_preserve_fp32_importance
                        ),
                    ).to(device=device, dtype=torch.bfloat16)
                )
    elif args.fusion == "evislot":
        bridge = EvidenceSlotCompressor(
            int(model.config.vision_config.hidden_size),
            slot_dim=args.evislot_dim,
            heads=args.evislot_heads,
            slots_per_region=args.evislot_slots_per_region,
            spatial_merge_size=int(
                model.config.vision_config.spatial_merge_size
            ),
            maximum_residual_scale=args.evislot_maximum_residual_scale,
            evidence_bias=args.evislot_evidence_bias,
            enable_spatial_anchor_bias=args.evislot_spatial_anchor_bias,
            spatial_anchor_strength=args.evislot_spatial_anchor_strength,
            spatial_anchor_sigma=args.evislot_spatial_anchor_sigma,
        ).to(device=device, dtype=torch.bfloat16)
    elif args.fusion in {
        "tracefovea",
        "tracefovea_human_gate",
        "eviweave_human_gate",
        "eviweave_normalized",
    }:
        if residual_bound is None:
            raise ValueError("TraceFovea requires a positive residual bound")
        bridge = TraceFoveaBridgeAdapter(
            TraceFoveationBlock(
                int(model.config.vision_config.hidden_size),
                fovea_dim=args.fovea_dim,
                spatial_merge_size=int(model.config.vision_config.spatial_merge_size),
                max_relative_residual=float(residual_bound),
                use_child_gate=args.fusion
                in {
                    "tracefovea_human_gate",
                    "eviweave_human_gate",
                    "eviweave_normalized",
                },
                modulate_child_residual=(
                    args.trace_residual_modulation
                    or args.fusion == "eviweave_normalized"
                ),
                normalized_child_routing=(
                    args.fusion == "eviweave_normalized"
                ),
                preserve_float32_importance=(
                    args.fusion == "eviweave_normalized"
                ),
                minimum_child_importance=args.minimum_child_importance,
                maximum_child_importance=args.maximum_child_importance,
            )
        ).to(device=device, dtype=torch.bfloat16)
    else:
        if residual_bound is None:
            raise ValueError("EviRelay requires a positive residual bound")
        bridge = EviRelayBridgeAdapter(
            EviRelayBlock(
                int(model.config.vision_config.hidden_size),
                relay_dim=args.relay_dim,
                relay_tokens=args.relay_tokens,
                max_relative_residual=float(residual_bound),
            )
        ).to(device=device, dtype=torch.bfloat16)
    if args.warm_start_bridge_checkpoint is not None:
        warm_start_path = args.warm_start_bridge_checkpoint
        if not warm_start_path.is_absolute():
            warm_start_path = args.project_root / warm_start_path
        warm_start_payload = torch.load(
            warm_start_path,
            map_location="cpu",
            weights_only=False,
        )
        expected_warm_start_version = {
            "sparse_bridge": "evivit_v3_sparse_mid_bridge_v1",
            "tracefovea": "tracefovea_fixed5k_v1",
            "tracefovea_human_gate": "tracefovea_human_gate_fixed5k_v1",
        }[args.fusion]
        if args.evitree_v2:
            expected_warm_start_version = (
                "evitree_v2_parent_fusion_human_gate_v1"
                if args.fusion == "tracefovea_human_gate"
                else "evitree_v2_parent_fusion_v1"
            )
        if warm_start_payload.get("version") != expected_warm_start_version:
            raise RuntimeError(
                "warm-start checkpoint version "
                f"{warm_start_payload.get('version')} does not match "
                f"{expected_warm_start_version}"
            )
        bridge.load_state_dict(warm_start_payload["bridge"], strict=True)
    progressive_bridge_blocks = tuple(args.progressive_bridge_blocks)
    if args.fusion in {
        "eviweave_human_gate",
        "eviweave_normalized",
    } and not progressive_bridge_blocks:
        progressive_bridge_blocks = (20, 24)
    if args.fusion not in {
        "eviweave_human_gate",
        "eviweave_normalized",
    } and progressive_bridge_blocks:
        raise ValueError(
            "--progressive-bridge-blocks is reserved for EviWeave variants"
        )
    if args.fusion == "evislot":
        if not isinstance(bridge, EvidenceSlotCompressor):
            raise RuntimeError("EviSlot compressor construction failed")
        encoder: torch.nn.Module = Qwen3VLEviSlotEncoder(
            model.model.visual,
            bridge,
            insertion_block=args.insertion_block,
            freeze_vision=True,
        )
    else:
        encoder = Qwen3VLEviViTV3MidEncoder(
            model.model.visual,
            bridge,
            insertion_block=args.insertion_block,
            freeze_vision=True,
            progressive_bridge_blocks=progressive_bridge_blocks,
            safe_joint_residual_mixer=safe_joint_residual_mixer,
            safe_joint_residual_blocks=(
                (args.safe_joint_block,)
                if safe_joint_residual_mixer is not None
                else ()
            ),
            trace_calibrated_bridge_residual=(
                trace_calibrated_bridge_residual
            ),
            global_frame_joint_topology="global_hub",
        )
    trainable_module: torch.nn.Module = (
        safe_joint_residual_mixer
        if safe_joint_residual_mixer is not None
        else (
            trace_calibrated_bridge_residual
            if trace_calibrated_bridge_residual is not None
            else bridge
        )
    )
    if safe_joint_residual_mixer is not None:
        bridge.eval()
        safe_joint_residual_mixer.train()
    elif trace_calibrated_bridge_residual is not None:
        bridge.eval()
        trace_calibrated_bridge_residual.train()
    else:
        bridge.train()
    scheduler = None
    recovery_trainable_names: list[str] = []
    recovery_bridge_parameters: list[torch.nn.Parameter] = []
    recovery_ptea_parameters: list[torch.nn.Parameter] = []
    recovery_h_safe_parameters: list[torch.nn.Parameter] = []
    if recovery is not None:
        # Keep the full frozen P3 forward in every E-* lane.  Only this
        # parameter boundary changes between lanes; PTEA, ContextNeed, H-Safe,
        # Qwen-ViT and the remaining host stay frozen.
        bridge.requires_grad_(recovery.train_bridge)
        bridge.train(recovery.train_bridge)
        language_named = (
            list(policy_model.named_parameters()) if policy_model is not None else []
        )
        bridge_named = list(bridge.named_parameters())
        frozen_host_named = (
            list(model.named_parameters()) if policy_model is None else []
        )
        all_named = [
            *((f"policy.{name}", parameter) for name, parameter in language_named),
            *((f"host.{name}", parameter) for name, parameter in frozen_host_named),
            *((f"encoder.{name}", parameter) for name, parameter in encoder.named_parameters()),
            *((f"selector.{name}", parameter) for name, parameter in selector.named_parameters()),
            *(
                (f"selector_teacher.{name}", parameter)
                for name, parameter in (
                    selector_teacher.named_parameters()
                    if selector_teacher is not None
                    else ()
                )
            ),
            *(
                (f"context_need.{name}", parameter)
                for name, parameter in (
                    context_need_head.named_parameters()
                    if context_need_head is not None
                    else ()
                )
            ),
            *(
                (f"h_safe.{name}", parameter)
                for name, parameter in (
                    adaptive_box_head.named_parameters()
                    if adaptive_box_head is not None
                    else ()
                )
            ),
            *(
                (f"h_safe_teacher.{name}", parameter)
                for name, parameter in (
                    adaptive_box_teacher.named_parameters()
                    if adaptive_box_teacher is not None
                    else ()
                )
            ),
        ]
        recovery_trainable_names = assert_recovery_trainable_scope(
            lane=recovery,
            language_named_parameters=language_named,
            bridge_named_parameters=bridge_named,
            ptea_named_parameters=list(selector.named_parameters()),
            h_safe_named_parameters=(
                list(adaptive_box_head.named_parameters())
                if adaptive_box_head is not None
                else ()
            ),
            all_named_parameters=all_named,
        )
        language_parameters = [
            parameter
            for name, parameter in language_named
            if parameter.requires_grad and "lora_" in name
        ]
        bridge_parameters = [
            parameter for _, parameter in bridge_named if parameter.requires_grad
        ]
        ptea_parameters = [
            parameter for parameter in selector.parameters() if parameter.requires_grad
        ]
        h_safe_parameters = [
            parameter
            for parameter in (
                adaptive_box_head.parameters()
                if adaptive_box_head is not None
                else ()
            )
            if parameter.requires_grad
        ]
        recovery_bridge_parameters = bridge_parameters
        recovery_ptea_parameters = ptea_parameters
        recovery_h_safe_parameters = h_safe_parameters
        parameter_groups = recovery_optimizer_groups(
            lane=recovery,
            language_parameters=language_parameters,
            bridge_parameters=bridge_parameters,
            ptea_parameters=ptea_parameters,
            h_safe_parameters=h_safe_parameters,
            language_learning_rate=args.language_learning_rate,
            bridge_learning_rate=args.bridge_learning_rate,
            ptea_learning_rate=args.ptea_learning_rate,
            h_safe_learning_rate=args.h_safe_learning_rate,
        )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=args.weight_decay,
            betas=RECOVERY_ADAMW_BETAS,
            eps=RECOVERY_ADAMW_EPS,
        )
        trainable_parameters_for_clip = [
            parameter
            for group in parameter_groups
            for parameter in group["params"]
        ]
        trainable = sum(parameter.numel() for parameter in trainable_parameters_for_clip)
        from transformers import get_scheduler

        remaining_micro_steps = args.max_steps - recovery_stage_start_micro_step
        if remaining_micro_steps <= 0 or (
            remaining_micro_steps % args.gradient_accumulation
        ):
            raise ValueError("recovery stage length must align to the optimizer boundary")
        optimizer_steps = remaining_micro_steps // args.gradient_accumulation
        if dynamic_grpo:optimizer_steps=args.effective_updates
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps(optimizer_steps, args.warmup_ratio),
            num_training_steps=optimizer_steps,
        )
    else:
        trainable = sum(
            parameter.numel()
            for parameter in trainable_module.parameters()
            if parameter.requires_grad
        )
        if trainable != sum(
            parameter.numel()
            for parameter in encoder.parameters()
            if parameter.requires_grad
        ):
            raise RuntimeError("parameters outside the evidence bridge are trainable")
        optimizer = torch.optim.AdamW(
            (
                parameter
                for parameter in trainable_module.parameters()
                if parameter.requires_grad
            ),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        trainable_parameters_for_clip = list(trainable_module.parameters())
    start_step = recovery_stage_start_micro_step + 1
    if model_state_checkpoint is not None:
        checkpoint_path = (
            model_state_checkpoint / "bridge_checkpoint.pt"
            if recovery is not None and model_state_checkpoint.is_dir()
            else model_state_checkpoint
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_version = {
            "sparse_bridge": "evivit_v3_sparse_mid_bridge_v1",
            "tracefovea": "tracefovea_fixed5k_v1",
            "tracefovea_human_gate": (
                "tracefovea_trace_residual_fixed5k_v1"
                if args.trace_residual_modulation
                else "tracefovea_human_gate_fixed5k_v1"
            ),
            "eviweave_human_gate": "eviweave_human_gate_fixed5k_v1",
            "eviweave_normalized": "eviweave_normalized_fixed5k_v2",
            "evirelay": "evirelay_fixed5k_v1",
            "eviblend_safe_joint": "eviblend_safe_joint_fixed5k_v1",
            "evislot": "evislot_midvit_fixedslots_v1",
            "trace_calibrated_bridge": (
                trace_calibrated_checkpoint_version(
                    args.trace_calibration_amplitude_mode,
                    preserve_fp32_importance=(
                        args.trace_calibration_preserve_fp32_importance
                    ),
                )
            ),
        }[args.fusion]
        if args.evitree_v2:
            expected_version = (
                "evitree_v2_parent_fusion_human_gate_v1"
                if args.fusion == "tracefovea_human_gate"
                else "evitree_v2_parent_fusion_v1"
            )
        if checkpoint.get("version") != expected_version:
            raise RuntimeError(
                f"resume checkpoint version={checkpoint.get('version')} "
                f"does not match {expected_version}"
            )
        bridge.load_state_dict(checkpoint["bridge"], strict=True)
        if safe_joint_residual_mixer is not None:
            safe_joint_residual_mixer.load_state_dict(
                checkpoint["safe_joint_residual_mixer"],
                strict=True,
            )
        if trace_calibrated_bridge_residual is not None:
            trace_calibrated_bridge_residual.load_state_dict(
                checkpoint["trace_calibrated_bridge_residual"],
                strict=True,
            )
        if recovery is None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_step = int(checkpoint["step"]) + 1
            if start_step > args.max_steps:
                raise ValueError("resume checkpoint is already beyond --max-steps")
            torch.set_rng_state(checkpoint["torch_rng_state"])
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
            random.setstate(checkpoint["python_rng_state"])
    merge_size = int(model.config.vision_config.spatial_merge_size)
    merged_stride = merge_size * int(model.config.vision_config.patch_size)
    config = {
        **vars(args),
        "train_manifest": str(manifest),
        "selector_checkpoint": str(selector_checkpoint),
        "trainable_parameters": trainable,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else None,
        "recovery_stage_start_micro_step": recovery_stage_start_micro_step,
        "start_step": start_step,
        "progressive_bridge_blocks": list(progressive_bridge_blocks),
        "recovery_trainable_names": recovery_trainable_names,
    }
    recovery_fingerprint = None
    if recovery is not None:
        recovery_protocol = {
            "format_version": (
                "evivit_visual_cot_recovery_grpo_p3_v1"
                if args.recovery_objective == "grpo"
                else "evivit_visual_cot_recovery_sft_v1"
            ),
            "lane": recovery.name,
            "objective": args.recovery_objective,
            "manifest": str(manifest),
            "manifest_sha256": _sha256_file(manifest),
            "model": str(args.model),
            "selector_checkpoint": str(selector_checkpoint),
            "selector_sha256": _sha256_file(selector_checkpoint),
            "bridge_initialization": str(args.warm_start_bridge_checkpoint),
            "rows": len(rows),
            "max_steps": args.max_steps,
            "gradient_accumulation": args.gradient_accumulation,
            "seed": args.seed,
            "formal_p3": _recovery_formal_p3_fields(args),
            "optimizer": {
                "language_learning_rate": args.language_learning_rate,
                "bridge_learning_rate": args.bridge_learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "scheduler": "cosine",
                "betas": list(RECOVERY_ADAMW_BETAS),
                "eps": RECOVERY_ADAMW_EPS,
            },
            "epoch_order": RECOVERY_EPOCH_ORDER_VERSION,
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_pattern": LANGUAGE_LORA_TARGET_PATTERN,
            },
            "trainable_names": recovery_trainable_names,
        }
        if args.recovery_objective == "grpo":
            assert initial_language_adapter is not None
            recovery_protocol["initial_language_adapter"] = str(
                initial_language_adapter
            )
            recovery_protocol["grpo"] = {
                "rollouts_per_prompt": args.grpo_rollouts_per_prompt,
                "rollout_batch_size": args.grpo_rollout_batch_size,
                "max_answer_tokens": args.grpo_max_answer_tokens,
                "temperature": args.grpo_temperature,
                "top_p": args.grpo_top_p,
                "clip_epsilon": args.grpo_clip_epsilon,
                "kl_coefficient": args.grpo_kl_coefficient,
                "reward": {
                    "exact": 1.0,
                    "strict_normalized_equal": 0.5,
                    "semantic_judge_correct": 1.0,
                    "wrong": 0.0,
                    "invalid": -0.2,
                },
                "reward_judge_model": (
                    str(reward_judge_model_path)
                    if reward_judge_model_path is not None
                    else None
                ),
                "reward_judge_prompt_version": (
                    SEMANTIC_REWARD_JUDGE_VERSION
                    if reward_judge_model_path is not None
                    else None
                ),
                "reward_judge_mode": (
                    "text_only_semantic_fallback_after_deterministic_match"
                    if reward_judge_model_path is not None
                    else None
                ),
                "visual_components": "all frozen; normal P3 allocation retained",
                "region_count": "one or two decisive plus one context under max_regions=3",
            }
            recovery_protocol['grpo'].update(binary_reward_v2=args.binary_reward_v2,
                effective_updates=args.effective_updates,sampled_group_cap=args.max_steps,
                save_effective_updates=args.save_effective_updates,
                skip_zero_variance=dynamic_grpo)
            if args.binary_reward_v2:
                recovery_protocol['grpo']['reward'].update(strict_normalized_equal=1.,invalid=0.)
                recovery_protocol['grpo']['reward_parser']='shared_final_round_reward_v2'
        if recovery.requires_router_aux:
            assert router_alignment is not None
            recovery_protocol.update(
                {
                    "recovery_stage": args.recovery_stage,
                    "parent_protocol_fingerprint": parent_protocol_fingerprint,
                    "stage_start_micro_step": recovery_stage_start_micro_step,
                }
            )
            recovery_protocol["optimizer"].update(
                {
                    "ptea_learning_rate": args.ptea_learning_rate,
                    "h_safe_learning_rate": args.h_safe_learning_rate,
                    "stage_optimizer_reset": args.recovery_stage == "B",
                    "stage_training_steps": (
                        args.max_steps - recovery_stage_start_micro_step
                    )
                    // args.gradient_accumulation,
                }
            )
            recovery_protocol["router"] = {
                "stage": args.recovery_stage,
                "stage_start_epoch": (
                    recovery_stage_start_micro_step / len(rows)
                ),
                "alignment": router_alignment.protocol_fields(),
                "teacher_selector_sha256": _sha256_file(selector_checkpoint),
                "decoder_inputs": "detached_probabilities",
                "answer_ce_to_ptea": "disabled_by_discrete_decoder",
                "ptea_learning_rate": args.ptea_learning_rate,
                "loss": {
                    "decisive": "bbox_soft_map_cross_entropy_plus_cosine",
                    "cosine_weight": args.ptea_cosine_weight,
                    "decisive_teacher_kl_weight": (
                        args.ptea_decisive_distillation_weight
                    ),
                    "context": "original_checkpoint_distillation_only",
                    "context_teacher_kl_weight": (
                        args.ptea_context_distillation_weight
                    ),
                },
                "h_safe": {
                    "mode": "trainable_geometry_only" if recovery.train_h_safe else "frozen",
                    "artifact_sha256": _sha256_file(adaptive_box_checkpoint_path),
                    "ptea_features": "detached",
                    "answer_ce_gradient": "disabled_by_discrete_boxes",
                    "teacher": "frozen_original_artifact",
                    "geometry_label_mask": (
                        h_safe_geometry_mask.protocol_fields()
                        if h_safe_geometry_mask is not None
                        else None
                    ),
                    "loss": {
                        "format_version": H_SAFE_GEOMETRY_LOSS_VERSION,
                        "target": (
                            "bounded_action_projection_of_raw_human_bbox"
                        ),
                        "supervised_roles": "all_available_decisive_1_to_2",
                        "context_role_supervision": (
                            "identity_and_frozen_teacher_only"
                        ),
                        "excess_area": "smooth_l1_positive_log_ratio",
                        "maximum_area_ratio": 1.8,
                        "giou_weight": args.h_safe_giou_weight,
                        "coverage_weight": args.h_safe_coverage_weight,
                        "excess_area_weight": args.h_safe_excess_area_weight,
                        "identity_weight": args.h_safe_identity_weight,
                        "drift_weight": args.h_safe_drift_weight,
                    },
                },
            }
        recovery_fingerprint = _protocol_fingerprint(recovery_protocol)
        recovery_protocol["protocol_fingerprint"] = recovery_fingerprint
        protocol_path = args.output_dir / "run_protocol.json"
        if resume_checkpoint is not None:
            resume_state = torch.load(
                resume_checkpoint / "training_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            if resume_state.get("format_version") != "evivit_visual_cot_recovery_sft_v1":
                raise ValueError("unsupported recovery resume checkpoint")
            if resume_state.get("protocol_fingerprint") != recovery_fingerprint:
                raise ValueError("recovery resume protocol fingerprint mismatch")
            resume_micro_step = int(resume_state.get("micro_step", -1))
            resume_optimizer_step = int(resume_state.get("optimizer_step", -1))
            expected_stage_optimizer_step = (
                resume_micro_step - recovery_stage_start_micro_step
            ) // args.gradient_accumulation
            saved_stage_optimizer_step = resume_state.get("stage_optimizer_step")
            if (
                resume_micro_step < recovery_stage_start_micro_step
                or resume_micro_step % args.gradient_accumulation
                or resume_optimizer_step
                != resume_micro_step // args.gradient_accumulation
                or (
                    saved_stage_optimizer_step is not None
                    and int(saved_stage_optimizer_step)
                    != expected_stage_optimizer_step
                )
                or (
                    args.recovery_stage == "B"
                    and saved_stage_optimizer_step is None
                )
            ):
                raise ValueError("recovery resume step/optimizer boundary is invalid")
            if recovery.train_ptea:
                selector_state = resume_state.get("selector_state_dict")
                if selector_state is None:
                    raise ValueError("router recovery resume lacks PTEA student state")
                selector.load_state_dict(selector_state, strict=True)
            elif resume_state.get("selector_state_dict") is not None:
                raise ValueError("answer-only recovery checkpoint contains PTEA state")
            if recovery.train_h_safe:
                h_safe_state = resume_state.get("h_safe_state_dict")
                if h_safe_state is None or adaptive_box_head is None:
                    raise ValueError("H-Safe recovery resume lacks H-Safe student state")
                adaptive_box_head.load_state_dict(h_safe_state, strict=True)
            elif resume_state.get("h_safe_state_dict") is not None:
                raise ValueError("frozen-H-Safe recovery checkpoint contains H-Safe state")
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            assert scheduler is not None
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
            random.setstate(resume_state["python_random_state"])
            np.random.set_state(resume_state["numpy_random_state"])
            torch.set_rng_state(resume_state["torch_cpu_rng_state"].cpu())
            torch.cuda.set_rng_state(
                resume_state["torch_cuda_rng_state"].cpu(), device=0
            )
            start_step = resume_micro_step + 1
            if start_step > args.max_steps:
                raise ValueError("recovery resume checkpoint is beyond --max-steps")
            config["start_step"] = start_step
        else:
            protocol_path.write_text(
                json.dumps(recovery_protocol, ensure_ascii=False, indent=2, default=str)
                + "\n",
                encoding="utf-8",
            )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    log_path = args.output_dir / "train_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    effective_groups=0;effective_optimizer_steps=0;skipped_groups=0
    for step in range(start_step, args.max_steps + 1):
        epoch = (step - 1) // len(rows)
        offset = (step - 1) % len(rows)
        row = rows[epoch_order(len(rows), epoch, args.seed)[offset]]
        image_path = Path(str(row["image"]))
        if not image_path.is_absolute():
            image_path = args.project_root / image_path
        source = Image.open(image_path).convert("RGB")
        sample_global_token_budget = args.global_token_budget
        sample_fine_token_budget = args.fine_token_budget
        native_ratio_plan = None
        native_need_plan = None
        native_cap_global_plan = None
        native_pixel_global_plan = None
        continuous_global_plan = None
        patch_exchange_global_plan = None
        patch_exchange_fine_plan = None
        effective_patch_exchange_soft_floor_tokens = int(
            args.patch_exchange_soft_floor_tokens
        )
        if args.native_ratio_budget:
            native_ratio_plan = plan_native_ratio_budget(
                source.width,
                source.height,
                global_ratio=args.native_global_ratio,
                fine_ratio=args.native_fine_ratio,
                minimum_global_tokens=args.native_ratio_min_global_tokens,
                minimum_fine_tokens=args.native_ratio_min_fine_tokens,
                alignment=args.native_budget_alignment,
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
            )
            sample_global_token_budget = native_ratio_plan.global_tokens
            sample_fine_token_budget = native_ratio_plan.fine_tokens
        elif args.native_need_ratio_budget and not args.patch_exchange_tree:
            # Global is known before the PTEA pass and remains a continuous
            # fraction of Qwen's native token demand.  Fine is updated below
            # after the question-conditioned Need head sees the PTEA map.
            native_need_plan = plan_native_need_spread_budget(
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
                alignment=args.native_budget_alignment,
                minimum_pixels=args.processor_min_pixels,
                maximum_pixels=args.processor_max_pixels,
                patch_size=int(model.config.vision_config.patch_size),
                merge_size=merge_size,
            )
            sample_global_token_budget = native_need_plan.global_tokens
            sample_fine_token_budget = native_need_plan.fine_tokens
        native_global_used = False
        native_global_tokens = None
        if args.patch_exchange:
            patch_exchange_global_plan = plan_patch_exchange_global_view(
                source.width,
                source.height,
                minimum_pixels=args.processor_min_pixels,
                maximum_pixels=args.processor_max_pixels,
                global_scale=args.patch_exchange_global_scale,
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
                    minimum_pixels=args.processor_min_pixels,
                    maximum_pixels=args.processor_max_pixels,
                    global_scale=args.patch_exchange_global_scale,
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
                )
            native_global_tokens = (
                patch_exchange_global_plan.native_plan.realized_tokens
            )
            sample_global_token_budget = (
                patch_exchange_global_plan.realized_tokens
            )
            native_inputs_cpu = None
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
                native_global_tokens = continuous_global_plan.native_tokens
            else:
                native_pixel_global_plan = plan_native_pixel_view(
                    source.width,
                    source.height,
                    minimum_pixels=args.processor_min_pixels,
                    maximum_pixels=args.global_processor_max_pixels,
                    patch_size=int(model.config.vision_config.patch_size),
                    merge_size=merge_size,
                )
                native_global_tokens = native_pixel_global_plan.realized_tokens
            sample_global_token_budget = (
                native_pixel_global_plan.realized_tokens
            )
            native_inputs_cpu = None
        elif args.native_token_caps:
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
            native_global_tokens = native_cap_global_plan.native_tokens
            sample_global_token_budget = (
                native_cap_global_plan.requested_tokens
            )
            native_inputs_cpu = None
        elif args.native_global_total_token_budget:
            native_inputs_cpu = processor(
                text=[vision_probe_text(processor, model_type)],
                images=[source], return_tensors="pt"
            )
            native_grid = [
                int(value)
                for value in native_inputs_cpu["image_grid_thw"][0].tolist()
            ]
            native_global_tokens = (
                native_grid[0]
                * (native_grid[1] // merge_size)
                * (native_grid[2] // merge_size)
            )
            native_global_used = (
                native_global_tokens <= args.native_global_total_token_budget
            )
        else:
            native_inputs_cpu = None
        if native_global_used:
            assert native_inputs_cpu is not None
            global_view = source
            global_inputs = native_inputs_cpu.to(device)
        else:
            if native_inputs_cpu is not None:
                del native_inputs_cpu
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
                "training Global grid differs from Native-Pixel plan"
            )
        question_tokens = qwen_projected_question_tokens(
            model,
            processor.tokenizer,
            str(row["question"]),
            projection,
            device=device,
        )
        if recovery is not None and recovery.requires_router_aux:
            # qwen_projected_question_tokens is an inference helper. Clone its
            # result back into a normal tensor so trainable PTEA projections
            # may safely save it for weight-gradient computation.
            question_tokens = question_tokens.detach().clone()
        ptea_loss_result = None
        if recovery is not None and recovery.requires_router_aux:
            if selector_teacher is None:
                raise RuntimeError("router recovery lacks its frozen PTEA teacher")
            aligned_row = router_rows_by_id.get(str(row["id"]))
            if aligned_row is None:
                raise RuntimeError("training row is absent from aligned router aux")
            student_paths = forward_mid_ptea_training_paths(
                selector,
                prepared.selector_visual_input(selector),
                question_tokens,
            )
            with torch.no_grad():
                teacher_paths = forward_mid_ptea_training_paths(
                    selector_teacher,
                    prepared.selector_visual_input(selector_teacher),
                    question_tokens,
                )
            # Only detached probabilities cross the discrete top-k/box/PIL
            # boundary. The differentiable copies above remain solely for the
            # bbox/map and immutable-teacher objectives.
            allocation = allocate_trace_split_evidence(
                student_paths.decoder_decisive_probability,
                student_paths.decoder_context_probability,
                fine_token_budget=sample_fine_token_budget,
                max_regions=args.max_regions,
                minimum_region_tokens=args.minimum_region_tokens,
                context_fraction=args.trace_split_context_fraction,
                context_expansion_factor=(
                    args.trace_split_context_expansion_factor
                ),
                context_decoder=args.trace_split_context_decoder,
                budget_mode=args.trace_split_budget_mode,
                min_context_fraction=args.trace_split_min_context_fraction,
                max_context_fraction=args.trace_split_max_context_fraction,
                context_need_head=context_need_head,
                adaptive_minimum_region_tokens=(
                    args.trace_split_adaptive_region_minimum
                ),
            )
            decisive_target = bbox_soft_map(
                aligned_row.boxes,
                int(student_paths.decisive_probability.shape[-2]),
                int(student_paths.decisive_probability.shape[-1]),
                device=device,
            )
            ptea_loss_result = ptea_recovery_loss(
                student_paths,
                decisive_target,
                teacher_paths.decisive_probability,
                teacher_paths.context_probability,
                cosine_weight=args.ptea_cosine_weight,
                decisive_distillation_weight=(
                    args.ptea_decisive_distillation_weight
                ),
                context_distillation_weight=(
                    args.ptea_context_distillation_weight
                ),
            )
        elif args.control_evidence_policy is not None:
            digest = hashlib.sha256(str(row["id"]).encode("utf-8")).digest()
            sample_control_seed = (
                args.seed + int.from_bytes(digest[:8], "big")
            ) % (2**63 - 1)
            allocation = allocate_control_evidence(
                prepared.map_feature_grid,
                policy=args.control_evidence_policy,
                fine_token_budget=sample_fine_token_budget,
                max_regions=args.max_regions,
                minimum_region_tokens=args.minimum_region_tokens,
                random_seed=sample_control_seed,
            )
        else:
            allocation = allocate_mid_ptea_evidence(
                selector,
                prepared.map_feature_grid,
                question_tokens,
                fine_token_budget=sample_fine_token_budget,
                max_regions=args.max_regions,
                minimum_region_tokens=args.minimum_region_tokens,
                evidence_decoder=args.evidence_decoder,
                contour_relative_threshold=args.contour_relative_threshold,
                contour_context_margin=args.contour_context_margin,
                contour_minimum_residual_mass=(
                    args.contour_minimum_residual_mass
                ),
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
                seeded_basin_config=seeded_basin_config,
                evicontour_utility=evicontour_utility,
                evicontour_image_size=(source.width, source.height),
                evicontour_question=str(row["question"]),
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
        sample_evidence_need = None
        if args.native_need_ratio_budget:
            if evitree_need_head is None:
                raise RuntimeError("EviTree Need head was not loaded")
            need_features = scalar_need_feature_vector(
                allocation.probability_map.detach().float().cpu().numpy(),
                width=source.width,
                height=source.height,
                question=str(row["question"]),
            )
            with torch.inference_mode():
                sample_evidence_need = float(
                    evitree_need_head(
                        torch.from_numpy(need_features).to(device).unsqueeze(0)
                    ).item()
                )
            native_need_plan = plan_native_need_spread_budget(
                source.width,
                source.height,
                evidence_need=sample_evidence_need,
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
                alignment=args.native_budget_alignment,
                minimum_pixels=args.processor_min_pixels,
                maximum_pixels=args.processor_max_pixels,
                patch_size=int(model.config.vision_config.patch_size),
                merge_size=merge_size,
            )
            sample_fine_token_budget = native_need_plan.fine_tokens
        if args.evitree_v2:
            if evitree_split_policy is None or evitree_split_threshold is None:
                raise RuntimeError("EviTree-v2 split policy was not loaded")
            allocation = allocate_evitree_hierarchical_evidence(
                allocation.probability_map,
                fine_token_budget=sample_fine_token_budget,
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
                minimum_branch_depth=args.evitree_minimum_branch_depth,
                branch_density_power=args.evitree_branch_density_power,
                recenter_branch_tips=args.evitree_recenter_branch_tips,
                branch_expansion_factor=args.evitree_branch_expansion_factor,
                maximum_branch_iou=args.evitree_maximum_branch_iou,
            )
        if recovery is not None:
            recovery_region_roles = [
                str(budget.role) for budget in allocation.region_budgets
            ]
            allowed_recovery_roles = [
                [
                    "tracesplit_decisive_mode_1_focus",
                    "tracesplit_context_rank_1",
                ],
                [
                    "tracesplit_decisive_mode_1_focus",
                    "tracesplit_decisive_mode_2_focus",
                    "tracesplit_context_rank_1",
                ],
            ]
            if recovery_region_roles not in allowed_recovery_roles:
                raise RuntimeError(
                    "formal recovery must emit one or two ordered decisive "
                    "regions followed by exactly one context region; got "
                    f"{recovery_region_roles}"
                )
            recovery_decisive_count = len(recovery_region_roles) - 1
        h_safe_loss_result = None
        h_safe_supervised_loss = None
        h_safe_geometry_supervision_active = None
        if adaptive_box_head is not None and args.control_evidence_policy is None:
            # H-Safe consumes a detached view of the current PTEA state.  Its
            # refined boxes cross only the discrete pixel-read boundary, so
            # answer CE cannot update either router component.
            if recovery is not None and recovery.requires_router_aux:
                with torch.no_grad():
                    adaptive_features = adaptive_box_feature_vectors(
                        selector,
                        prepared.selector_visual_input(selector),
                        question_tokens,
                        allocation,
                        image_size=[source.width, source.height],
                    ).to(device)
            else:
                adaptive_features = adaptive_box_feature_vectors(
                    selector,
                    prepared.selector_visual_input(selector),
                    question_tokens,
                    allocation,
                    image_size=[source.width, source.height],
                ).to(device)
            anchor_boxes = torch.tensor(
                [
                    [float(value) / 1000.0 for value in budget.bbox]
                    for budget in allocation.region_budgets
                ],
                dtype=torch.float32,
                device=device,
            )
            if recovery is not None and recovery.train_h_safe:
                if adaptive_box_teacher is None or aligned_row is None:
                    raise RuntimeError("E-PHLB lacks frozen H-Safe teacher/targets")
                with torch.no_grad():
                    frozen_h_safe_residuals = adaptive_box_teacher(
                        adaptive_features.detach()
                    )
                h_safe_targets = torch.tensor(
                    aligned_row.boxes,
                    dtype=torch.float32,
                    device=device,
                )
                h_safe_loss_result = h_safe_geometry_loss(
                    adaptive_box_head,
                    adaptive_features,
                    anchor_boxes,
                    h_safe_targets,
                    frozen_teacher_residuals=frozen_h_safe_residuals,
                    giou_weight=args.h_safe_giou_weight,
                    coverage_weight=args.h_safe_coverage_weight,
                    excess_area_weight=args.h_safe_excess_area_weight,
                    identity_weight=args.h_safe_identity_weight,
                    drift_weight=args.h_safe_drift_weight,
                    supervised_decisive_anchors=recovery_decisive_count,
                )
                h_safe_geometry_supervision_active = (
                    aligned_row.id not in h_safe_geometry_mask_ids
                )
                h_safe_supervised_loss = (
                    h_safe_loss_result.total
                    if h_safe_geometry_supervision_active
                    else h_safe_loss_result.total * 0.0
                )
                adaptive_boxes = h_safe_loss_result.refined_boxes.detach().cpu()
            else:
                with torch.no_grad():
                    adaptive_residuals = adaptive_box_head(adaptive_features)
                adaptive_boxes = apply_residual(
                    anchor_boxes,
                    adaptive_residuals,
                    minimum_side=adaptive_box_head.config.minimum_side,
                    maximum_side=adaptive_box_head.config.maximum_side,
                ).detach().cpu()
            if not torch.isfinite(adaptive_boxes).all() or not bool(
                ((adaptive_boxes >= 0.0) & (adaptive_boxes <= 1.0)).all()
            ):
                raise RuntimeError("H-Safe produced invalid normalized boxes")
            allocation.region_budgets = [
                RegionBudget(
                    bbox=tuple(
                        int(round(float(value) * 1000.0))
                        for value in refined
                    ),
                    token_budget=budget.token_budget,
                    score=budget.score,
                    role=budget.role,
                    source_rank=budget.source_rank,
                )
                for budget, refined in zip(
                    allocation.region_budgets,
                    adaptive_boxes.tolist(),
                )
            ]
        # PatchExchange must see the final region set.  For v5 this is the
        # original TraceSplit allocation; for v8 it is the EviTree allocation
        # produced immediately above.  Planning before the tree rewrite would
        # train on different crops from inference.
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
                soft_floor_tokens=effective_patch_exchange_soft_floor_tokens,
                fill_soft_floor=args.patch_exchange_balanced_soft_floor,
                minimum_view_pixels=args.patch_exchange_view_min_pixels,
                maximum_view_pixels=args.processor_max_pixels,
                patch_size=int(model.config.vision_config.patch_size),
                merge_size=merge_size,
            )
        maximum_weight = max(
            (budget.score for budget in allocation.region_budgets), default=1.0
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
            if native_global_tokens is None:
                raise RuntimeError(
                    "EviContour-CSR requires the source-native Global token count"
                )
            continuous_fine_floor_plan = plan_continuous_fine_floor(
                int(native_global_tokens),
                [
                    int(budget.token_budget)
                    for budget in allocation.region_budgets
                ],
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
            args.native_token_caps
            and args.native_cap_fine_floor_native_ratio > 0
            and allocation.region_budgets
        ):
            continuous_fine_floor_plan = plan_continuous_fine_floor(
                int(native_global_tokens),
                [
                    int(budget.token_budget)
                    for budget in allocation.region_budgets
                ],
                native_ratio=args.native_cap_fine_floor_native_ratio,
                minimum_total_tokens=args.native_cap_minimum_fine_total,
                maximum_total_tokens=args.fine_token_budget,
            )
            region_fine_floors = list(
                continuous_fine_floor_plan.region_floor_tokens
            )
        fine_views: list[MidViTFineViewInput] = []
        realized_fine_tokens = 0
        native_cap_fine_plans: list[dict[str, Any]] = []
        native_pixel_fine_plans: list[dict[str, Any]] = []
        opened: list[Image.Image] = []
        for region_index, budget in enumerate(allocation.region_budgets):
            patch_exchange_view_plan = (
                patch_exchange_fine_plan.views[region_index]
                if patch_exchange_fine_plan is not None
                else None
            )
            if (
                patch_exchange_view_plan is not None
                and patch_exchange_view_plan.dropped
            ):
                continue
            pixel_box = policy_to_pixels(budget.bbox, *source.size)
            crop = source.crop(pixel_box).convert("RGB")
            if patch_exchange_view_plan is not None:
                fine_h = patch_exchange_view_plan.grid_height
                fine_w = patch_exchange_view_plan.grid_width
            elif args.native_pixel_contract:
                region_minimum_pixels = max(
                    args.processor_min_pixels,
                    int(region_fine_floors[region_index] * merged_stride**2),
                )
                native_pixel_fine_plan = plan_native_pixel_view(
                    crop.width,
                    crop.height,
                    minimum_pixels=region_minimum_pixels,
                    maximum_pixels=args.fine_processor_max_pixels,
                    patch_size=int(model.config.vision_config.patch_size),
                    merge_size=merge_size,
                )
                fine_h = native_pixel_fine_plan.grid_height
                fine_w = native_pixel_fine_plan.grid_width
                native_pixel_fine_plans.append(
                    native_pixel_fine_plan.to_dict()
                )
            elif args.native_token_caps:
                native_cap_fine_plan = plan_native_token_cap(
                    crop.width,
                    crop.height,
                    token_cap=budget.token_budget,
                    minimum_tokens=region_fine_floors[region_index],
                    minimum_pixels=args.processor_min_pixels,
                    maximum_pixels=args.processor_max_pixels,
                    patch_size=int(model.config.vision_config.patch_size),
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
            fine_views.append(
                MidViTFineViewInput(
                    pixel_values=fine_inputs["pixel_values"],
                    grid_thw=fine_inputs["image_grid_thw"],
                    bbox_xyxy=tuple(value / 1000.0 for value in budget.bbox),
                    evidence_weight=budget.score / max(maximum_weight, 1e-8),
                )
            )
            realized_fine_tokens += fine_h * fine_w
            opened.extend([crop, fine_image])

        vision = encoder.complete_from_prepared(
            prepared,
            fine_views,
            bridge_mode=args.bridge_mode,
            visual_output_mode=args.visual_output_mode,
            global_evidence_map=(
                allocation.probability_map
                if (
                    safe_joint_residual_mixer is not None
                    or trace_calibrated_bridge_residual is not None
                    or args.fusion == "evislot"
                )
                else None
            ),
        )
        if args.native_pixel_contract and args.fusion != "evislot":
            planned_fine = [
                int(plan["realized_tokens"])
                for plan in native_pixel_fine_plans
            ]
            if list(vision.fine_view_token_counts) != planned_fine:
                raise AssertionError(
                    "training Fine grids differ from Native-Pixel plans"
                )
        if patch_exchange_fine_plan is not None and args.fusion != "evislot":
            planned_fine = [
                int(view.realized_tokens)
                for view in patch_exchange_fine_plan.views
                if not view.dropped
            ]
            if (
                vision.global_image_tokens
                != patch_exchange_global_plan.realized_tokens
            ):
                raise AssertionError(
                    "training Global grid differs from PatchExchange plan"
                )
            if list(vision.fine_view_token_counts) != planned_fine:
                raise AssertionError(
                    "training Fine grids differ from PatchExchange plan"
                )
            if (
                vision.global_image_tokens + vision.fine_encoder_tokens
                > patch_exchange_fine_plan.total_token_ceiling
            ):
                raise AssertionError(
                    "PatchExchange training exceeded per-image native ceiling"
                )
        grpo_metrics: dict[str, float] | None = None
        if args.recovery_objective == "grpo":
            if policy_model is None:
                raise RuntimeError("formal P3 GRPO lacks a policy adapter")
            prompt_text = build_chat_text(
                processor,
                str(row["question"]),
                None,
                answer_protocol=args.answer_protocol,
                enable_thinking=(
                    False
                    if model_type == "qwen3_5" and args.disable_thinking
                    else None
                ),
            )
            prompt = prepare_evivit_visual_prompt(
                model,
                processor.tokenizer,
                prompt_text,
                vision.image_embeds,
                vision.deepstack_features,
                visual_coordinates(
                    vision,
                    coordinate_bins=args.coordinate_bins,
                    scale_bins=args.scale_bins,
                ),
                device=device,
            )
            if args.gradient_checkpointing:
                configure_grpo_language_checkpointing(
                    model.model.language_model, enabled=False
                )
            activate_policy_adapter(policy_model, "default", trainable=True)
            completions = sample_completions(
                model,
                processor.tokenizer,
                prompt,
                count=args.grpo_rollouts_per_prompt,
                max_new_tokens=args.grpo_max_answer_tokens,
                temperature=args.grpo_temperature,
                top_p=args.grpo_top_p,
                rollout_batch_size=args.grpo_rollout_batch_size,
            )
            if args.gradient_checkpointing:
                configure_grpo_language_checkpointing(
                    model.model.language_model, enabled=True
                )
            target = str(row.get("answer", ""))
            rollout_rows: list[dict[str, Any]] = []
            for completion in completions:
                prediction, parse_error = (parse_answer if args.binary_reward_v2 else direct_prediction)(completion.text)
                valid = parse_error is None
                exact = (
                    valid
                    and normalized_answer(prediction)
                    == normalized_answer(target)
                )
                relaxed = (
                    valid
                    and strict_relaxed_reward_match(prediction, target)
                )
                rollout_rows.append(
                    {
                        "prediction": prediction,
                        "token_ids": list(completion.token_ids),
                        "tokens": len(completion.token_ids),
                        "valid": valid,
                        "exact": exact,
                        "relaxed": relaxed,
                    }
                )
            uncertain = list(
                dict.fromkeys(
                    str(item["prediction"])
                    for item in rollout_rows
                    if item["valid"] and not item["exact"] and not item["relaxed"]
                )
            )
            judge_by_prediction: dict[str, dict[str, Any]] = {}
            if semantic_reward_judge is not None and uncertain:
                assert reward_judge_tokenizer is not None
                judged = (reliable_judge if args.binary_reward_v2 else semantic_reward_judge_batch)(
                    reward_judge_tokenizer,
                    semantic_reward_judge,
                    question=str(row["question"]),
                    target=target,
                    candidates=uncertain,
                    device=device,
                )
                judge_by_prediction = {
                    str(item["candidate"]): item for item in judged
                }
            rewards: list[float] = []
            for item in rollout_rows:
                judged = judge_by_prediction.get(str(item["prediction"]))
                if not item["valid"]:
                    reward_source = "invalid"
                elif item["exact"]:
                    reward_source = "exact"
                elif item["relaxed"]:
                    reward_source = "relaxed"
                elif judged is not None and judged["correct"]:
                    reward_source = "semantic_judge"
                elif judged is not None and judged["judge_error"]:
                    reward_source = "semantic_judge_error_fallback_wrong"
                else:
                    reward_source = "wrong"
                reward = (
                    1.0
                    if reward_source in {"exact", "semantic_judge"}
                    else 0.5
                    if reward_source == "relaxed"
                    else -0.2
                    if reward_source == "invalid"
                    else 0.0
                )
                item["reward"] = reward
                if args.binary_reward_v2:
                    reward=binary_reward(reward_source)
                    item['reward']=reward
                item["reward_source"] = reward_source
                item["semantic_judge"] = judged
                rewards.append(reward)
            advantages = group_advantages(rewards)
            accepted_group=not dynamic_grpo or active_group(rewards)
            if dynamic_grpo:
                effective_groups+=int(accepted_group)
                skipped_groups+=int(not accepted_group)
            policy_losses: list[float] = []
            kl_values: list[float] = []
            clip_values: list[float] = []
            completion_token_values: list[float] = []
            for completion, advantage, rollout_row in zip(
                completions, advantages, rollout_rows
            ):
                rollout_row["advantage"] = advantage
                if not accepted_group:continue
                ids = torch.tensor(
                    [list(completion.token_ids)],
                    dtype=torch.long,
                    device=device,
                )
                mask = torch.ones_like(ids, dtype=torch.bool)
                activate_policy_adapter(policy_model, "reference", trainable=False)
                with torch.no_grad():
                    reference_log_probs = completion_log_probs(
                        model, prompt, ids, mask
                    )
                activate_policy_adapter(policy_model, "default", trainable=True)
                current_log_probs = completion_log_probs(
                    model, prompt, ids, mask
                )
                rollout_loss, rollout_loss_metrics = grpo_clipped_loss(
                    current_log_probs=current_log_probs,
                    old_log_probs=current_log_probs.detach(),
                    reference_log_probs=reference_log_probs,
                    completion_mask=mask,
                    advantages=torch.tensor([advantage], device=device),
                    clip_epsilon=args.grpo_clip_epsilon,
                    kl_coefficient=args.grpo_kl_coefficient,
                )
                (
                    rollout_loss
                    / (
                        args.grpo_rollouts_per_prompt
                        * args.gradient_accumulation
                    )
                ).backward()
                policy_losses.append(
                    float(rollout_loss_metrics["policy_loss"].cpu())
                )
                kl_values.append(float(rollout_loss_metrics["kl"].cpu()))
                clip_values.append(
                    float(rollout_loss_metrics["clip_fraction"].cpu())
                )
                completion_token_values.append(
                    float(
                        rollout_loss_metrics["mean_completion_tokens"].cpu()
                    )
                )
                del ids, mask, reference_log_probs, current_log_probs
                del rollout_loss, rollout_loss_metrics
            grpo_metrics = {
                "reward_mean": sum(rewards) / len(rewards),
                "exact_rate": sum(
                    bool(value["exact"]) for value in rollout_rows
                )
                / len(rollout_rows),
                "relaxed_rate": sum(
                    bool(value["relaxed"]) for value in rollout_rows
                )
                / len(rollout_rows),
                "invalid_rate": sum(
                    not bool(value["valid"]) for value in rollout_rows
                )
                / len(rollout_rows),
                "zero_variance_group": float(
                    all(value == rewards[0] for value in rewards)
                ),
                "active_advantage_group": float(any(abs(value) > 0 for value in advantages)),
                "mean_absolute_advantage": sum(abs(value) for value in advantages) / len(advantages),
                "metric_semantics": "answer_token_accuracy legacy field contains reward_mean in GRPO mode",
                "policy_loss": sum(policy_losses) / max(1,len(policy_losses)),
                "kl": sum(kl_values) / max(1,len(kl_values)),
                "clip_fraction": sum(clip_values) / max(1,len(clip_values)),
                "completion_tokens": sum(completion_token_values)
                / max(1,len(completion_token_values)),
                "effective_groups": effective_groups,
                "skipped_groups": skipped_groups,
                "accepted_group": accepted_group,
            }
            with (args.output_dir / "grpo_rollouts.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "id": row["id"],
                            "question": row["question"],
                            "answer": target,
                            "regions": len(fine_views),
                            "rollouts": rollout_rows,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            qa_loss = torch.tensor(
                grpo_metrics["policy_loss"], device=device
            )
            token_accuracy = torch.tensor(
                grpo_metrics["reward_mean"], device=device
            )
            answer_tokens = int(round(grpo_metrics["completion_tokens"]))
            del prompt, completions
        else:
            qa_loss, token_accuracy, answer_tokens = answer_only_loss(
                model,
                processor,
                str(row["question"]),
                str(row.get("answer", "")),
                vision,
                device=device,
                coordinate_bins=args.coordinate_bins,
                scale_bins=args.scale_bins,
                answer_protocol=args.answer_protocol,
                native_global_mrope=args.native_global_mrope,
                enable_thinking=(
                    False
                    if model_type == "qwen3_5" and args.disable_thinking
                    else None
                ),
            )
        trace_supervision_loss = qa_loss.new_zeros(())
        trace_target_peak = None
        trace_global_supervision_active = None
        trace_supervision_by_block: dict[str, float] = {}
        if args.fusion in {
            "tracefovea_human_gate",
            "eviweave_human_gate",
            "eviweave_normalized",
            "eviblend_safe_joint",
            "trace_calibrated_bridge",
        }:
            label_index = trace_label_manifest[str(row["id"])]
            if args.fusion in {
                "eviblend_safe_joint",
                "trace_calibrated_bridge",
            }:
                if args.fusion == "eviblend_safe_joint":
                    if safe_joint_residual_mixer is None:
                        raise RuntimeError("EviBlend mixer is missing")
                    fine_gate_logits = (
                        encoder.last_safe_joint_gate_logits_by_block.get(
                            args.safe_joint_block
                        )
                    )
                    global_gate_logits = (
                        encoder.last_safe_joint_global_gate_logits_by_block.get(
                            args.safe_joint_block
                        )
                    )
                    fine_centers = encoder.last_safe_joint_fine_centers_xy
                    global_centers = encoder.last_safe_joint_global_centers_xy
                    trace_stage_name = str(args.safe_joint_block)
                else:
                    if trace_calibrated_bridge_residual is None:
                        raise RuntimeError(
                            "trace-calibrated bridge module is missing"
                        )
                    fine_gate_logits = (
                        encoder
                        .last_trace_calibrated_fine_importance_logits
                    )
                    global_gate_logits = (
                        encoder
                        .last_trace_calibrated_global_importance_logits
                    )
                    fine_centers = (
                        encoder.last_trace_calibrated_fine_centers_xy
                    )
                    global_centers = (
                        encoder.last_trace_calibrated_global_centers_xy
                    )
                    trace_stage_name = str(args.insertion_block)
                if fine_gate_logits is None or global_gate_logits is None:
                    raise RuntimeError(
                        "safe calibration importance logits are missing"
                    )
                if fine_centers is None or int(fine_centers.shape[0]) == 0:
                    raise RuntimeError(
                        "safe calibration requires fine evidence tokens"
                    )
                if (
                    global_centers is None
                    or int(global_centers.shape[0]) == 0
                ):
                    raise RuntimeError(
                        "safe calibration requires global tokens"
                    )
                fine_trace_probability = combine_trace_maps(
                    {
                        channel: torch.as_tensor(
                            trace_maps[channel][label_index],
                            device=device,
                            dtype=torch.float32,
                        )
                        for channel in ("read", "core")
                    },
                    weights={"read": 0.55, "core": 0.45},
                )
                context_probability = torch.as_tensor(
                    trace_maps["context"][label_index],
                    device=device,
                    dtype=torch.float32,
                )
                fine_target = sample_trace_distribution(
                    fine_trace_probability, fine_centers
                )
                fine_trace_loss = soft_distribution_cross_entropy(
                    fine_gate_logits, fine_target
                )
                context_mass = float(
                    context_probability.sum().detach().cpu()
                )
                global_trace_loss = None
                global_target = None
                trace_global_supervision_active = (
                    context_mass > 0 or not args.skip_empty_context_trace
                )
                if trace_global_supervision_active:
                    context_probability = context_probability / (
                        context_probability.sum().clamp_min(1e-12)
                    )
                    global_target = sample_trace_distribution(
                        context_probability, global_centers
                    )
                    global_trace_loss = soft_distribution_cross_entropy(
                        global_gate_logits, global_target
                    )
                available_trace_losses = [fine_trace_loss]
                if global_trace_loss is not None:
                    available_trace_losses.append(global_trace_loss)
                trace_supervision_loss = torch.stack(
                    available_trace_losses
                ).mean()
                trace_target_peak = float(
                    max(
                        [
                            fine_target.max().detach().cpu(),
                            *(
                                [global_target.max().detach().cpu()]
                                if global_target is not None
                                else []
                            ),
                        ]
                    )
                )
                trace_supervision_by_block[trace_stage_name] = float(
                    trace_supervision_loss.detach().cpu()
                )
                if global_trace_loss is not None:
                    trace_supervision_by_block[
                        f"{trace_stage_name}_global_context"
                    ] = float(global_trace_loss.detach().cpu())
                trace_supervision_by_block[
                    f"{trace_stage_name}_fine_read_core"
                ] = float(fine_trace_loss.detach().cpu())
            else:
                if not isinstance(bridge, TraceFoveaBridgeAdapter):
                    raise RuntimeError(
                        "human-gated TraceFovea adapter is missing"
                    )
                gate_centers = bridge.last_child_centers_xy
                if gate_centers is None:
                    raise RuntimeError(
                        "human-gated TraceFovea outputs are missing"
                    )
            if args.fusion == "tracefovea_human_gate":
                gate_logits = bridge.last_child_gate_logits
                if gate_logits is None:
                    raise RuntimeError("human gate logits are missing")
                trace_probability = combine_trace_maps(
                    {
                        channel: torch.as_tensor(
                            trace_maps[channel][label_index],
                            device=device,
                            dtype=torch.float32,
                        )
                        for channel in ("context", "read", "core")
                    }
                )
                trace_target = sample_trace_distribution(
                    trace_probability, gate_centers
                )
                trace_supervision_loss = soft_distribution_cross_entropy(
                    gate_logits, trace_target
                )
                trace_target_peak = float(trace_target.max().detach().cpu())
                trace_supervision_by_block[str(args.insertion_block)] = float(
                    trace_supervision_loss.detach().cpu()
                )
            elif args.fusion in {
                "eviweave_human_gate",
                "eviweave_normalized",
            }:
                gate_logits_by_block = encoder.last_bridge_gate_logits_by_block
                stage_blocks = (
                    args.insertion_block,
                    *progressive_bridge_blocks,
                )
                if tuple(gate_logits_by_block) != stage_blocks:
                    raise RuntimeError(
                        "EviWeave gate stages do not match configured blocks"
                    )
                stage_channels = ("context", "read", "core")
                if len(stage_blocks) != len(stage_channels):
                    raise RuntimeError(
                        "EviWeave v1 requires exactly context/read/core stages"
                    )
                stage_losses = []
                stage_peaks = []
                for block_index, channel in zip(stage_blocks, stage_channels):
                    trace_probability = torch.as_tensor(
                        trace_maps[channel][label_index],
                        device=device,
                        dtype=torch.float32,
                    )
                    trace_mass = float(
                        trace_probability.sum().detach().cpu()
                    )
                    if channel == "context":
                        trace_global_supervision_active = trace_mass > 0
                    if (
                        args.fusion == "eviweave_normalized"
                        and trace_mass <= 0
                    ):
                        # The normalized v22 lane never fabricates a uniform
                        # target from a missing human stage. QA and all other
                        # valid trace stages remain active for this sample.
                        trace_supervision_by_block[
                            f"{block_index}_skipped_empty"
                        ] = 1.0
                        continue
                    trace_probability = trace_probability / (
                        trace_probability.sum().clamp_min(1e-12)
                    )
                    trace_target = sample_trace_distribution(
                        trace_probability, gate_centers
                    )
                    stage_loss = soft_distribution_cross_entropy(
                        gate_logits_by_block[block_index], trace_target
                    )
                    stage_losses.append(stage_loss)
                    stage_peaks.append(float(trace_target.max().detach().cpu()))
                    trace_supervision_by_block[str(block_index)] = float(
                        stage_loss.detach().cpu()
                    )
                if not stage_losses:
                    raise RuntimeError(
                        "EviWeave sample has no valid trace stage"
                    )
                trace_supervision_loss = torch.stack(stage_losses).mean()
                trace_target_peak = sum(stage_peaks) / len(stage_peaks)
        evislot_diversity_loss = qa_loss.new_zeros(())
        evislot_parent_consistency_loss = qa_loss.new_zeros(())
        if args.fusion == "evislot":
            slot_losses = vision.slot_auxiliary_losses
            trace_supervision_loss = slot_losses.trace_support
            evislot_diversity_loss = slot_losses.slot_diversity
            evislot_parent_consistency_loss = (
                slot_losses.parent_consistency
            )
        safe_joint_identity_loss = qa_loss.new_zeros(())
        if safe_joint_residual_mixer is not None:
            if safe_joint_residual_mixer.last_injected_relative_l2 is None:
                raise RuntimeError("EviBlend injected residual is missing")
            safe_joint_identity_loss = (
                safe_joint_residual_mixer.last_injected_relative_l2
            )
        trace_calibration_identity_loss = qa_loss.new_zeros(())
        if trace_calibrated_bridge_residual is not None:
            if (
                trace_calibrated_bridge_residual
                .last_calibration_relative_l2
                is None
            ):
                raise RuntimeError(
                    "trace-calibrated residual metric is missing"
                )
            trace_calibration_identity_loss = (
                trace_calibrated_bridge_residual
                .last_calibration_relative_l2
            )
        learned_identity_loss = (
            safe_joint_identity_loss
            if safe_joint_residual_mixer is not None
            else (
                trace_calibration_identity_loss
                if trace_calibrated_bridge_residual is not None
                else vision.relative_residual_l2
            )
        )
        total_loss = (
            qa_loss
            + args.identity_weight * learned_identity_loss
            + args.trace_supervision_weight * trace_supervision_loss
            + args.evislot_diversity_weight * evislot_diversity_loss
            + args.evislot_parent_consistency_weight
            * evislot_parent_consistency_loss
        )
        if ptea_loss_result is not None:
            total_loss = total_loss + ptea_loss_result.total
        if h_safe_loss_result is not None:
            assert h_safe_supervised_loss is not None
            total_loss = total_loss + h_safe_supervised_loss
        qa_bridge_grad_norm = None
        ptea_aux_grad_norm = None
        ptea_aux_grad_none_tensors = None
        ptea_aux_grad_nonzero_tensors = None
        h_safe_aux_grad_norm = None
        h_safe_aux_grad_none_tensors = None
        h_safe_aux_grad_nonzero_tensors = None
        if (
            ptea_loss_result is not None
            and args.recovery_engineering_smoke
        ):
            ptea_gradients = torch.autograd.grad(
                ptea_loss_result.total,
                recovery_ptea_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            squared_ptea_norm = ptea_loss_result.total.new_zeros(
                (), dtype=torch.float32
            )
            ptea_aux_grad_none_tensors = 0
            ptea_aux_grad_nonzero_tensors = 0
            for gradient in ptea_gradients:
                if gradient is None:
                    ptea_aux_grad_none_tensors += 1
                    continue
                if not torch.isfinite(gradient).all():
                    raise RuntimeError("PTEA auxiliary gradient is non-finite")
                if bool(torch.count_nonzero(gradient)):
                    ptea_aux_grad_nonzero_tensors += 1
                squared_ptea_norm = (
                    squared_ptea_norm + gradient.float().pow(2).sum()
                )
            ptea_aux_grad_norm = float(
                squared_ptea_norm.sqrt().detach().cpu()
            )
            if not np.isfinite(ptea_aux_grad_norm) or ptea_aux_grad_norm <= 0:
                raise RuntimeError("PTEA auxiliary gradient gate failed")
        if (
            h_safe_loss_result is not None
            and h_safe_geometry_supervision_active
            and args.recovery_engineering_smoke
        ):
            h_safe_gradients = torch.autograd.grad(
                h_safe_supervised_loss,
                recovery_h_safe_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            squared_h_safe_norm = h_safe_supervised_loss.new_zeros(
                (), dtype=torch.float32
            )
            h_safe_aux_grad_none_tensors = 0
            h_safe_aux_grad_nonzero_tensors = 0
            for gradient in h_safe_gradients:
                if gradient is None:
                    h_safe_aux_grad_none_tensors += 1
                    continue
                if not torch.isfinite(gradient).all():
                    raise RuntimeError("H-Safe geometry gradient is non-finite")
                if bool(torch.count_nonzero(gradient)):
                    h_safe_aux_grad_nonzero_tensors += 1
                squared_h_safe_norm = (
                    squared_h_safe_norm + gradient.float().pow(2).sum()
                )
            h_safe_aux_grad_norm = float(
                squared_h_safe_norm.sqrt().detach().cpu()
            )
            if (
                not np.isfinite(h_safe_aux_grad_norm)
                or h_safe_aux_grad_norm <= 0
                or h_safe_aux_grad_nonzero_tensors <= 0
            ):
                raise RuntimeError("H-Safe geometry gradient gate failed")
        if (
            recovery is not None
            and recovery.train_bridge
            and args.recovery_engineering_smoke
        ):
            qa_bridge_gradients = torch.autograd.grad(
                qa_loss,
                recovery_bridge_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            squared_norm = qa_loss.new_zeros((), dtype=torch.float32)
            for gradient in qa_bridge_gradients:
                if gradient is not None:
                    squared_norm = squared_norm + gradient.float().pow(2).sum()
            qa_bridge_grad_norm = float(squared_norm.sqrt().detach().cpu())
        if args.recovery_objective != "grpo":
            (total_loss / args.gradient_accumulation).backward()
        optimizer_step = step % args.gradient_accumulation == 0
        if dynamic_grpo:
            optimizer_step=accepted_group and effective_groups % args.gradient_accumulation == 0
        grad_norm = None
        if optimizer_step:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters_for_clip, args.max_grad_norm
                ).detach()
            )
            if dynamic_grpo and not np.isfinite(grad_norm):
                raise RuntimeError('nonfinite effective RL gradient; optimizer not stepped')
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if dynamic_grpo:effective_optimizer_steps+=1

        metric = {
        "format_version": (
                {
                    "sparse_bridge": "evivit_v3_bridge_train_v1",
                    "tracefovea": "tracefovea_fixed5k_train_v1",
                    "tracefovea_human_gate": (
                        "tracefovea_trace_residual_fixed5k_train_v1"
                        if args.trace_residual_modulation
                        else "tracefovea_human_gate_fixed5k_train_v1"
                    ),
                    "eviweave_human_gate": (
                        "eviweave_human_gate_fixed5k_train_v1"
                    ),
                    "eviweave_normalized": (
                        "eviweave_normalized_fp32_fixed5k_train_v2"
                    ),
                    "evirelay": "evirelay_fixed5k_train_v1",
                    "eviblend_safe_joint": (
                        "eviblend_safe_joint_fixed5k_train_v1"
                    ),
                    "evislot": "evislot_midvit_fixedslots_train_v1",
                    "trace_calibrated_bridge": (
                        (
                            (
                                "trace_calibrated_bridge_centered_"
                                "fp32importance_fixed5k_train_v3"
                            )
                            if args.trace_calibration_preserve_fp32_importance
                            else (
                                "trace_calibrated_bridge_centered_"
                                "fixed5k_train_v2"
                            )
                            if args.trace_calibration_amplitude_mode
                            == "positive_centered"
                            else "trace_calibrated_bridge_fixed5k_train_v1"
                        )
                    ),
                }[args.fusion]
            ),
            "step": step,
            "epoch": epoch + 1,
            "id": row["id"],
            "loss": float(total_loss.detach().cpu()),
            "qa_loss": float(qa_loss.detach().cpu()),
            "trace_supervision_loss": float(
                trace_supervision_loss.detach().cpu()
            ),
            "evislot_diversity_loss": float(
                evislot_diversity_loss.detach().cpu()
            ),
            "evislot_parent_consistency_loss": float(
                evislot_parent_consistency_loss.detach().cpu()
            ),
            "trace_target_peak": trace_target_peak,
            "trace_global_supervision_active": (
                trace_global_supervision_active
            ),
            "trace_supervision_by_block": trace_supervision_by_block,
            "answer_token_accuracy": float(token_accuracy.detach().cpu()),
            "answer_tokens": answer_tokens,
            "recovery_objective": args.recovery_objective,
            "grpo": grpo_metrics,
            "relative_residual_l2": float(vision.relative_residual_l2.detach().cpu()),
            "safe_joint_injected_relative_l2": float(
                safe_joint_identity_loss.detach().cpu()
            ),
            "trace_calibration_relative_l2": float(
                trace_calibration_identity_loss.detach().cpu()
            ),
            "global_relative_residual_l2": float(
                vision.global_relative_residual_l2.detach().cpu()
            ),
            "fine_relative_residual_l2": float(
                vision.fine_relative_residual_l2.detach().cpu()
            ),
            "optimizer_step": optimizer_step,
            "optimizer_step_index": (
                (effective_optimizer_steps if dynamic_grpo else step // args.gradient_accumulation) if optimizer_step else None
            ),
            "stage_optimizer_step_index": (
                (effective_optimizer_steps if dynamic_grpo else (step - recovery_stage_start_micro_step)
                // args.gradient_accumulation)
                if optimizer_step
                else None
            ),
            "grad_norm": grad_norm,
            "qa_to_bridge_grad_norm": qa_bridge_grad_norm,
            "learning_rates": [
                float(group["lr"]) for group in optimizer.param_groups
            ],
            "map_entropy": allocation.map_entropy,
            "map_peak_probability": allocation.map_peak_probability,
            "regions": len(fine_views),
            # Audit-only serialization of the final discrete router plan.  This
            # is deliberately recorded after optional H-Safe refinement and
            # does not participate in allocation, cropping, or any loss.
            "region_roles": [
                str(budget.role) for budget in allocation.region_budgets
            ],
            "region_boxes_normalized": [
                [float(value) / 1000.0 for value in budget.bbox]
                for budget in allocation.region_budgets
            ],
            "region_allocated_token_budgets": [
                int(budget.token_budget)
                for budget in allocation.region_budgets
            ],
            "region_allocated_token_total": sum(
                int(budget.token_budget)
                for budget in allocation.region_budgets
            ),
            "planned_global_tokens": sample_global_token_budget,
            "planned_fine_tokens": sample_fine_token_budget,
            "evitree_v2": args.evitree_v2,
            "evitree_evidence_need": sample_evidence_need,
            "native_ratio_budget": (
                native_ratio_plan.to_dict()
                if native_ratio_plan is not None
                else None
            ),
            "native_need_spread_budget": (
                native_need_plan.to_dict() if native_need_plan is not None else None
            ),
            "realized_fine_tokens": realized_fine_tokens,
            "global_visual_tokens": vision.global_image_tokens,
            "fine_visual_tokens": vision.fine_image_tokens,
            "fine_encoder_tokens": vision.fine_encoder_tokens,
            "total_visual_tokens": int(vision.image_embeds.shape[0]),
            "native_global_used": native_global_used,
            "native_global_tokens": native_global_tokens,
            "native_pixel_contract": args.native_pixel_contract,
            "continuous_native_global": args.continuous_native_global,
            "continuous_global_plan": (
                continuous_global_plan.to_dict()
                if continuous_global_plan is not None
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
            "seeded_basin_config": seeded_basin_config,
            "seeded_basin_config_path": (
                str(seeded_basin_config_path)
                if seeded_basin_config_path is not None
                else None
            ),
            "evicontour_utility_checkpoint": (
                str(evicontour_utility_path)
                if evicontour_utility_path is not None
                else None
            ),
            "evicontour_utility_format_version": (
                evicontour_utility_checkpoint.get("format_version")
                if evicontour_utility_checkpoint is not None
                else None
            ),
            "evicontour_csr_fine_floor_plan": (
                continuous_fine_floor_plan.to_dict()
                if args.evidence_decoder == "evicontour_csr"
                and continuous_fine_floor_plan is not None
                else None
            ),
            "native_token_caps": args.native_token_caps,
            "native_cap_minimum_fine_tokens": (
                args.native_cap_minimum_fine_tokens
                if args.native_token_caps
                else None
            ),
            "native_cap_fine_floor_native_ratio": (
                args.native_cap_fine_floor_native_ratio
                if args.native_token_caps
                else None
            ),
            "native_cap_continuous_fine_floor_plan": (
                continuous_fine_floor_plan.to_dict()
                if continuous_fine_floor_plan is not None
                else None
            ),
            "native_cap_global_plan": (
                native_cap_global_plan.to_dict()
                if native_cap_global_plan is not None
                else None
            ),
            "native_cap_fine_plans": (
                native_cap_fine_plans if args.native_token_caps else None
            ),
            "visual_output_mode": vision.visual_output_mode,
            "native_global_mrope": args.native_global_mrope,
            "bridge_mode": args.bridge_mode,
            "fusion": args.fusion,
            "answer_protocol": args.answer_protocol,
            "recovery_lane": recovery.name if recovery is not None else None,
            "covered_global_tokens": (
                None
                if args.fusion == "evislot"
                else vision.bridge_diagnostics.covered_global_tokens
            ),
            "peak_gpu_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
            "elapsed_sec": time.time() - started,
        }
        if ptea_loss_result is not None:
            metric.update(
                {
                    "ptea_aux_grad_norm": ptea_aux_grad_norm,
                    "ptea_aux_grad_none_tensors": ptea_aux_grad_none_tensors,
                    "ptea_aux_grad_nonzero_tensors": (
                        ptea_aux_grad_nonzero_tensors
                    ),
                    "ptea_loss": float(ptea_loss_result.total.detach().cpu()),
                    "ptea_decisive_map_loss": float(
                        ptea_loss_result.decisive_map.detach().cpu()
                    ),
                    "ptea_decisive_teacher_kl": float(
                        ptea_loss_result.decisive_teacher_kl.detach().cpu()
                    ),
                    "ptea_context_teacher_kl": float(
                        ptea_loss_result.context_teacher_kl.detach().cpu()
                    ),
                }
            )
        if h_safe_loss_result is not None:
            assert h_safe_supervised_loss is not None
            assert h_safe_geometry_supervision_active is not None
            metric.update(
                {
                    "h_safe_geometry_loss_version": (
                        H_SAFE_GEOMETRY_LOSS_VERSION
                    ),
                    "h_safe_aux_grad_norm": h_safe_aux_grad_norm,
                    "h_safe_aux_grad_none_tensors": h_safe_aux_grad_none_tensors,
                    "h_safe_aux_grad_nonzero_tensors": (
                        h_safe_aux_grad_nonzero_tensors
                    ),
                    "h_safe_geometry_supervision_active": (
                        h_safe_geometry_supervision_active
                    ),
                    "h_safe_loss": float(h_safe_supervised_loss.detach().cpu()),
                    "h_safe_unmasked_diagnostic_loss": float(
                        h_safe_loss_result.total.detach().cpu()
                    ),
                    "h_safe_regression_loss": float(
                        h_safe_loss_result.regression.detach().cpu()
                    ),
                    "h_safe_giou_loss": float(
                        h_safe_loss_result.giou.detach().cpu()
                    ),
                    "h_safe_coverage_loss": float(
                        h_safe_loss_result.coverage.detach().cpu()
                    ),
                    "h_safe_matched_coverage_loss": float(
                        h_safe_loss_result.matched_coverage.detach().cpu()
                    ),
                    "h_safe_union_coverage_loss": float(
                        h_safe_loss_result.union_coverage.detach().cpu()
                    ),
                    "h_safe_excess_area_loss": float(
                        h_safe_loss_result.excess_area.detach().cpu()
                    ),
                    "h_safe_identity_loss": float(
                        h_safe_loss_result.identity.detach().cpu()
                    ),
                    "h_safe_teacher_drift_loss": float(
                        h_safe_loss_result.drift.detach().cpu()
                    ),
                    "h_safe_raw_target_min_area": float(
                        h_safe_loss_result.raw_target_min_area.detach().cpu()
                    ),
                    "h_safe_realizable_target_min_area": float(
                        h_safe_loss_result.realizable_target_min_area.detach().cpu()
                    ),
                    "h_safe_max_refined_to_reference_area_ratio": float(
                        h_safe_loss_result.maximum_refined_to_reference_area_ratio
                        .detach()
                        .cpu()
                    ),
                    "h_safe_unrealizable_target_fraction": float(
                        h_safe_loss_result.unrealizable_target_fraction.detach().cpu()
                    ),
                    "h_safe_minimum_side": float(
                        adaptive_box_head.config.minimum_side
                    ),
                    "h_safe_matched_anchors": int(
                        h_safe_loss_result.matching.matched_mask.sum().detach().cpu()
                    ),
                    "h_safe_supervised_decisive_anchors": int(
                        h_safe_loss_result.supervised_decisive_anchors
                    ),
                }
            )
        if isinstance(bridge, TraceFoveaBridgeAdapter):
            trace_diagnostics = bridge.last_trace_diagnostics
            if trace_diagnostics is None:
                raise RuntimeError("TraceFovea diagnostics are missing")
            metric["tracefovea"] = trace_diagnostics.as_dict()
        if isinstance(bridge, EviRelayBridgeAdapter):
            relay_diagnostics = bridge.last_relay_diagnostics
            if relay_diagnostics is None:
                raise RuntimeError("EviRelay diagnostics are missing")
            metric["evirelay"] = relay_diagnostics.as_dict()
        if getattr(vision, "safe_joint_residual_diagnostics", None):
            metric["eviblend"] = {
                str(block): diagnostics.as_dict()
                for block, diagnostics
                in vision.safe_joint_residual_diagnostics.items()
            }
        if getattr(vision, "trace_calibrated_bridge_diagnostics", None) is not None:
            metric["trace_calibrated_bridge"] = (
                vision.trace_calibrated_bridge_diagnostics.as_dict()
            )
        if args.fusion == "evislot":
            metric["evislot"] = vision.slot_diagnostics.as_dict()
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric, ensure_ascii=False) + "\n")
        print(json.dumps(metric, ensure_ascii=False), flush=True)

        checkpoint_due = (
            step % args.save_steps == 0
            or step == args.max_steps
            or step == args.recovery_stop_after_step
        )
        if dynamic_grpo:
            checkpoint_due=(optimizer_step and (effective_optimizer_steps % args.save_effective_updates==0
                            or effective_optimizer_steps>=args.effective_updates)) or step==args.max_steps
        if checkpoint_due:
            if recovery is not None:
                assert scheduler is not None and recovery_fingerprint is not None
                save_recovery_checkpoint(
                    args.output_dir / (f'checkpoint-update-{effective_optimizer_steps:05d}' if dynamic_grpo else f"checkpoint_step_{step:05d}"),
                    bridge=bridge,
                    policy_model=policy_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    micro_step=step,
                    optimizer_step=effective_optimizer_steps if dynamic_grpo else step // args.gradient_accumulation,
                    protocol_fingerprint=recovery_fingerprint,
                    config=config,
                    selector=selector if recovery.train_ptea else None,
                    h_safe=adaptive_box_head if recovery.train_h_safe else None,
                    protocol=recovery_protocol,
                )
                save_recovery_checkpoint(
                    args.output_dir / "resume-latest",
                    bridge=bridge,
                    policy_model=policy_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    micro_step=step,
                    optimizer_step=effective_optimizer_steps if dynamic_grpo else step // args.gradient_accumulation,
                    protocol_fingerprint=recovery_fingerprint,
                    config=config,
                    selector=selector if recovery.train_ptea else None,
                    h_safe=adaptive_box_head if recovery.train_h_safe else None,
                    protocol=recovery_protocol,
                )
            else:
                save_checkpoint(
                    args.output_dir / f"checkpoint_step_{step:05d}.pt",
                    bridge,
                    optimizer,
                    step=step,
                    config=config,
                    safe_joint_residual_mixer=safe_joint_residual_mixer,
                    trace_calibrated_bridge_residual=(
                        trace_calibrated_bridge_residual
                    ),
                )
        closed: set[int] = set()
        for image in [*opened, global_view, source]:
            if id(image) not in closed:
                image.close()
                closed.add(id(image))
        del global_inputs, prepared, question_tokens, allocation, fine_views
        del vision, qa_loss, total_loss
        if dynamic_grpo and effective_optimizer_steps>=args.effective_updates:
            break
        if ptea_loss_result is not None:
            del ptea_loss_result, student_paths, teacher_paths, decisive_target
        if h_safe_loss_result is not None:
            del h_safe_loss_result, h_safe_targets, frozen_h_safe_residuals
            del h_safe_supervised_loss
        if args.recovery_stop_after_step and step == args.recovery_stop_after_step:
            stopped = {
                "status": "intentionally_stopped_for_recovery_resume_smoke",
                "recovery_lane": recovery.name if recovery is not None else None,
                "micro_step": step,
                "optimizer_step": step // args.gradient_accumulation,
                "resume_checkpoint": str(args.output_dir / "resume-latest"),
            }
            (args.output_dir / "smoke_stopped.json").write_text(
                json.dumps(stopped, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(stopped, ensure_ascii=False), flush=True)
            return 0

    metrics = read_jsonl(log_path)
    window = min(16, max(1, len(metrics) // 4))
    first = metrics[:window]
    last = metrics[-window:]
    mean_first_loss = sum(float(row["qa_loss"]) for row in first) / len(first)
    mean_last_loss = sum(float(row["qa_loss"]) for row in last) / len(last)
    mean_first_accuracy = sum(
        float(row["answer_token_accuracy"]) for row in first
    ) / len(first)
    mean_last_accuracy = sum(
        float(row["answer_token_accuracy"]) for row in last
    ) / len(last)
    summary = {
        "version": (
            {
                "sparse_bridge": "evivit_v3_bridge_train_summary_v1",
                "tracefovea": "tracefovea_fixed5k_train_summary_v1",
                "tracefovea_human_gate": (
                    "tracefovea_trace_residual_fixed5k_train_summary_v1"
                    if args.trace_residual_modulation
                    else "tracefovea_human_gate_fixed5k_train_summary_v1"
                ),
                "eviweave_human_gate": (
                    "eviweave_human_gate_fixed5k_train_summary_v1"
                ),
                "eviweave_normalized": (
                    "eviweave_normalized_fp32_fixed5k_train_summary_v2"
                ),
                "evirelay": "evirelay_fixed5k_train_summary_v1",
                "eviblend_safe_joint": (
                    "eviblend_safe_joint_fixed5k_train_summary_v1"
                ),
                "evislot": "evislot_midvit_fixedslots_train_summary_v1",
                "trace_calibrated_bridge": (
                    (
                        (
                            "trace_calibrated_bridge_centered_"
                            "fp32importance_fixed5k_train_summary_v3"
                        )
                        if args.trace_calibration_preserve_fp32_importance
                        else (
                            "trace_calibrated_bridge_centered_"
                            "fixed5k_train_summary_v2"
                        )
                        if args.trace_calibration_amplitude_mode
                        == "positive_centered"
                        else "trace_calibrated_bridge_fixed5k_train_summary_v1"
                    )
                ),
            }[args.fusion]
        ),
        "fusion": args.fusion,
        "recovery_lane": recovery.name if recovery is not None else None,
        "recovery_stage": args.recovery_stage if recovery is not None else None,
        "stage_start_micro_step": recovery_stage_start_micro_step,
        "parent_protocol_fingerprint": parent_protocol_fingerprint,
        "steps": step,
        "effective_optimizer_steps": effective_optimizer_steps if dynamic_grpo else None,
        "skipped_groups": skipped_groups if dynamic_grpo else None,
        "pending_groups_discarded_at_cap": effective_groups % args.gradient_accumulation if dynamic_grpo else None,
        "status": ('complete' if effective_optimizer_steps>=args.effective_updates else 'sampled_cap_reached') if dynamic_grpo else 'complete',
        "rows": len(rows),
        "trainable_parameters": trainable,
        "mean_first_window_qa_loss": mean_first_loss,
        "mean_last_window_qa_loss": mean_last_loss,
        "loss_reduction": 1.0 - mean_last_loss / max(mean_first_loss, 1e-12),
        "mean_first_window_answer_token_accuracy": mean_first_accuracy,
        "mean_last_window_answer_token_accuracy": mean_last_accuracy,
        "answer_token_accuracy_gain": mean_last_accuracy - mean_first_accuracy,
        "overfit_loss_gate_pass": mean_last_loss <= 0.8 * mean_first_loss,
        "overfit_accuracy_gate_pass": mean_last_accuracy >= mean_first_accuracy + 0.10,
        "peak_gpu_mib": max(float(row["peak_gpu_mib"]) for row in metrics),
        "elapsed_sec": time.time() - started,
    }
    if recovery is not None and recovery.requires_router_aux:
        first_ptea = [float(row["ptea_loss"]) for row in first]
        last_ptea = [float(row["ptea_loss"]) for row in last]
        gradient_values = [
            float(row["ptea_aux_grad_norm"])
            for row in metrics
            if row.get("ptea_aux_grad_norm") is not None
        ]
        summary["router_stage"] = args.recovery_stage
        summary["mean_first_window_ptea_loss"] = sum(first_ptea) / len(first_ptea)
        summary["mean_last_window_ptea_loss"] = sum(last_ptea) / len(last_ptea)
        summary["ptea_gradient_gate_pass"] = (
            bool(gradient_values)
            and all(np.isfinite(value) and value > 0 for value in gradient_values)
            if args.recovery_engineering_smoke
            else None
        )
        summary["ptea_teacher_frozen"] = True
        summary["discrete_decoder_detached"] = True
    if recovery is not None and recovery.train_h_safe:
        first_h_safe = [float(row["h_safe_loss"]) for row in first]
        last_h_safe = [float(row["h_safe_loss"]) for row in last]
        h_safe_gradient_values = [
            float(row["h_safe_aux_grad_norm"])
            for row in metrics
            if row.get("h_safe_aux_grad_norm") is not None
        ]
        summary["mean_first_window_h_safe_loss"] = sum(first_h_safe) / len(
            first_h_safe
        )
        summary["mean_last_window_h_safe_loss"] = sum(last_h_safe) / len(
            last_h_safe
        )
        summary["h_safe_gradient_gate_pass"] = (
            bool(h_safe_gradient_values)
            and all(
                np.isfinite(value) and value > 0
                for value in h_safe_gradient_values
            )
            if args.recovery_engineering_smoke
            else None
        )
        summary["h_safe_teacher_frozen"] = True
        summary["ptea_to_h_safe_detached"] = True
        summary["answer_ce_to_h_safe_disabled"] = True
    if dynamic_grpo:
        summary['policy_reference_weight_delta']=compare_policy_reference_weights(policy_model)
        if effective_optimizer_steps>0 and summary['policy_reference_weight_delta']['bitwise_equal']:
            raise RuntimeError('effective updates did not change policy weights')
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
