#!/usr/bin/env python3
"""Stage-1 EviViT control: one prompt image span, global + foveal tokens.

This is intentionally not the final trainable EviViT.  It reuses frozen
EviMap Portfolio regions to test the central interface hypothesis: can Qwen
retain the evidence benefit when global and high-resolution region features
are fused into one coordinate-aware visual sequence, without presenting eight
independent crop images to the language model?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_qwen_original_qa import (  # noqa: E402
    clean_question,
    normalized_answer,
    parse_answer,
    read_jsonl,
    relaxed_answer,
    relaxed_answer_match,
    summarize,
)
from evivit_core.evivit import (  # noqa: E402
    allocate_region_budgets,
    grid_for_token_budget,
    multiresolution_token_metadata,
    multiresolution_fusion_indices,
    quantize_mrope_coordinates,
    select_region_candidates,
    selector_question_from_prompt,
)
from evivit_core.geometry import policy_to_pixels  # noqa: E402
from evivit_core.gpu_safety import check_gpu  # noqa: E402
from evivit_core.latency import summarize_latency  # noqa: E402
from evivit_core.qwen_family import language_model_visual_forward  # noqa: E402


SYSTEM_PROMPTS = {
    "canonical": (
        "You are a careful fine-grained visual question answering assistant. "
        "Answer using only the image evidence. Return compact JSON only."
    ),
    "adaptive": (
        "You are a careful fine-grained visual question answering assistant. "
        "The single visual input is encoded with global context and internally allocated "
        "high-resolution evidence tokens from the same source image. Cross-check local detail "
        "with global identity, layout, and spatial relations. Return compact JSON only."
    ),
}


def resize_to_token_grid(
    image: Image.Image,
    grid_h: int,
    grid_w: int,
    *,
    merged_patch_stride: int = 32,
    downsample_resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    """Resize an internal view so Qwen emits the requested merged-token grid."""

    size = (int(grid_w) * merged_patch_stride, int(grid_h) * merged_patch_stride)
    if image.size == size:
        return image
    downsample = size[0] < image.width or size[1] < image.height
    return image.resize(
        size,
        downsample_resample if downsample else Image.Resampling.BICUBIC,
    )


def build_single_span_prompt(processor: Any, question: str, *, prompt_variant: str) -> str:
    if prompt_variant not in SYSTEM_PROMPTS:
        raise ValueError(f"unknown prompt variant: {prompt_variant}")
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPTS[prompt_variant]}],
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
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def expand_single_image_token(
    token_ids: Sequence[int], image_token_id: int, visual_tokens: int
) -> tuple[list[int], int, int]:
    positions = [index for index, value in enumerate(token_ids) if value == image_token_id]
    if len(positions) != 1:
        raise RuntimeError(f"expected one image placeholder, found {len(positions)}")
    if visual_tokens <= 0:
        raise ValueError("visual_tokens must be positive")
    start = positions[0]
    expanded = list(token_ids[:start]) + [image_token_id] * visual_tokens + list(token_ids[start + 1 :])
    return expanded, start, start + visual_tokens


def build_position_ids(
    sequence_length: int,
    visual_start: int,
    visual_end: int,
    visual_coordinates: np.ndarray,
    *,
    device: str,
) -> tuple[torch.Tensor, int]:
    """Construct Qwen temporal/height/width positions for one multiresolution span."""

    visual_tokens = visual_end - visual_start
    if visual_coordinates.shape != (3, visual_tokens):
        raise ValueError(
            f"coordinate shape {visual_coordinates.shape} does not match {visual_tokens} tokens"
        )
    positions = torch.zeros(3, sequence_length, dtype=torch.long, device=device)
    if visual_start:
        prefix = torch.arange(visual_start, device=device)
        positions[:, :visual_start] = prefix
    coords = torch.from_numpy(visual_coordinates).to(device=device, dtype=torch.long)
    positions[:, visual_start:visual_end] = coords + visual_start
    suffix_base = int(positions[:, visual_start:visual_end].max().item()) + 1
    suffix_length = sequence_length - visual_end
    if suffix_length:
        suffix = torch.arange(suffix_base, suffix_base + suffix_length, device=device)
        positions[:, visual_end:] = suffix
    return positions.unsqueeze(1), suffix_base + suffix_length


@torch.inference_mode()
def predict_online_ptea_candidates(
    model: Any,
    processor: Any,
    selector: Any,
    projection: torch.Tensor,
    global_embeds: torch.Tensor,
    global_grid_shape: Sequence[int],
    question: str,
    *,
    device: str,
    top_k: int = 8,
    modes: int = 3,
    context_factor: float = 1.8,
    residual_suppression: float = 0.05,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Predict Portfolio candidates directly from the reused global ViT pass.

    This reproduces the frozen PTEA-A preprocessing online: the same fixed
    random projection is applied to merged Qwen visual embeddings and raw
    question token embeddings.  No precomputed feature or box file is used.
    """

    from scripts.build_evimap_portfolio_boxes import decode_portfolio

    grid_h, grid_w = (int(global_grid_shape[0]), int(global_grid_shape[1]))
    if int(global_embeds.shape[0]) != grid_h * grid_w:
        raise RuntimeError(
            "global embedding/grid mismatch: "
            f"{tuple(global_embeds.shape)} vs {grid_h}x{grid_w}"
        )
    # PTEA was trained and evaluated from cached float16 projected features.
    # Preserve that numerical protocol online; otherwise tiny float32 changes
    # can be amplified by residual suppression when decoding later modes.
    visual = F.normalize(global_embeds.float() @ projection, dim=-1).to(
        torch.float16
    ).reshape(grid_h, grid_w, -1)
    encoded = processor.tokenizer(
        clean_question(question),
        add_special_tokens=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    token_ids = encoded.input_ids.to(device)
    question_embeds = model.get_input_embeddings()(token_ids).squeeze(0)
    question_tokens = F.normalize(question_embeds.float() @ projection, dim=-1).to(
        torch.float16
    )
    question_mask = torch.ones(
        question_tokens.shape[0], dtype=torch.bool, device=device
    )
    logits = selector(visual, question_tokens, question_mask)[0]
    probability = torch.softmax(logits.flatten(), dim=0).reshape_as(logits)
    candidates = decode_portfolio(
        probability.float().cpu().numpy(),
        top_k=top_k,
        modes=modes,
        context_factor=context_factor,
        residual_suppression=residual_suppression,
    )
    distribution = probability.flatten().float()
    entropy = float(
        (-(distribution.clamp_min(1e-12) * distribution.clamp_min(1e-12).log()).sum())
        .detach()
        .cpu()
    )
    return candidates, {
        "map_entropy": entropy,
        "map_peak_probability": float(probability.max().detach().cpu()),
        "map_tokens": float(probability.numel()),
    }


@torch.inference_mode()
def encode_internal_views_sequentially(
    model: Any,
    processor: Any,
    internal_views: Sequence[Image.Image],
    *,
    device: str,
    merge_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor], list[list[int]]]:
    """Encode internal views one at a time and fuse after the frozen ViT.

    Qwen vision attention is applied before spatial merging.  Batching a 1K
    global grid and eight fine grids can therefore create a large transient
    activation peak even though the final fused sequence is modest.  Sequential
    shared-weight encoding preserves exactly the same frozen features while
    bounding peak memory by the largest single internal view.
    """

    final_chunks: list[torch.Tensor] = []
    deepstack_chunks: list[list[torch.Tensor]] = []
    grid_shapes: list[list[int]] = []
    for view in internal_views:
        vision_inputs = processor(
            text=["."], images=[view], return_tensors="pt", padding=True
        ).to(device)
        grid_thw = vision_inputs["image_grid_thw"]
        image_embeds, deepstack = model.model.get_image_features(
            vision_inputs["pixel_values"], grid_thw
        )
        if len(image_embeds) != 1:
            raise RuntimeError(f"expected one encoded internal view, got {len(image_embeds)}")
        final_chunks.append(image_embeds[0])
        if not deepstack_chunks:
            deepstack_chunks = [[] for _ in deepstack]
        if len(deepstack) != len(deepstack_chunks):
            raise RuntimeError("DeepStack layer count changed across internal views")
        for layer_index, feature in enumerate(deepstack):
            deepstack_chunks[layer_index].append(feature)
        row = grid_thw[0].detach().cpu().tolist()
        grid_shapes.append([int(row[1]) // merge_size, int(row[2]) // merge_size])
        del vision_inputs, image_embeds, deepstack
    return (
        torch.cat(final_chunks, dim=0),
        [torch.cat(chunks, dim=0) for chunks in deepstack_chunks],
        grid_shapes,
    )


@torch.inference_mode()
def greedy_generate_from_visual_span(
    model: Any,
    processor: Any,
    prompt_text: str,
    image_embeds: torch.Tensor,
    deepstack_embeds: list[torch.Tensor],
    visual_coordinates: np.ndarray,
    *,
    device: str,
    max_new_tokens: int,
    record_timing: bool = False,
    record_confidence: bool = False,
) -> str | tuple[str, dict[str, float | int]]:
    tokenizer = processor.tokenizer
    raw_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    image_token_id = int(model.config.image_token_id)
    expanded_ids, visual_start, visual_end = expand_single_image_token(
        raw_ids, image_token_id, int(image_embeds.shape[0])
    )
    input_ids = torch.tensor([expanded_ids], dtype=torch.long, device=device)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    visual_mask = input_ids.eq(image_token_id)
    inputs_embeds = inputs_embeds.masked_scatter(
        visual_mask.unsqueeze(-1), image_embeds.to(inputs_embeds.dtype)
    )
    position_ids, next_text_position = build_position_ids(
        input_ids.shape[1],
        visual_start,
        visual_end,
        visual_coordinates,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    cache_position = torch.arange(input_ids.shape[1], device=device)
    if record_timing:
        torch.cuda.synchronize()
    generation_started = time.perf_counter()
    output = language_model_visual_forward(
        model,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_mask=visual_mask,
        deepstack_visual_embeds=[
            row.to(inputs_embeds.dtype) for row in deepstack_embeds
        ],
        cache_position=cache_position,
        use_cache=True,
    )
    past = output.past_key_values
    next_logits = model.lm_head(output.last_hidden_state[:, -1])
    next_token = next_logits.argmax(dim=-1, keepdim=True)
    token_logprobs = [
        float(
            next_logits.float()
            .log_softmax(dim=-1)
            .gather(-1, next_token)
            .item()
        )
    ]
    if record_timing:
        torch.cuda.synchronize()
    first_token_time = time.perf_counter()
    generated = [int(next_token.item())]
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos = {eos_ids}
    else:
        eos = {int(value) for value in (eos_ids or [])}
    for step in range(1, max_new_tokens):
        if generated[-1] in eos:
            break
        token_embed = model.get_input_embeddings()(next_token)
        full_length = input_ids.shape[1] + step
        token_position = torch.full(
            (3, 1, 1), next_text_position + step - 1, dtype=torch.long, device=device
        )
        output = model.model.language_model(
            inputs_embeds=token_embed,
            attention_mask=torch.ones(1, full_length, dtype=torch.long, device=device),
            position_ids=token_position,
            past_key_values=past,
            cache_position=torch.tensor([full_length - 1], device=device),
            use_cache=True,
        )
        past = output.past_key_values
        next_logits = model.lm_head(output.last_hidden_state[:, -1])
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        token_logprobs.append(
            float(
                next_logits.float()
                .log_softmax(dim=-1)
                .gather(-1, next_token)
                .item()
            )
        )
        generated.append(int(next_token.item()))
    if record_timing:
        torch.cuda.synchronize()
    generation_finished = time.perf_counter()
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    if not record_timing and not record_confidence:
        return decoded
    metadata: dict[str, float | int] = {"generated_tokens": len(generated)}
    if record_timing:
        metadata.update(
            {
                "_first_token_wall_time": first_token_time,
                "_generation_finished_wall_time": generation_finished,
                "llm_prefill_to_first_token_seconds": first_token_time
                - generation_started,
                "generation_seconds": generation_finished - generation_started,
                "decode_seconds": generation_finished - first_token_time,
            }
        )
    if record_confidence:
        scored_logprobs = [
            logprob
            for token, logprob in zip(generated, token_logprobs, strict=True)
            if token not in eos
        ]
        if not scored_logprobs:
            scored_logprobs = token_logprobs[:1]
        metadata.update(
            {
                "answer_mean_logprob": sum(scored_logprobs) / len(scored_logprobs),
                "answer_min_logprob": min(scored_logprobs),
                "answer_scored_tokens": len(scored_logprobs),
            }
        )
    return decoded, metadata


def expand_multiple_image_tokens(
    token_ids: Sequence[int],
    image_token_id: int,
    visual_token_counts: Sequence[int],
) -> tuple[list[int], list[tuple[int, int]]]:
    """Expand ordered image placeholders into independent visual spans."""

    positions = [index for index, value in enumerate(token_ids) if value == image_token_id]
    if len(positions) != len(visual_token_counts):
        raise RuntimeError(
            f"expected {len(visual_token_counts)} image placeholders, found {len(positions)}"
        )
    expanded: list[int] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for position, count in zip(positions, visual_token_counts, strict=True):
        if count <= 0:
            raise ValueError("every visual span must contain at least one token")
        expanded.extend(token_ids[cursor:position])
        start = len(expanded)
        expanded.extend([image_token_id] * int(count))
        spans.append((start, len(expanded)))
        cursor = position + 1
    expanded.extend(token_ids[cursor:])
    return expanded, spans


def build_multiple_span_position_ids(
    sequence_length: int,
    spans: Sequence[tuple[int, int]],
    visual_coordinates: Sequence[np.ndarray],
    *,
    device: str,
) -> tuple[torch.Tensor, int]:
    """Construct MRoPE positions while preserving each image's local frame."""

    if len(spans) != len(visual_coordinates):
        raise ValueError("visual span/coordinate count mismatch")
    positions = torch.zeros(3, sequence_length, dtype=torch.long, device=device)
    sequence_cursor = 0
    position_cursor = 0
    for (start, end), coordinates in zip(spans, visual_coordinates, strict=True):
        if start < sequence_cursor or end <= start:
            raise ValueError(f"invalid visual span {(start, end)}")
        text_length = start - sequence_cursor
        if text_length:
            text_positions = torch.arange(
                position_cursor,
                position_cursor + text_length,
                dtype=torch.long,
                device=device,
            )
            positions[:, sequence_cursor:start] = text_positions
            position_cursor += text_length
        visual_tokens = end - start
        if coordinates.shape != (3, visual_tokens):
            raise ValueError(
                f"coordinate shape {coordinates.shape} does not match {visual_tokens} tokens"
            )
        coordinate_tensor = torch.from_numpy(coordinates).to(
            device=device, dtype=torch.long
        )
        coordinate_tensor = coordinate_tensor - coordinate_tensor.amin(dim=1, keepdim=True)
        positions[:, start:end] = coordinate_tensor + position_cursor
        position_cursor = int(positions[:, start:end].max().item()) + 1
        sequence_cursor = end
    suffix_length = sequence_length - sequence_cursor
    if suffix_length:
        suffix = torch.arange(
            position_cursor,
            position_cursor + suffix_length,
            dtype=torch.long,
            device=device,
        )
        positions[:, sequence_cursor:] = suffix
        position_cursor += suffix_length
    return positions.unsqueeze(1), position_cursor


@torch.inference_mode()
def greedy_generate_from_visual_spans(
    model: Any,
    processor: Any,
    prompt_text: str,
    image_embeds_by_image: Sequence[torch.Tensor],
    deepstack_embeds_by_image: Sequence[list[torch.Tensor]],
    visual_coordinates_by_image: Sequence[np.ndarray],
    *,
    device: str,
    max_new_tokens: int,
    record_timing: bool = False,
    record_confidence: bool = False,
) -> str | tuple[str, dict[str, float | int]]:
    """Generate from multiple ordered EviViT spans in a single LLM context."""

    if not image_embeds_by_image:
        raise ValueError("at least one EviViT image span is required")
    if not (
        len(image_embeds_by_image)
        == len(deepstack_embeds_by_image)
        == len(visual_coordinates_by_image)
    ):
        raise ValueError("multi-image visual input count mismatch")
    tokenizer = processor.tokenizer
    raw_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    image_token_id = int(model.config.image_token_id)
    visual_counts = [int(row.shape[0]) for row in image_embeds_by_image]
    expanded_ids, spans = expand_multiple_image_tokens(
        raw_ids, image_token_id, visual_counts
    )
    input_ids = torch.tensor([expanded_ids], dtype=torch.long, device=device)
    image_embeds = torch.cat(list(image_embeds_by_image), dim=0)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    visual_mask = input_ids.eq(image_token_id)
    inputs_embeds = inputs_embeds.masked_scatter(
        visual_mask.unsqueeze(-1), image_embeds.to(inputs_embeds.dtype)
    )
    layer_count = len(deepstack_embeds_by_image[0])
    if any(len(rows) != layer_count for rows in deepstack_embeds_by_image):
        raise RuntimeError("DeepStack layer count changed across images")
    deepstack_embeds = [
        torch.cat(
            [rows[layer_index] for rows in deepstack_embeds_by_image], dim=0
        )
        for layer_index in range(layer_count)
    ]
    position_ids, next_text_position = build_multiple_span_position_ids(
        input_ids.shape[1], spans, visual_coordinates_by_image, device=device
    )
    attention_mask = torch.ones_like(input_ids)
    cache_position = torch.arange(input_ids.shape[1], device=device)
    if record_timing:
        torch.cuda.synchronize()
    generation_started = time.perf_counter()
    output = language_model_visual_forward(
        model,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_mask=visual_mask,
        deepstack_visual_embeds=[
            row.to(inputs_embeds.dtype) for row in deepstack_embeds
        ],
        cache_position=cache_position,
        use_cache=True,
    )
    past = output.past_key_values
    next_logits = model.lm_head(output.last_hidden_state[:, -1])
    next_token = next_logits.argmax(dim=-1, keepdim=True)
    token_logprobs = [
        float(
            next_logits.float()
            .log_softmax(dim=-1)
            .gather(-1, next_token)
            .item()
        )
    ]
    if record_timing:
        torch.cuda.synchronize()
    first_token_time = time.perf_counter()
    generated = [int(next_token.item())]
    eos_ids = model.generation_config.eos_token_id
    eos = {eos_ids} if isinstance(eos_ids, int) else {
        int(value) for value in (eos_ids or [])
    }
    for step in range(1, max_new_tokens):
        if generated[-1] in eos:
            break
        token_embed = model.get_input_embeddings()(next_token)
        full_length = input_ids.shape[1] + step
        token_position = torch.full(
            (3, 1, 1),
            next_text_position + step - 1,
            dtype=torch.long,
            device=device,
        )
        output = model.model.language_model(
            inputs_embeds=token_embed,
            attention_mask=torch.ones(
                1, full_length, dtype=torch.long, device=device
            ),
            position_ids=token_position,
            past_key_values=past,
            cache_position=torch.tensor([full_length - 1], device=device),
            use_cache=True,
        )
        past = output.past_key_values
        next_logits = model.lm_head(output.last_hidden_state[:, -1])
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        token_logprobs.append(
            float(
                next_logits.float()
                .log_softmax(dim=-1)
                .gather(-1, next_token)
                .item()
            )
        )
        generated.append(int(next_token.item()))
    if record_timing:
        torch.cuda.synchronize()
    generation_finished = time.perf_counter()
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    if not record_timing and not record_confidence:
        return decoded
    metadata: dict[str, float | int] = {
        "generated_tokens": len(generated),
        "image_count": len(image_embeds_by_image),
        "visual_tokens": sum(visual_counts),
    }
    if record_timing:
        metadata.update(
            {
                "_first_token_wall_time": first_token_time,
                "_generation_finished_wall_time": generation_finished,
                "llm_prefill_to_first_token_seconds": first_token_time
                - generation_started,
                "generation_seconds": generation_finished - generation_started,
                "decode_seconds": generation_finished - first_token_time,
            }
        )
    if record_confidence:
        scored_logprobs = [
            logprob
            for token, logprob in zip(generated, token_logprobs, strict=True)
            if token not in eos
        ] or token_logprobs[:1]
        metadata.update(
            {
                "answer_mean_logprob": sum(scored_logprobs) / len(scored_logprobs),
                "answer_min_logprob": min(scored_logprobs),
                "answer_scored_tokens": len(scored_logprobs),
            }
        )
    return decoded, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--box-predictions", type=Path)
    parser.add_argument(
        "--selector-checkpoint",
        type=Path,
        help="Enable online PTEA allocation instead of reading precomputed boxes.",
    )
    parser.add_argument("--selector-projection-dim", type=int, default=512)
    parser.add_argument("--selector-projection-seed", type=int, default=20260715)
    parser.add_argument("--selector-max-pixels", type=int, default=1048576)
    parser.add_argument(
        "--selector-question-protocol",
        choices=("full_prompt", "question_stem"),
        default="full_prompt",
        help="Whether online PTEA sees shuffled answer choices or only the question stem.",
    )
    parser.add_argument(
        "--selector-global-preprocess",
        choices=("qwen_native", "evivit_lanczos"),
        default="qwen_native",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--model", type=Path,
        default=Path("models/Qwen3-VL-4B-Instruct"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--memory-budget-mib", type=int, default=36000)
    parser.add_argument("--global-token-budget", type=int, default=1024)
    parser.add_argument("--fine-token-budget", type=int, default=3072)
    parser.add_argument(
        "--fusion-mode",
        choices=("append", "replace_coarse", "replace_raster"),
        default="append",
        help="How global and fine tokens form the single visual span.",
    )
    parser.add_argument(
        "--global-resample",
        choices=("lanczos", "bicubic"),
        default="lanczos",
        help="Keep lanczos for Stage-1 reproducibility; online PTEA uses bicubic.",
    )
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument(
        "--candidate-policy",
        choices=("portfolio", "focus_only"),
        default="portfolio",
        help="Which frozen EviMap regions become internal foveal views.",
    )
    parser.add_argument("--minimum-region-tokens", type=int, default=64)
    parser.add_argument("--allocation-mode", choices=("uniform", "evidence"), default="evidence")
    parser.add_argument("--coordinate-bins", type=int, default=128)
    parser.add_argument(
        "--coordinate-mode",
        choices=("original_xy_scale_t", "original_xy_zero_t"),
        default="original_xy_scale_t",
        help="MRoPE protocol for the unified static-image visual span.",
    )
    parser.add_argument(
        "--prompt-variant", choices=tuple(SYSTEM_PROMPTS), default="canonical"
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--record-timing", action="store_true")
    parser.add_argument("--timing-warmup-samples", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if (args.box_predictions is None) == (args.selector_checkpoint is None):
        parser.error(
            "provide exactly one of --box-predictions or --selector-checkpoint"
        )

    decision = check_gpu(args.gpu, args.memory_budget_mib, ROOT / "configs/gpu_safety.json")
    print(json.dumps({"gpu_preflight": decision.as_dict()}, ensure_ascii=False), flush=True)
    if not decision.allowed:
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = "cuda:0"
    exposed_total_mib = torch.cuda.get_device_properties(0).total_memory / 1024**2
    torch.cuda.set_per_process_memory_fraction(
        min(0.95, args.memory_budget_mib / exposed_total_mib), device=0
    )
    torch.cuda.reset_peak_memory_stats(0)

    rows = read_jsonl(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    box_rows: dict[str, dict[str, Any]] = {}
    if args.box_predictions is not None:
        box_rows = {str(row["id"]): row for row in read_jsonl(args.box_predictions)}
        missing = [str(row["id"]) for row in rows if str(row["id"]) not in box_rows]
        if missing:
            raise RuntimeError(
                f"missing {len(missing)} evidence portfolios; first={missing[:5]}"
            )
    existing = read_jsonl(args.output) if args.resume and args.output.exists() else []
    completed = {str(row["id"]) for row in existing}
    remaining = [row for row in rows if str(row["id"]) not in completed]

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    # All views are explicitly pre-sized.  The processor ceiling only prevents
    # accidental second-stage rescaling; it is not the effective visual budget.
    processor = AutoProcessor.from_pretrained(
        str(args.model), local_files_only=True, max_pixels=16 * 1024 * 1024, min_pixels=4096
    )
    selector_processor = None
    if (
        args.selector_checkpoint is not None
        and args.selector_global_preprocess == "qwen_native"
    ):
        # Use the exact native preprocessing protocol that produced PTEA's
        # training/evaluation features. The resulting global ViT embeddings are
        # reused by the QA path, so this is not a second global image encoding.
        selector_processor = AutoProcessor.from_pretrained(
            str(args.model),
            local_files_only=True,
            max_pixels=args.selector_max_pixels,
            min_pixels=4096,
        )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model), dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": device}, local_files_only=True,
    )
    model.eval()
    merge_size = int(model.config.vision_config.spatial_merge_size)
    patch_size = int(model.config.vision_config.patch_size)
    merged_stride = merge_size * patch_size
    selector = None
    selector_projection = None
    if args.selector_checkpoint is not None:
        from scripts.extract_qwen_dense_features import build_projection
        from scripts.predict_patch_text_topk_boxes import load_selector

        hidden_size = int(model.config.text_config.hidden_size)
        selector_projection = build_projection(
            hidden_size,
            args.selector_projection_dim,
            args.selector_projection_seed,
            device,
        )
        selector = load_selector(
            args.selector_checkpoint,
            args.selector_projection_dim,
            device,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if existing and args.resume else "w"
    output_rows = list(existing)
    started = time.perf_counter()
    with args.output.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(remaining, 1):
            if args.record_timing:
                torch.cuda.synchronize(0)
            request_started = time.perf_counter()
            row_id = str(row["id"])
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = args.project_root / image_path
            source = Image.open(image_path).convert("RGB")
            global_h, global_w = grid_for_token_budget(
                source.width, source.height, args.global_token_budget
            )
            if selector is None:
                global_view = resize_to_token_grid(
                    source,
                    global_h,
                    global_w,
                    merged_patch_stride=merged_stride,
                    downsample_resample=(
                        Image.Resampling.BICUBIC
                        if args.global_resample == "bicubic"
                        else Image.Resampling.LANCZOS
                    ),
                )
            elif args.selector_global_preprocess == "qwen_native":
                global_view = source
            else:
                global_view = resize_to_token_grid(
                    source,
                    global_h,
                    global_w,
                    merged_patch_stride=merged_stride,
                    downsample_resample=Image.Resampling.LANCZOS,
                )
            online_global = None
            selector_stats: dict[str, float] = {}
            if selector is None:
                raw_candidates = box_rows[row_id]["candidates"]
            else:
                if selector_projection is None:
                    raise AssertionError("online selector projection was not initialized")
                global_processor = (
                    selector_processor
                    if args.selector_global_preprocess == "qwen_native"
                    else processor
                )
                if global_processor is None:
                    raise AssertionError("online selector processor was not initialized")
                online_global = encode_internal_views_sequentially(
                    model,
                    global_processor,
                    [global_view],
                    device=device,
                    merge_size=merge_size,
                )
                selector_question = selector_question_from_prompt(
                    str(row["question"]), protocol=args.selector_question_protocol
                )
                raw_candidates, selector_stats = predict_online_ptea_candidates(
                    model,
                    processor,
                    selector,
                    selector_projection,
                    online_global[0],
                    online_global[2][0],
                    selector_question,
                    device=device,
                )
            candidates = select_region_candidates(
                raw_candidates,
                policy=args.candidate_policy,
                max_regions=args.max_regions,
            )
            budgets = allocate_region_budgets(
                candidates,
                total_tokens=args.fine_token_budget,
                max_regions=len(candidates),
                minimum_tokens=args.minimum_region_tokens,
                mode=args.allocation_mode,
            )

            internal_views = [global_view]
            boxes: list[list[int]] = [[0, 0, 1000, 1000]]
            plans: list[dict[str, Any]] = []
            for budget in budgets:
                pixel_box = policy_to_pixels(budget.bbox, *source.size)
                crop = source.crop(pixel_box).convert("RGB")
                grid_h, grid_w = grid_for_token_budget(
                    crop.width, crop.height, budget.token_budget
                )
                internal_views.append(
                    resize_to_token_grid(
                        crop, grid_h, grid_w, merged_patch_stride=merged_stride
                    )
                )
                boxes.append(list(budget.bbox))
                plans.append(
                    {
                        **budget.as_dict(),
                        "pixel_bbox": list(pixel_box),
                        "token_grid": [grid_h, grid_w],
                        "realized_tokens": grid_h * grid_w,
                    }
                )

            if online_global is None:
                unified, deepstack, grid_shapes = encode_internal_views_sequentially(
                    model,
                    processor,
                    internal_views,
                    device=device,
                    merge_size=merge_size,
                )
            else:
                fine, fine_deepstack, fine_grid_shapes = encode_internal_views_sequentially(
                    model,
                    processor,
                    internal_views[1:],
                    device=device,
                    merge_size=merge_size,
                )
                unified = torch.cat([online_global[0], fine], dim=0)
                deepstack = [
                    torch.cat([global_layer, fine_layer], dim=0)
                    for global_layer, fine_layer in zip(
                        online_global[1], fine_deepstack
                    )
                ]
                grid_shapes = online_global[2] + fine_grid_shapes
            metadata = multiresolution_token_metadata(grid_shapes, boxes)
            fusion_indices = multiresolution_fusion_indices(
                metadata, boxes, mode=args.fusion_mode
            )
            fusion_tensor_indices = torch.from_numpy(fusion_indices).to(
                device=unified.device, dtype=torch.long
            )
            unified = unified.index_select(0, fusion_tensor_indices)
            deepstack = [
                layer.index_select(0, fusion_tensor_indices) for layer in deepstack
            ]
            metadata = {
                key: np.asarray(value)[fusion_indices]
                for key, value in metadata.items()
            }
            coordinates = quantize_mrope_coordinates(
                metadata,
                coordinate_bins=args.coordinate_bins,
                mode=args.coordinate_mode,
            )
            generation_result = greedy_generate_from_visual_span(
                model,
                processor,
                build_single_span_prompt(
                    processor, str(row["question"]), prompt_variant=args.prompt_variant
                ),
                unified,
                deepstack,
                coordinates,
                device=device,
                max_new_tokens=args.max_new_tokens,
                record_timing=args.record_timing,
            )
            if args.record_timing:
                raw, generation_timing = generation_result
            else:
                raw = generation_result
            prediction, extracted, parse_error = parse_answer(raw)
            target = str(row.get("answer", ""))
            output = {
                "id": row["id"],
                "eval_tier": row.get("eval_tier"),
                "image": row["image"],
                "question": row["question"],
                "answer": target,
                "raw_prediction": raw,
                "extracted_prediction": extracted,
                "prediction": prediction,
                "normalized_prediction": normalized_answer(prediction),
                "normalized_answer": normalized_answer(target),
                "relaxed_prediction": relaxed_answer(prediction),
                "relaxed_answer": relaxed_answer(target),
                "exact_correct": normalized_answer(prediction) == normalized_answer(target),
                "relaxed_correct": relaxed_answer_match(prediction, target),
                "parse_error": parse_error,
                "method": (
                    "evivit_stage2_online_ptea_focus3"
                    if selector is not None
                    else "evivit_stage1_unified_multiresolution_tokens"
                ),
                "interface_images": 1,
                "source_images": 1,
                "internal_views": len(internal_views),
                "global_token_budget": args.global_token_budget,
                "fine_token_budget": args.fine_token_budget,
                "global_resample": args.global_resample,
                "global_preprocess": (
                    f"{args.selector_global_preprocess}_selector"
                    if selector is not None
                    else args.global_resample
                ),
                "realized_visual_tokens": int(unified.shape[0]),
                "pre_fusion_visual_tokens": int(len(fusion_indices))
                if args.fusion_mode == "append"
                else int(sum(shape[0] * shape[1] for shape in grid_shapes)),
                "fusion_mode": args.fusion_mode,
                "grid_shapes": grid_shapes,
                "allocation_mode": args.allocation_mode,
                "candidate_policy": args.candidate_policy,
                "selector_mode": (
                    "online_ptea" if selector is not None else "precomputed_portfolio"
                ),
                "selector_checkpoint": (
                    str(args.selector_checkpoint) if args.selector_checkpoint else None
                ),
                "selector_global_preprocess": (
                    args.selector_global_preprocess if selector is not None else None
                ),
                "selector_question_protocol": (
                    args.selector_question_protocol if selector is not None else None
                ),
                "selector_stats": selector_stats,
                "coordinate_protocol": (
                    f"{args.coordinate_mode}_mrope_{args.coordinate_bins}_v1"
                ),
                "prompt_variant": args.prompt_variant,
                "region_plans": plans,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            }
            if args.record_timing:
                output["timing"] = {
                    "request_to_first_token_seconds": float(
                        generation_timing["_first_token_wall_time"]
                    )
                    - request_started,
                    "end_to_end_seconds": float(
                        generation_timing["_generation_finished_wall_time"]
                    )
                    - request_started,
                    "generation_seconds": generation_timing["generation_seconds"],
                    "decode_seconds": generation_timing["decode_seconds"],
                    "llm_prefill_to_first_token_seconds": generation_timing[
                        "llm_prefill_to_first_token_seconds"
                    ],
                    "generated_tokens": generation_timing["generated_tokens"],
                }
            handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            output_rows.append(output)
            print(
                json.dumps(
                    {
                        "index": index,
                        "remaining": len(remaining),
                        "id": row_id,
                        "exact": output["exact_correct"],
                        "relaxed": output["relaxed_correct"],
                        "visual_tokens": output["realized_visual_tokens"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - started
    peak_reserved_mib = torch.cuda.max_memory_reserved() / 1024**2
    summary_row = summarize(output_rows, elapsed, peak_reserved_mib)
    summary_row.update(
        {
            "method": (
                "evivit_stage2_online_ptea_focus3"
                if selector is not None
                else "evivit_stage1_unified_multiresolution_tokens"
            ),
            "output": str(args.output),
            "global_token_budget": args.global_token_budget,
            "fine_token_budget": args.fine_token_budget,
            "fusion_mode": args.fusion_mode,
            "global_resample": args.global_resample,
            "global_preprocess": (
                f"{args.selector_global_preprocess}_selector"
                if selector is not None
                else args.global_resample
            ),
            "allocation_mode": args.allocation_mode,
            "candidate_policy": args.candidate_policy,
            "selector_mode": (
                "online_ptea" if selector is not None else "precomputed_portfolio"
            ),
            "selector_global_preprocess": (
                args.selector_global_preprocess if selector is not None else None
            ),
            "selector_question_protocol": (
                args.selector_question_protocol if selector is not None else None
            ),
            "prompt_variant": args.prompt_variant,
            "coordinate_mode": args.coordinate_mode,
            "max_regions": args.max_regions,
            "mean_realized_visual_tokens": (
                sum(int(row["realized_visual_tokens"]) for row in output_rows) / len(output_rows)
                if output_rows else 0.0
            ),
            "elapsed_seconds": elapsed,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "record_timing": args.record_timing,
            "latency": (
                summarize_latency(
                    output_rows, warmup_samples=args.timing_warmup_samples
                )
                if args.record_timing
                else None
            ),
        }
    )
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary_row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary_row, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
