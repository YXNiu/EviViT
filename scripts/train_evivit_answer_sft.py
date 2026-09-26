#!/usr/bin/env python3
"""Matched answer-only LoRA SFT for Base Qwen-ViT and EviViT.

The four registered variants share one prompt, QA manifest, sample order,
answer-token loss, LoRA topology, optimizer, scheduler, effective batch, and
checkpoint format.  They differ only in the frozen visual input path:

* ``base_q4m`` / ``base_q16m``: native Qwen image processor and Qwen-ViT.
* ``evivit_a2`` / ``evivit_a4``: frozen Mid-PTEA, frozen sparse Bridge, and
  the registered EviViT visual budgets.

Only language-model q/k/v/o LoRA parameters may receive gradients.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_evivit_v3_mid_bridge import bridge_from_checkpoint  # noqa: E402
from scripts.extract_qwen_dense_features import build_projection  # noqa: E402
from scripts.predict_patch_text_topk_boxes import load_selector  # noqa: E402
from scripts.train_evivit_v3_bridge import (  # noqa: E402
    build_chat_text,
    build_position_ids,
    resize_to_token_grid,
    visual_coordinates,
)
from evivit_core.continuous_visual_budget import plan_continuous_budget  # noqa: E402
from evivit_core.evivit import grid_for_token_budget, selector_question_from_prompt  # noqa: E402
from evivit_core.evivit_posttraining import (  # noqa: E402
    LANGUAGE_LORA_TARGET_PATTERN,
    SYSTEM_PROMPT,
    VARIANTS,
    assert_language_lora_only,
    expand_single_image_labels,
    labels_from_offsets,
    micro_batches_in_optimizer_step,
    parameter_report,
    read_qa_jsonl,
    render_answer_texts,
    resolve_training_schedule,
    sha256_file,
    warmup_steps,
)
from evivit_core.evivit_recovery_sft import (  # noqa: E402
    RECOVERY_ADAMW_BETAS,
    RECOVERY_ADAMW_EPS,
    RECOVERY_EPOCH_ORDER_VERSION,
    epoch_order,
)
from evivit_core.evivit_v3_online import (  # noqa: E402
    allocate_mid_ptea_evidence,
    qwen_projected_question_tokens,
)
from evivit_core.geometry import policy_to_pixels  # noqa: E402
from evivit_core.gpu_safety import check_gpu  # noqa: E402
from evivit_core.qwen3vl_evivit_v3_mid_encoder import (  # noqa: E402
    MidViTFineViewInput,
    Qwen3VLEviViTV3MidEncoder,
)


FORMAT_VERSION = "evivit_answer_sft_v1"


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def resolve_path(path: Path, project_root: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(value), ensure_ascii=False) + "\n")
        handle.flush()


def cuda_snapshot(gpu: int) -> dict[str, int] | None:
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        values = [int(item.strip()) for item in result.stdout.strip().split(",")]
    except Exception:
        return None
    return {
        "memory_used_mib": values[0],
        "memory_free_mib": values[1],
        "utilization_gpu": values[2],
        "temperature_gpu": values[3],
    }


def rendered_token_labels(
    processor: Any,
    question: str,
    answer: str,
    *,
    answer_protocol: str = "direct_concise",
) -> tuple[str, list[int], list[int]]:
    """Return rendered full chat, raw ids, and answer-only raw labels."""

    if answer_protocol == "direct_concise":
        _, full_text, answer_span = render_answer_texts(processor, question, answer)
    elif answer_protocol == "compact_json":
        prompt_text = build_chat_text(
            processor, question, None, answer_protocol="compact_json"
        )
        full_text = build_chat_text(
            processor, question, answer, answer_protocol="compact_json"
        )
        if not full_text.startswith(prompt_text):
            raise RuntimeError("compact-JSON target is not a prompt continuation")
        target_json = json.dumps(
            {"answer": str(answer).strip()},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        encoded_value = json.dumps(str(answer).strip(), ensure_ascii=False)
        target_start = full_text.rfind(target_json)
        value_offset = target_json.find(encoded_value)
        if target_start < 0 or value_offset < 0:
            raise RuntimeError("could not locate compact-JSON answer value")
        value_start = target_start + value_offset
        value_end = value_start + len(encoded_value)
        if encoded_value.startswith('"') and encoded_value.endswith('"'):
            value_start += 1
            value_end -= 1
        answer_span = (value_start, value_end)
    else:
        raise ValueError(f"unknown answer protocol: {answer_protocol}")
    encoded = processor.tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    raw_ids = [int(value) for value in encoded.input_ids]
    raw_labels = labels_from_offsets(raw_ids, encoded.offset_mapping, answer_span)
    return full_text, raw_ids, raw_labels


def answer_loss_from_hidden(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    lm_head: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Project only supervised causal positions instead of the full vocabulary grid."""

    shifted_targets = labels[:, 1:]
    mask = shifted_targets.ne(-100)
    count = int(mask.sum().item())
    if count <= 0:
        raise ValueError("batch has no answer target tokens")
    selected_hidden = hidden_states[:, :-1][mask]
    targets = shifted_targets[mask]
    logits = lm_head(selected_hidden)
    loss = F.cross_entropy(logits.float(), targets)
    with torch.no_grad():
        accuracy = logits.argmax(dim=-1).eq(targets).float().mean()
    return loss, accuracy, count


def build_base_batch(
    *,
    processor: Any,
    image: Image.Image,
    question: str,
    answer: str,
    image_token_id: int,
    device: str,
    answer_protocol: str = "direct_concise",
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    full_text, raw_ids, raw_labels = rendered_token_labels(
        processor, question, answer, answer_protocol=answer_protocol
    )
    batch = processor(
        text=[full_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    expanded_ids = [int(value) for value in batch["input_ids"][0].tolist()]
    expanded_labels, visual_start, visual_end = expand_single_image_labels(
        raw_ids, raw_labels, expanded_ids, image_token_id
    )
    labels = torch.tensor([expanded_labels], dtype=torch.long, device=device)
    moved = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    return moved, labels, {
        "input_tokens": len(expanded_ids),
        "answer_tokens": sum(value != -100 for value in expanded_labels),
        "visual_tokens": visual_end - visual_start,
        "regions": 0,
    }


@dataclass
class EviRuntime:
    selector: Any
    projection: torch.Tensor
    bridge: Any
    encoder: Qwen3VLEviViTV3MidEncoder
    merged_stride: int
    insertion_block: int
    coordinate_bins: int
    scale_bins: int


def build_evi_runtime(
    *,
    qwen: Any,
    selector_checkpoint: Path,
    bridge_checkpoint: Path,
    device: str,
    insertion_block: int,
    bridge_dim: int,
    bridge_heads: int,
    neighborhood_radius: int,
    max_relative_residual: float,
    projection_dim: int,
    projection_seed: int,
    coordinate_bins: int,
    scale_bins: int,
) -> EviRuntime:
    selector = load_selector(
        selector_checkpoint,
        int(qwen.config.vision_config.hidden_size),
        device,
    ).eval()
    selector.requires_grad_(False)
    projection = build_projection(
        int(qwen.config.text_config.hidden_size),
        projection_dim,
        projection_seed,
        device,
    )
    bridge, metadata = bridge_from_checkpoint(
        bridge_checkpoint,
        model=qwen,
        device=device,
        default_insertion_block=insertion_block,
        default_bridge_dim=bridge_dim,
        default_bridge_heads=bridge_heads,
        default_neighborhood_radius=neighborhood_radius,
        default_max_relative_residual=max_relative_residual,
    )
    bridge.requires_grad_(False)
    bridge.eval()
    encoder = Qwen3VLEviViTV3MidEncoder(
        qwen.model.visual,
        bridge,
        insertion_block=int(metadata["insertion_block"]),
        freeze_vision=True,
    ).eval()
    merge_size = int(qwen.config.vision_config.spatial_merge_size)
    merged_stride = merge_size * int(qwen.config.vision_config.patch_size)
    return EviRuntime(
        selector=selector,
        projection=projection,
        bridge=bridge,
        encoder=encoder,
        merged_stride=merged_stride,
        insertion_block=int(metadata["insertion_block"]),
        coordinate_bins=coordinate_bins,
        scale_bins=scale_bins,
    )


@torch.no_grad()
def encode_evivit_image(
    *,
    runtime: EviRuntime,
    qwen: Any,
    processor: Any,
    source: Image.Image,
    question: str,
    variant: dict[str, Any],
    device: str,
    minimum_region_tokens: int,
    selector_attention_backend: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    continuous_config = variant.get("continuous_budget")
    native_global_total_budget = int(
        variant.get("native_global_total_token_budget", 0)
    )
    if continuous_config is not None and native_global_total_budget:
        raise ValueError("continuous budgets cannot be combined with Native-Fit routing")
    continuous_plan = (
        plan_continuous_budget(source.width, source.height, continuous_config)
        if continuous_config is not None
        else None
    )
    global_token_budget = (
        continuous_plan.global_tokens
        if continuous_plan is not None
        else int(variant["global_token_budget"])
    )
    configured_fine_token_budget = (
        continuous_plan.fine_tokens
        if continuous_plan is not None
        else int(variant["fine_token_budget"])
    )
    use_native_global = False
    native_global_tokens: int | None = None
    owned_global_view = True
    if native_global_total_budget > 0:
        native_inputs_cpu = processor(
            text=["."], images=[source], return_tensors="pt"
        )
        native_grid = [
            int(value)
            for value in native_inputs_cpu["image_grid_thw"][0].tolist()
        ]
        merge_size = int(qwen.config.vision_config.spatial_merge_size)
        native_global_tokens = (
            native_grid[0]
            * (native_grid[1] // merge_size)
            * (native_grid[2] // merge_size)
        )
        use_native_global = native_global_tokens <= native_global_total_budget
    else:
        native_inputs_cpu = None

    if use_native_global:
        assert native_inputs_cpu is not None and native_global_tokens is not None
        global_view = source
        owned_global_view = False
        global_inputs = native_inputs_cpu.to(device)
        fine_token_budget = min(
            configured_fine_token_budget,
            max(0, native_global_total_budget - native_global_tokens),
        )
    else:
        if native_inputs_cpu is not None:
            del native_inputs_cpu
        global_h, global_w = grid_for_token_budget(
            source.width, source.height, global_token_budget
        )
        global_view = resize_to_token_grid(
            source,
            global_h,
            global_w,
            merged_patch_stride=runtime.merged_stride,
        )
        global_inputs = processor(
            text=["."], images=[global_view], return_tensors="pt"
        ).to(device)
        fine_token_budget = configured_fine_token_budget
    prepared = runtime.encoder.prepare_global(
        global_inputs["pixel_values"], global_inputs["image_grid_thw"]
    )
    max_regions = int(variant["max_regions"])
    selector_question_protocol = str(
        variant.get("selector_question_protocol", "full_prompt")
    )
    if fine_token_budget > 0 and max_regions > 0:
        selector_question = selector_question_from_prompt(
            question,
            protocol=selector_question_protocol,
        )
        question_tokens = qwen_projected_question_tokens(
            qwen,
            processor.tokenizer,
            selector_question,
            runtime.projection,
            device=device,
        )
        allocation = allocate_mid_ptea_evidence(
            runtime.selector,
            prepared.map_feature_grid,
            question_tokens,
            fine_token_budget=fine_token_budget,
            max_regions=max_regions,
            minimum_region_tokens=minimum_region_tokens,
            evidence_decoder=str(variant["evidence_decoder"]),
            region_expansion_factor=float(variant["region_expansion_factor"]),
            topology_union_factor=float(variant["topology_union_factor"]),
            selector_attention_backend=selector_attention_backend,
        )
        region_budgets = allocation.region_budgets
        map_entropy: float | None = allocation.map_entropy
        map_peak_probability: float | None = allocation.map_peak_probability
    elif fine_token_budget == 0 and max_regions == 0:
        # A mechanism-only global-path control.  Skipping PTEA here is
        # intentional: with no fine budget its map cannot affect the visual
        # sequence, and computing it would only add irrelevant cost.
        region_budgets = ()
        map_entropy = None
        map_peak_probability = None
    else:
        raise ValueError("fine_token_budget and max_regions must both be zero or positive")
    maximum_weight = max((budget.score for budget in region_budgets), default=1.0)
    fine_views: list[MidViTFineViewInput] = []
    region_plans: list[dict[str, Any]] = []
    for budget in region_budgets:
        pixel_box = policy_to_pixels(budget.bbox, *source.size)
        crop = source.crop(pixel_box).convert("RGB")
        fine_h, fine_w = grid_for_token_budget(
            crop.width, crop.height, budget.token_budget
        )
        fine_image = resize_to_token_grid(
            crop,
            fine_h,
            fine_w,
            merged_patch_stride=runtime.merged_stride,
        )
        fine_inputs = processor(
            text=["."], images=[fine_image], return_tensors="pt"
        ).to(device)
        fine_views.append(
            MidViTFineViewInput(
                pixel_values=fine_inputs["pixel_values"],
                grid_thw=fine_inputs["image_grid_thw"],
                bbox_xyxy=tuple(value / 1000.0 for value in budget.bbox),
                evidence_weight=budget.score / max(maximum_weight, 1e-8),
            )
        )
        region_plans.append(
            {
                "bbox": list(budget.bbox),
                "token_budget": int(budget.token_budget),
                "realized_tokens": fine_h * fine_w,
                "evidence_weight": float(
                    budget.score / max(maximum_weight, 1e-8)
                ),
            }
        )
        crop.close()
        fine_image.close()
    bridge_mode = str(
        variant.get(
            "native_global_bridge_mode" if use_native_global else "fallback_bridge_mode",
            variant.get("bridge_mode", "bidirectional"),
        )
    )
    salience_enabled = runtime.encoder.evidence_salience_adapter is not None
    if salience_enabled and fine_token_budget <= 0:
        raise ValueError("salience adapter requires a PTEA allocation")
    vision = runtime.encoder.complete_from_prepared(
        prepared,
        fine_views,
        bridge_mode=bridge_mode,
        visual_output_mode=str(variant.get("visual_output_mode", "append")),
        global_evidence_map=(
            allocation.probability_map if salience_enabled else None
        ),
    )
    if owned_global_view:
        global_view.close()
    return vision, {
        "visual_tokens": int(vision.image_embeds.shape[0]),
        "global_tokens": int(vision.global_image_tokens),
        "fine_tokens": int(vision.fine_image_tokens),
        "fine_encoder_tokens": int(vision.fine_encoder_tokens),
        "native_global_used": use_native_global,
        "native_global_tokens": native_global_tokens,
        "native_global_total_token_budget": native_global_total_budget,
        "effective_fine_token_budget": fine_token_budget,
        "budget_policy": (
            continuous_plan.policy if continuous_plan is not None else "fixed"
        ),
        "source_pixels": source.width * source.height,
        "nominal_total_token_budget": (
            continuous_plan.total_tokens
            if continuous_plan is not None
            else global_token_budget + configured_fine_token_budget
        ),
        "nominal_global_token_budget": global_token_budget,
        "nominal_fine_token_budget": configured_fine_token_budget,
        "nominal_global_fraction": (
            continuous_plan.global_fraction
            if continuous_plan is not None
            else None
        ),
        "raw_continuous_total_tokens": (
            continuous_plan.raw_total_tokens
            if continuous_plan is not None
            else None
        ),
        "regions": len(fine_views),
        "region_plans": region_plans,
        "map_entropy": map_entropy,
        "map_peak_probability": map_peak_probability,
        "selector_question_protocol": selector_question_protocol,
        "bridge_mode": bridge_mode,
        "visual_output_mode": vision.visual_output_mode,
        "bridge_global_relative_residual_l2": float(
            vision.global_relative_residual_l2.detach().float().cpu()
        ),
        "bridge_fine_relative_residual_l2": float(
            vision.fine_relative_residual_l2.detach().float().cpu()
        ),
        "bridge_relative_residual_l2": float(
            vision.relative_residual_l2.detach().float().cpu()
        ),
        "relay_relative_residual_l2": float(
            vision.relay_relative_residual_l2.detach().float().cpu()
        ),
        "relay_gate_mean": (
            vision.relay_diagnostics.gate_mean
            if vision.relay_diagnostics is not None
            else None
        ),
        "relay_gate_std": (
            vision.relay_diagnostics.gate_std
            if vision.relay_diagnostics is not None
            else None
        ),
        "relay_attention_entropy_normalized": (
            vision.relay_diagnostics.attention_entropy_normalized
            if vision.relay_diagnostics is not None
            else None
        ),
        "relay_pre_projection_anchor_attention_mass": (
            vision.relay_diagnostics.pre_projection_anchor_attention_mass
            if vision.relay_diagnostics is not None
            else None
        ),
        "relay_anchor_attention_mass": (
            vision.relay_diagnostics.anchor_attention_mass
            if vision.relay_diagnostics is not None
            else None
        ),
        "relay_region_attention_mass": (
            vision.relay_diagnostics.region_attention_mass
            if vision.relay_diagnostics is not None
            else None
        ),
        "relay_self_region_attention_mass": (
            vision.relay_diagnostics.self_region_attention_mass
            if vision.relay_diagnostics is not None
            else None
        ),
        "relay_anchor_floor_activation_rate": (
            vision.relay_diagnostics.anchor_floor_activation_rate
            if vision.relay_diagnostics is not None
            else None
        ),
        "salience_relative_residual_l2": float(
            vision.salience_relative_residual_l2.detach().float().cpu()
        ),
        "salience_importance_std": (
            vision.salience_diagnostics.importance_std
            if vision.salience_diagnostics is not None
            else None
        ),
        "salience_importance_maximum": (
            vision.salience_diagnostics.importance_maximum
            if vision.salience_diagnostics is not None
            else None
        ),
    }


def evivit_answer_loss(
    *,
    qwen: Any,
    processor: Any,
    vision: Any,
    question: str,
    answer: str,
    runtime: EviRuntime,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    full_text, raw_ids, raw_labels = rendered_token_labels(
        processor, question, answer
    )
    visual_tokens = int(vision.image_embeds.shape[0])
    image_token_id = int(qwen.config.image_token_id)
    positions = [index for index, value in enumerate(raw_ids) if value == image_token_id]
    if len(positions) != 1:
        raise ValueError(f"expected one image placeholder, found {len(positions)}")
    visual_start = positions[0]
    expanded_ids = (
        raw_ids[:visual_start]
        + [image_token_id] * visual_tokens
        + raw_ids[visual_start + 1 :]
    )
    labels_list, checked_start, visual_end = expand_single_image_labels(
        raw_ids, raw_labels, expanded_ids, image_token_id
    )
    if checked_start != visual_start:
        raise RuntimeError("visual start changed during label expansion")
    input_ids = torch.tensor([expanded_ids], dtype=torch.long, device=device)
    labels = torch.tensor([labels_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    inputs_embeds = qwen.get_input_embeddings()(input_ids)
    visual_mask = input_ids.eq(image_token_id)
    inputs_embeds = inputs_embeds.masked_scatter(
        visual_mask.unsqueeze(-1),
        vision.image_embeds.detach().to(inputs_embeds.dtype),
    )
    position_ids = build_position_ids(
        input_ids.shape[1],
        visual_start,
        visual_end,
        visual_coordinates(
            vision,
            coordinate_bins=runtime.coordinate_bins,
            scale_bins=runtime.scale_bins,
        ),
        device=device,
    )
    outputs = qwen.model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_pos_masks=visual_mask,
        deepstack_visual_embeds=[
            feature.detach().to(inputs_embeds.dtype)
            for feature in vision.deepstack_features
        ],
        use_cache=False,
    )
    loss, accuracy, answer_tokens = answer_loss_from_hidden(
        outputs.last_hidden_state, labels, qwen.lm_head
    )
    return loss, accuracy, {
        "input_tokens": len(expanded_ids),
        "answer_tokens": answer_tokens,
        "visual_tokens": visual_tokens,
    }


def protocol_fingerprint(config: dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(json_safe(config), ensure_ascii=False, sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def save_adapter_atomic(model: Any, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    model.save_pretrained(temporary)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.rename(destination)


def save_resume_state(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    output_dir: Path,
    rows: list[dict[str, Any]],
    global_step: int,
    micro_step: int,
    row_index: int,
    next_epoch_to_save: float,
    target_micro_steps: int,
    optimizer_steps: int,
    gradient_accumulation_steps: int,
    fingerprint: str,
) -> Path:
    latest = output_dir / "resume-latest"
    temporary = output_dir / "resume-latest.tmp"
    previous = output_dir / "resume-latest.previous"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary)
    state = {
        "format_version": FORMAT_VERSION,
        "protocol_fingerprint": fingerprint,
        "global_step": global_step,
        "micro_step": micro_step,
        "row_index": row_index,
        "next_epoch_to_save": next_epoch_to_save,
        "target_micro_steps": target_micro_steps,
        "optimizer_steps": optimizer_steps,
        "selected_rows": len(rows),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "row_order": [str(row["id"]) for row in rows],
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state(0),
    }
    torch.save(state, temporary / "training_state.pt")
    atomic_write_json(
        temporary / "training_state.json",
        {
            key: value
            for key, value in state.items()
            if key
            not in {
                "row_order",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "python_random_state",
                "torch_cpu_rng_state",
                "torch_cuda_rng_state",
            }
        },
    )
    shutil.rmtree(previous, ignore_errors=True)
    if latest.exists():
        latest.rename(previous)
    temporary.rename(latest)
    shutil.rmtree(previous, ignore_errors=True)
    return latest


def restore_resume_state(
    *,
    checkpoint: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    rows: list[dict[str, Any]],
    device: str,
    target_micro_steps: int,
    optimizer_steps: int,
    gradient_accumulation_steps: int,
    fingerprint: str,
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    state = torch.load(
        checkpoint / "training_state.pt", map_location=device, weights_only=False
    )
    expected = {
        "format_version": FORMAT_VERSION,
        "protocol_fingerprint": fingerprint,
        "target_micro_steps": target_micro_steps,
        "optimizer_steps": optimizer_steps,
        "selected_rows": len(rows),
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    mismatch = {
        key: {"checkpoint": state.get(key), "current": value}
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatch:
        raise ValueError(f"resume protocol mismatch: {mismatch}")
    by_id = {str(row["id"]): row for row in rows}
    order = [str(value) for value in state["row_order"]]
    if len(by_id) != len(rows) or set(order) != set(by_id):
        raise ValueError("resume row order does not match the manifest")
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    random.setstate(state["python_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
    torch.cuda.set_rng_state(state["torch_cuda_rng_state"].cpu(), device=0)
    return [by_id[row_id] for row_id in order], {
        "global_step": int(state["global_step"]),
        "micro_step": int(state["micro_step"]),
        "row_index": int(state["row_index"]),
        "next_epoch_to_save": float(state["next_epoch_to_save"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument(
        "--recovery-lane",
        choices=("B-L",),
        help="Enable the strict Base/B16 side of the Visual-CoT recovery matrix.",
    )
    parser.add_argument(
        "--recovery-engineering-smoke",
        action="store_true",
        help="Allow at most 128 rows while validating B-L before the 2K run.",
    )
    parser.add_argument(
        "--answer-protocol",
        choices=("direct_concise", "compact_json"),
        default="direct_concise",
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--project-root", type=Path, default=Path(".")
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen3-VL-4B-Instruct"),
    )
    parser.add_argument("--selector-checkpoint", type=Path)
    parser.add_argument("--bridge-checkpoint", type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--memory-budget-mib", type=int, default=85000)
    parser.add_argument("--recovery-recipe", choices=("r2", "final_v1"), default="r2")
    parser.add_argument("--num-epochs", type=int, default=8)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--scheduler-type", choices=("linear", "cosine"), default="linear"
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--save-resume-steps", type=int, default=24)
    parser.add_argument(
        "--stop-after-optimizer-steps",
        type=int,
        default=0,
        help="Smoke-only controlled interruption used to verify exact resume.",
    )
    parser.add_argument("--minimum-region-tokens", type=int, default=64)
    parser.add_argument("--insertion-block", type=int, default=16)
    parser.add_argument("--bridge-dim", type=int, default=256)
    parser.add_argument("--bridge-heads", type=int, default=4)
    parser.add_argument("--neighborhood-radius", type=int, default=1)
    parser.add_argument("--max-relative-residual", type=float, default=0.2)
    parser.add_argument("--coordinate-bins", type=int, default=128)
    parser.add_argument("--scale-bins", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--projection-seed", type=int, default=20260715)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variant = dict(VARIANTS[args.variant])
    if args.recovery_lane:
        if args.variant != "base_q16m":
            raise ValueError("B-L recovery must use the native Base/B16 visual path")
        if args.answer_protocol != "compact_json":
            raise ValueError("B-L recovery must use compact_json")
        if args.scheduler_type != "cosine":
            raise ValueError("B-L recovery must use cosine scheduling")
        if args.recovery_engineering_smoke:
            if args.max_train_rows <= 0 or args.max_train_rows > 128:
                raise ValueError("B-L engineering smoke is limited to 1-128 rows")
        elif (
            args.num_epochs != 3
            or args.gradient_accumulation_steps != 20
            or args.learning_rate != (5e-6 if args.recovery_recipe == "final_v1" else 1e-6)
        ):
            raise ValueError(
                "formal B-L requires 3 epochs, accumulation 20, and the registered recipe LR"
            )
    if variant.get("relay_required"):
        raise ValueError(
            "Global Anchor Relay variants use their dedicated visual-module trainer; "
            "the language-LoRA SFT entry point must not silently omit the Relay checkpoint"
        )
    if args.num_epochs <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("epochs and gradient accumulation must be positive")
    if variant["family"] == "evivit" and (
        args.selector_checkpoint is None or args.bridge_checkpoint is None
    ):
        raise ValueError("EviViT variants require selector and bridge checkpoints")
    if args.resume_from_checkpoint is None:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty output directory: {args.output_dir}"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not args.resume_from_checkpoint.is_dir():
            raise FileNotFoundError(args.resume_from_checkpoint)
        args.output_dir.mkdir(parents=True, exist_ok=True)

    decision = check_gpu(
        args.gpu, args.memory_budget_mib, ROOT / "configs/gpu_safety.json"
    )
    print(json.dumps({"gpu_preflight": decision.as_dict()}), flush=True)
    if not decision.allowed:
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = "cuda:0"
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    exposed_mib = torch.cuda.get_device_properties(0).total_memory / 1024**2
    torch.cuda.set_per_process_memory_fraction(
        min(0.95, args.memory_budget_mib / exposed_mib), device=0
    )
    torch.cuda.reset_peak_memory_stats(0)

    manifest = resolve_path(args.train_manifest, args.project_root)
    rows = read_qa_jsonl(manifest)
    if args.max_train_rows > 0:
        rows = rows[: args.max_train_rows]
    target_micro_steps, optimizer_steps = resolve_training_schedule(
        selected_rows=len(rows),
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    selector_checkpoint = (
        resolve_path(args.selector_checkpoint, args.project_root)
        if args.selector_checkpoint is not None
        else None
    )
    bridge_checkpoint = (
        resolve_path(args.bridge_checkpoint, args.project_root)
        if args.bridge_checkpoint is not None
        else None
    )
    protocol = {
        "format_version": FORMAT_VERSION,
        "variant": args.variant,
        "variant_config": variant,
        "system_prompt": SYSTEM_PROMPT,
        "train_manifest": str(manifest),
        "train_manifest_sha256": sha256_file(manifest),
        "selected_rows": len(rows),
        "num_epochs": args.num_epochs,
        "target_micro_steps": target_micro_steps,
        "optimizer_steps": optimizer_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "model": str(args.model),
        "selector_checkpoint": str(selector_checkpoint) if selector_checkpoint else None,
        "selector_sha256": sha256_file(selector_checkpoint) if selector_checkpoint else None,
        "bridge_checkpoint": str(bridge_checkpoint) if bridge_checkpoint else None,
        "bridge_sha256": sha256_file(bridge_checkpoint) if bridge_checkpoint else None,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "optimizer": "torch.optim.AdamW",
        "optimizer_betas": (
            list(RECOVERY_ADAMW_BETAS) if args.recovery_lane else [0.9, 0.999]
        ),
        "optimizer_eps": RECOVERY_ADAMW_EPS,
        "scheduler": args.scheduler_type,
        "epoch_order": (
            RECOVERY_EPOCH_ORDER_VERSION if args.recovery_lane else "legacy_shuffle"
        ),
        "answer_protocol": args.answer_protocol,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "minimum_region_tokens": args.minimum_region_tokens,
        "insertion_block": args.insertion_block,
        "bridge_dim": args.bridge_dim,
        "bridge_heads": args.bridge_heads,
        "neighborhood_radius": args.neighborhood_radius,
        "max_relative_residual": args.max_relative_residual,
        "coordinate_bins": args.coordinate_bins,
        "scale_bins": args.scale_bins,
        "projection_dim": args.projection_dim,
        "projection_seed": args.projection_seed,
        "trainer_sha256": sha256_file(Path(__file__)),
        "shared_utilities_sha256": sha256_file(
            ROOT / "evivit_core/evivit_posttraining.py"
        ),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "model_index_sha256": (
            sha256_file(args.model / "model.safetensors.index.json")
            if (args.model / "model.safetensors.index.json").is_file()
            else None
        ),
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_pattern": LANGUAGE_LORA_TARGET_PATTERN,
        },
        "answer_only": True,
        "vision_frozen": True,
        "mid_ptea_frozen": True,
        "bridge_frozen": True,
    }
    fingerprint = protocol_fingerprint(protocol)
    protocol["protocol_fingerprint"] = fingerprint
    atomic_write_json(args.output_dir / "run_protocol.json", protocol)

    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import (
        AutoProcessor,
        Qwen3VLForConditionalGeneration,
        get_scheduler,
    )

    processor = AutoProcessor.from_pretrained(
        str(args.model),
        local_files_only=True,
        max_pixels=int(variant["max_pixels"]),
        min_pixels=int(variant.get("min_pixels", 4096)),
    )
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": device},
        local_files_only=True,
    )
    qwen.requires_grad_(False)
    qwen.config.use_cache = False
    if args.gradient_checkpointing:
        qwen.gradient_checkpointing_enable()
        qwen.enable_input_require_grads()
    if args.resume_from_checkpoint:
        model = PeftModel.from_pretrained(
            qwen, str(args.resume_from_checkpoint), is_trainable=True
        )
    else:
        if args.recovery_lane:
            # Match the E-L/E-LB adapter initialization exactly rather than
            # inheriting any RNG consumption from model construction.
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        model = get_peft_model(
            qwen,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=LANGUAGE_LORA_TARGET_PATTERN,
            ),
        )
    model.train()
    qwen = model.get_base_model()
    qwen.model.visual.eval()
    report = parameter_report(model)
    assert_language_lora_only(report)
    atomic_write_json(args.output_dir / "trainable_parameters.json", report)
    print(
        json.dumps(
            {
                "trainable_parameters": {
                    key: value for key, value in report.items() if key != "trainable_names"
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    evi_runtime = None
    if variant["family"] == "evivit":
        assert selector_checkpoint is not None and bridge_checkpoint is not None
        evi_runtime = build_evi_runtime(
            qwen=qwen,
            selector_checkpoint=selector_checkpoint,
            bridge_checkpoint=bridge_checkpoint,
            device=device,
            insertion_block=args.insertion_block,
            bridge_dim=args.bridge_dim,
            bridge_heads=args.bridge_heads,
            neighborhood_radius=args.neighborhood_radius,
            max_relative_residual=args.max_relative_residual,
            projection_dim=args.projection_dim,
            projection_seed=args.projection_seed,
            coordinate_bins=args.coordinate_bins,
            scale_bins=args.scale_bins,
        )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_kwargs: dict[str, Any] = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    if args.recovery_lane:
        optimizer_kwargs.update(
            betas=RECOVERY_ADAMW_BETAS,
            eps=RECOVERY_ADAMW_EPS,
        )
    optimizer = torch.optim.AdamW(trainable, **optimizer_kwargs)
    scheduler = get_scheduler(
        args.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps(optimizer_steps, args.warmup_ratio),
        num_training_steps=optimizer_steps,
    )
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    micro_step = 0
    row_index = 0
    next_epoch_to_save: float = 0.5 if args.recovery_lane else 1.0
    if args.resume_from_checkpoint:
        rows, counters = restore_resume_state(
            checkpoint=args.resume_from_checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
            rows=rows,
            device=device,
            target_micro_steps=target_micro_steps,
            optimizer_steps=optimizer_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fingerprint=fingerprint,
        )
        global_step = counters["global_step"]
        micro_step = counters["micro_step"]
        row_index = counters["row_index"]
        next_epoch_to_save = counters["next_epoch_to_save"]
        print(json.dumps({"resume_loaded": counters}, ensure_ascii=False), flush=True)

    step_log = args.output_dir / "train_metrics.jsonl"
    if args.resume_from_checkpoint is None:
        step_log.write_text("", encoding="utf-8")
    started = time.perf_counter()
    accumulation: dict[str, Any] = {
        "loss": 0.0,
        "token_accuracy": 0.0,
        "answer_tokens": 0,
        "input_tokens": 0,
        "visual_tokens": 0,
        "regions": 0,
        "row_ids": [],
    }
    epoch_orders: dict[int, list[int]] = {}
    window_size = 0
    while micro_step < target_micro_steps:
        if not accumulation["row_ids"]:
            window_size = micro_batches_in_optimizer_step(
                micro_step_before=micro_step,
                target_micro_steps=target_micro_steps,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                selected_rows=len(rows),
            )
        epoch = row_index // len(rows)
        offset = row_index % len(rows)
        if args.recovery_lane:
            order = epoch_orders.setdefault(
                epoch, epoch_order(len(rows), epoch, args.seed)
            )
            row = rows[order[offset]]
        else:
            if row_index > 0 and offset == 0:
                random.shuffle(rows)
            row = rows[offset]
        row_index += 1
        image_path = resolve_path(Path(str(row["image"])), args.project_root)
        with Image.open(image_path) as opened:
            source = opened.convert("RGB")
        if variant["family"] == "base":
            batch, labels, metrics = build_base_batch(
                processor=processor,
                image=source,
                question=str(row["question"]),
                answer=str(row["answer"]),
                image_token_id=int(qwen.config.image_token_id),
                device=device,
                answer_protocol=args.answer_protocol,
            )
            outputs = qwen.model(
                **{key: value for key, value in batch.items() if key != "labels"},
                use_cache=False,
            )
            loss, accuracy, answer_tokens = answer_loss_from_hidden(
                outputs.last_hidden_state, labels, qwen.lm_head
            )
            metrics["answer_tokens"] = answer_tokens
            regions = 0
            del batch, labels, outputs
        else:
            assert evi_runtime is not None
            vision, vision_metrics = encode_evivit_image(
                runtime=evi_runtime,
                qwen=qwen,
                processor=processor,
                source=source,
                question=str(row["question"]),
                variant=variant,
                device=device,
                minimum_region_tokens=args.minimum_region_tokens,
            )
            loss, accuracy, metrics = evivit_answer_loss(
                qwen=qwen,
                processor=processor,
                vision=vision,
                question=str(row["question"]),
                answer=str(row["answer"]),
                runtime=evi_runtime,
                device=device,
            )
            metrics.update(vision_metrics)
            regions = int(vision_metrics["regions"])
            del vision
        source.close()
        (loss / window_size).backward()
        micro_step += 1
        accumulation["loss"] += float(loss.detach().cpu())
        accumulation["token_accuracy"] += float(accuracy.detach().cpu())
        accumulation["answer_tokens"] += int(metrics["answer_tokens"])
        accumulation["input_tokens"] += int(metrics["input_tokens"])
        accumulation["visual_tokens"] += int(metrics["visual_tokens"])
        accumulation["regions"] += regions
        accumulation["row_ids"].append(str(row["id"]))
        del loss, accuracy, metrics

        if len(accumulation["row_ids"]) != window_size:
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        count = len(accumulation["row_ids"])
        epoch_progress = micro_step / len(rows)
        log = {
            "format_version": FORMAT_VERSION,
            "variant": args.variant,
            "optimizer_step": global_step,
            "micro_step": micro_step,
            "epoch_progress": round(epoch_progress, 6),
            "micro_batches": count,
            "row_ids": list(accumulation["row_ids"]),
            "loss": accumulation["loss"] / count,
            "answer_token_accuracy": accumulation["token_accuracy"] / count,
            "answer_tokens": accumulation["answer_tokens"],
            "input_tokens_mean": accumulation["input_tokens"] / count,
            "visual_tokens_mean": accumulation["visual_tokens"] / count,
            "regions_mean": accumulation["regions"] / count,
            "grad_norm": float(grad_norm.detach().cpu()),
            "learning_rate": scheduler.get_last_lr()[0],
            "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2,
            "gpu_snapshot": cuda_snapshot(args.gpu),
            "elapsed_seconds": time.perf_counter() - started,
        }
        append_jsonl(step_log, log)
        print(json.dumps({"train_step": log}, ensure_ascii=False), flush=True)

        epoch_saved = False
        while epoch_progress >= next_epoch_to_save:
            epoch_label = (
                f"{next_epoch_to_save:03.1f}".replace(".", "p")
                if args.recovery_lane
                else f"{int(next_epoch_to_save):02d}"
            )
            destination = (
                args.output_dir
                / f"checkpoint-epoch-{epoch_label}-step-{global_step:05d}"
            )
            save_adapter_atomic(model, destination)
            append_jsonl(
                step_log,
                {
                    "checkpoint_saved": {
                        "epoch": next_epoch_to_save,
                        "optimizer_step": global_step,
                        "micro_step": micro_step,
                        "path": str(destination),
                    }
                },
            )
            next_epoch_to_save += 0.5 if args.recovery_lane else 1.0
            epoch_saved = True
        if (
            epoch_saved
            or args.save_resume_steps > 0
            and global_step % args.save_resume_steps == 0
        ):
            resume_path = save_resume_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                output_dir=args.output_dir,
                rows=rows,
                global_step=global_step,
                micro_step=micro_step,
                row_index=row_index,
                next_epoch_to_save=next_epoch_to_save,
                target_micro_steps=target_micro_steps,
                optimizer_steps=optimizer_steps,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                fingerprint=fingerprint,
            )
            append_jsonl(
                step_log,
                {
                    "resume_checkpoint_saved": {
                        "optimizer_step": global_step,
                        "micro_step": micro_step,
                        "path": str(resume_path),
                    }
                },
            )
        if (
            args.stop_after_optimizer_steps > 0
            and global_step >= args.stop_after_optimizer_steps
            and micro_step < target_micro_steps
        ):
            resume_path = save_resume_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                output_dir=args.output_dir,
                rows=rows,
                global_step=global_step,
                micro_step=micro_step,
                row_index=row_index,
                next_epoch_to_save=next_epoch_to_save,
                target_micro_steps=target_micro_steps,
                optimizer_steps=optimizer_steps,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                fingerprint=fingerprint,
            )
            stopped = {
                "status": "intentionally_stopped_for_resume_smoke",
                "variant": args.variant,
                "optimizer_step": global_step,
                "micro_step": micro_step,
                "resume_checkpoint": str(resume_path),
            }
            atomic_write_json(args.output_dir / "train_report.json", stopped)
            print(json.dumps({"training_stopped": stopped}), flush=True)
            return 0
        accumulation = {
            "loss": 0.0,
            "token_accuracy": 0.0,
            "answer_tokens": 0,
            "input_tokens": 0,
            "visual_tokens": 0,
            "regions": 0,
            "row_ids": [],
        }

    if accumulation["row_ids"]:
        raise RuntimeError("training ended with unapplied accumulated gradients")
    if global_step != optimizer_steps:
        raise RuntimeError(
            f"optimizer-step mismatch: expected {optimizer_steps}, got {global_step}"
        )
    resume_path = save_resume_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        output_dir=args.output_dir,
        rows=rows,
        global_step=global_step,
        micro_step=micro_step,
        row_index=row_index,
        next_epoch_to_save=next_epoch_to_save,
        target_micro_steps=target_micro_steps,
        optimizer_steps=optimizer_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fingerprint=fingerprint,
    )
    report_out = {
        "status": "completed",
        "format_version": FORMAT_VERSION,
        "variant": args.variant,
        "protocol_fingerprint": fingerprint,
        "rows": len(rows),
        "epochs": args.num_epochs,
        "micro_steps": micro_step,
        "optimizer_steps": global_step,
        "resume_checkpoint": str(resume_path),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / 1024**2,
    }
    atomic_write_json(args.output_dir / "train_report.json", report_out)
    print(json.dumps({"training_complete": report_out}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
