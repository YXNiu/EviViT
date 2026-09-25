#!/usr/bin/env python3
"""Cache channel-preserving Qwen-ViT mid-block grids for EviViT-v3.

One frozen global forward is paused at several block counts. Each pre-merger
state is pooled only across Qwen's native 2x2 spatial merge groups; channels are
not randomly projected. The resulting H/2 x W/2 x D grids let Mid-PTEA compare
Block 8/12/16 without repeatedly loading the 4B VLM or discarding mid-level
information.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.extract_qwen_dense_features import (  # noqa: E402
    clean_question,
    read_jsonl,
    resize_evivit_global,
    resolve_image,
)
from evivit_core.gpu_safety import check_gpu  # noqa: E402
from evivit_core.evivit_native_scale import (  # noqa: E402
    plan_continuous_native_global_view,
    plan_native_pixel_view,
)
from evivit_core.evivit_patch_exchange import (  # noqa: E402
    continuous_soft_floor_tokens,
    plan_patch_exchange_global_view,
)
from evivit_core.qwen3vl_evivit_v3 import (  # noqa: E402
    advance_segmented_qwen3vl_vision,
    prepare_segmented_qwen3vl_vision,
)
from evivit_core.qwen_family import load_qwen_vlm, vision_probe_text  # noqa: E402
from evivit_core.simple_evivit import plan_native_token_cap  # noqa: E402


def parse_stop_points(value: str) -> tuple[int, ...]:
    points = tuple(int(item) for item in value.split(",") if item.strip())
    if not points or tuple(sorted(set(points))) != points:
        raise ValueError("stop points must be unique and strictly increasing")
    return points


def qwen_merge_group_pool(
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    merge_size: int,
) -> list[torch.Tensor]:
    """Average each native Qwen merge group while preserving channel width."""

    outputs: list[torch.Tensor] = []
    offset = 0
    for temporal, height, width in grid_thw.detach().cpu().tolist():
        temporal, height, width = int(temporal), int(height), int(width)
        if height % merge_size or width % merge_size:
            raise ValueError(
                f"grid {(temporal, height, width)} is not divisible by {merge_size}"
            )
        count = temporal * height * width
        chunk = hidden_states[offset : offset + count]
        if int(chunk.shape[0]) != count:
            raise RuntimeError("hidden-state length does not match grid_thw")
        # Qwen orders tokens as t, merged-row, merged-column, intra-row,
        # intra-column before the Patch Merger.
        grouped = chunk.reshape(
            temporal,
            height // merge_size,
            width // merge_size,
            merge_size,
            merge_size,
            chunk.shape[-1],
        )
        pooled = grouped.float().mean(dim=(3, 4))
        if temporal > 1:
            pooled = pooled.mean(dim=0)
        else:
            pooled = pooled[0]
        outputs.append(pooled.to(torch.float16).cpu())
        offset += count
    if offset != int(hidden_states.shape[0]):
        raise RuntimeError(
            f"unused hidden states: consumed {offset}, total {hidden_states.shape[0]}"
        )
    return outputs


def load_completed(output_dir: Path) -> tuple[set[str], int]:
    completed: set[str] = set()
    maximum = -1
    pattern = re.compile(r"dense_features_shard_(\d+)\.pt$")
    for path in sorted(output_dir.glob("dense_features_shard_*.pt")):
        match = pattern.search(path.name)
        if match:
            maximum = max(maximum, int(match.group(1)))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        completed.update(str(row["id"]) for row in payload.get("records", []))
    return completed, maximum + 1


def save_shard(
    output_dir: Path,
    shard_index: int,
    records: list[dict[str, Any]],
    meta: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the established shard name so the frozen PTEA data loader can read
    # v3 features without a second, subtly different loading path.
    path = output_dir / f"dense_features_shard_{shard_index:05d}.pt"
    torch.save({"records": records, "meta": {**meta, "records": len(records)}}, path)
    return path


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stop-points", default="8,12,16")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--memory-budget-mib", type=int, default=32000)
    parser.add_argument("--max-pixels", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--min-pixels", type=int, default=4096)
    parser.add_argument("--global-token-budget", type=int, default=1024)
    parser.add_argument(
        "--native-token-caps",
        action="store_true",
        help=(
            "Interpret --global-token-budget as a ceiling. Preserve Qwen's "
            "native grid below the ceiling and cap only oversized images."
        ),
    )
    parser.add_argument(
        "--native-pixel-contract",
        action="store_true",
        help=(
            "Resize Global only through Qwen pixel minima/maxima. The output "
            "token grid is observed rather than forced to a target count."
        ),
    )
    parser.add_argument(
        "--global-processor-max-pixels",
        type=int,
        default=4 * 1024 * 1024,
    )
    parser.add_argument("--continuous-native-global", action="store_true")
    parser.add_argument(
        "--patch-exchange-global-scale",
        type=float,
        default=0.0,
        help=(
            "Enable EviViT-v5 Global feature extraction. The value is the "
            "effective patch scale relative to Qwen p16 (1.5 means p24)."
        ),
    )
    parser.add_argument(
        "--patch-exchange-soft-floor-tokens",
        type=int,
        default=0,
        help=(
            "Match EviViT-v5 Balanced SoftFloor Global geometry. Zero keeps "
            "the original PatchExchange Global view."
        ),
    )
    parser.add_argument(
        "--patch-exchange-balanced-soft-floor",
        action="store_true",
        help=(
            "Allocate the Global share of the continuous soft floor exactly "
            "as the formal v5 evaluator does."
        ),
    )
    parser.add_argument(
        "--patch-exchange-preserve-native-global",
        action="store_true",
        help=(
            "Match EviViT-v7 by never shrinking the PatchExchange Global "
            "view below Qwen's native processor grid."
        ),
    )
    parser.add_argument(
        "--patch-exchange-continuous-soft-floor",
        action="store_true",
        help=(
            "Match EviViT-v7/P3 by deriving a per-image soft floor from "
            "the source-native Qwen visual-token demand."
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.native_token_caps and args.native_pixel_contract:
        raise ValueError(
            "Native-Cap and Native-Pixel feature contracts are exclusive"
        )
    if (
        args.patch_exchange_preserve_native_global
        or args.patch_exchange_continuous_soft_floor
    ) and not args.patch_exchange_global_scale:
        raise ValueError(
            "v7 PatchExchange options require --patch-exchange-global-scale"
        )
    if (
        args.patch_exchange_preserve_native_global
        and args.patch_exchange_soft_floor_tokens <= 0
    ):
        raise ValueError(
            "preserving native Global requires a positive soft-floor token budget"
        )
    if args.patch_exchange_continuous_soft_floor and not (
        0
        <= args.patch_exchange_continuous_soft_floor_ramp_start_tokens
        < args.patch_exchange_continuous_soft_floor_ramp_end_tokens
    ):
        raise ValueError("continuous soft-floor ramp must be strictly increasing")
    if args.patch_exchange_global_scale < 0:
        raise ValueError("PatchExchange Global scale must be non-negative")
    if args.patch_exchange_soft_floor_tokens < 0:
        raise ValueError("PatchExchange soft floor must be non-negative")
    if args.patch_exchange_balanced_soft_floor and (
        not args.patch_exchange_global_scale
        or args.patch_exchange_soft_floor_tokens <= 0
    ):
        raise ValueError(
            "Balanced PatchExchange extraction requires a Global scale and "
            "a positive soft floor"
        )
    if args.patch_exchange_global_scale and (
        args.native_token_caps
        or args.native_pixel_contract
        or args.continuous_native_global
    ):
        raise ValueError(
            "PatchExchange Global features cannot mix with Native-Cap or "
            "Native-Pixel feature contracts"
        )
    if args.continuous_native_global and not args.native_pixel_contract:
        raise ValueError("continuous Native Global requires --native-pixel-contract")
    if not 0.0 <= args.continuous_global_exponent <= 1.0:
        raise ValueError("continuous Global exponent must lie in [0, 1]")
    if not (
        args.min_pixels
        <= args.continuous_global_base_pixels
        <= args.continuous_global_max_pixels
    ):
        raise ValueError("invalid continuous Global pixel interval")

    stop_points = parse_stop_points(args.stop_points)
    if args.memory_budget_mib > 0:
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
    torch.cuda.reset_peak_memory_stats(0)

    output_dirs = {
        point: args.output_root / f"block_{point:02d}" for point in stop_points
    }
    if args.no_resume:
        completed: set[str] = set()
        shard_index = 0
    else:
        states = {point: load_completed(path) for point, path in output_dirs.items()}
        completed_sets = [state[0] for state in states.values()]
        if completed_sets and any(
            value != completed_sets[0] for value in completed_sets[1:]
        ):
            raise RuntimeError(
                "mid-feature block directories have mismatched completed IDs; "
                "repair explicitly instead of silently duplicating records"
            )
        completed = completed_sets[0] if completed_sets else set()
        shard_index = max((state[1] for state in states.values()), default=0)

    rows = read_jsonl(args.manifest, limit=args.limit)
    pending = [row for row in rows if str(row["id"]) not in completed]
    if not pending:
        print(json.dumps({"status": "complete", "existing_records": len(completed)}))
        return 0

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(args.model),
        local_files_only=True,
        max_pixels=max(
            args.max_pixels,
            args.global_processor_max_pixels,
            (
                args.continuous_global_max_pixels
                if args.continuous_native_global
                else 0
            ),
        ),
        min_pixels=args.min_pixels,
    )
    model, model_type = load_qwen_vlm(args.model, device=device)
    model.eval()
    vision_model = model.model.visual
    merge_size = int(vision_model.spatial_merge_size)
    patch_size = int(model.config.vision_config.patch_size)
    merged_patch_stride = merge_size * patch_size
    hidden_size = int(vision_model.config.hidden_size)

    buffers: dict[int, list[dict[str, Any]]] = {
        point: [] for point in stop_points
    }
    started = time.time()
    processed = 0
    saved: list[str] = []
    for start in range(0, len(pending), args.batch_size):
        batch_rows = pending[start : start + args.batch_size]
        source_images = [
            Image.open(resolve_image(args.project_root, str(row["image"]))).convert("RGB")
            for row in batch_rows
        ]
        native_cap_plans = (
            [
                plan_native_token_cap(
                    image.width,
                    image.height,
                    token_cap=args.global_token_budget,
                    minimum_tokens=0,
                    minimum_pixels=args.min_pixels,
                    maximum_pixels=args.max_pixels,
                    patch_size=patch_size,
                    merge_size=merge_size,
                )
                for image in source_images
            ]
            if args.native_token_caps
            else [None] * len(source_images)
        )
        native_pixel_plans = []
        continuous_global_plans = []
        patch_exchange_global_plans = []
        for image in source_images:
            continuous = None
            pixel_plan = None
            if args.native_pixel_contract:
                if args.continuous_native_global:
                    continuous = plan_continuous_native_global_view(
                        image.width,
                        image.height,
                        minimum_pixels=args.min_pixels,
                        base_pixels=args.continuous_global_base_pixels,
                        maximum_pixels=args.continuous_global_max_pixels,
                        exponent=args.continuous_global_exponent,
                        patch_size=patch_size,
                        merge_size=merge_size,
                    )
                    pixel_plan = continuous.view_plan
                else:
                    pixel_plan = plan_native_pixel_view(
                        image.width,
                        image.height,
                        minimum_pixels=args.min_pixels,
                        maximum_pixels=args.global_processor_max_pixels,
                        patch_size=patch_size,
                        merge_size=merge_size,
                    )
            continuous_global_plans.append(continuous)
            native_pixel_plans.append(pixel_plan)
            exchange_plan = (
                plan_patch_exchange_global_view(
                    image.width,
                    image.height,
                    minimum_pixels=args.min_pixels,
                    maximum_pixels=args.max_pixels,
                    global_scale=args.patch_exchange_global_scale,
                    patch_size=patch_size,
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
                if args.patch_exchange_global_scale
                else None
            )
            if exchange_plan is not None and args.patch_exchange_continuous_soft_floor:
                effective_floor = continuous_soft_floor_tokens(
                    exchange_plan.native_plan.realized_tokens,
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
                exchange_plan = plan_patch_exchange_global_view(
                    image.width,
                    image.height,
                    minimum_pixels=args.min_pixels,
                    maximum_pixels=args.max_pixels,
                    global_scale=args.patch_exchange_global_scale,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    soft_floor_tokens=effective_floor,
                    soft_floor_global_share=(
                        1.0 / (args.patch_exchange_global_scale**2)
                    ),
                    preserve_native_global=(
                        args.patch_exchange_preserve_native_global
                    ),
                )
            patch_exchange_global_plans.append(exchange_plan)
        images = []
        for image, cap_plan, pixel_plan, exchange_plan in zip(
            source_images,
            native_cap_plans,
            native_pixel_plans,
            patch_exchange_global_plans,
        ):
            plan = cap_plan or pixel_plan or exchange_plan
            if plan is None:
                images.append(
                    resize_evivit_global(
                        image,
                        token_budget=args.global_token_budget,
                        merged_patch_stride=merged_patch_stride,
                    )
                )
                continue
            target = (
                plan.grid_width * merged_patch_stride,
                plan.grid_height * merged_patch_stride,
            )
            if image.size == target:
                images.append(image)
            else:
                downsample = target[0] < image.width or target[1] < image.height
                images.append(
                    image.resize(
                        target,
                        (
                            Image.Resampling.LANCZOS
                            if downsample
                            else Image.Resampling.BICUBIC
                        ),
                    )
                )
        processor_text = [vision_probe_text(processor, model_type)] * len(images)
        encoded = processor(
            text=processor_text,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)
        grid_thw = encoded["image_grid_thw"]
        grid_rows = [
            [int(value) for value in row]
            for row in grid_thw.detach().cpu().tolist()
        ]
        merged_tokens = [
            temporal * (height // merge_size) * (width // merge_size)
            for temporal, height, width in grid_rows
        ]
        if processed == 0 or processed % 10 == 0:
            print(
                json.dumps(
                    {
                        "batch_start": start,
                        "ids": [str(row["id"]) for row in batch_rows],
                        "source_sizes": [list(image.size) for image in source_images],
                        "resized_sizes": [list(image.size) for image in images],
                        "grid_thw": grid_rows,
                        "merged_tokens": merged_tokens,
                        "native_token_caps": args.native_token_caps,
                        "native_cap_plans": [
                            plan.to_dict() if plan is not None else None
                            for plan in native_cap_plans
                        ],
                        "native_pixel_plans": [
                            plan.to_dict() if plan is not None else None
                            for plan in native_pixel_plans
                        ],
                        "continuous_global_plans": [
                            plan.to_dict() if plan is not None else None
                            for plan in continuous_global_plans
                        ],
                        "patch_exchange_global_plans": [
                            plan.to_dict() if plan is not None else None
                            for plan in patch_exchange_global_plans
                        ],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        oversized = [
            (str(row["id"]), grid, tokens)
            for row, grid, tokens in zip(batch_rows, grid_rows, merged_tokens)
            if not args.native_pixel_contract
            and not args.patch_exchange_global_scale
            and tokens > args.global_token_budget
        ]
        if oversized:
            raise RuntimeError(
                "Qwen preprocessing exceeded the requested global token budget: "
                f"budget={args.global_token_budget}, rows={oversized}"
            )
        pixel_mismatches = [
            (str(row["id"]), tokens, plan.realized_tokens)
            for row, tokens, plan in zip(
                batch_rows, merged_tokens, native_pixel_plans
            )
            if plan is not None and tokens != plan.realized_tokens
        ]
        if pixel_mismatches:
            raise RuntimeError(
                "Qwen preprocessing differs from the Native-Pixel plan: "
                f"{pixel_mismatches}"
            )
        exchange_mismatches = [
            (str(row["id"]), tokens, plan.realized_tokens)
            for row, tokens, plan in zip(
                batch_rows, merged_tokens, patch_exchange_global_plans
            )
            if plan is not None and tokens != plan.realized_tokens
        ]
        if exchange_mismatches:
            raise RuntimeError(
                "Qwen preprocessing differs from the PatchExchange plan: "
                f"{exchange_mismatches}"
            )
        state = prepare_segmented_qwen3vl_vision(
            vision_model, encoded["pixel_values"], grid_thw
        )
        for point in stop_points:
            state = advance_segmented_qwen3vl_vision(
                vision_model, state, stop_after_blocks=point
            )
            pooled = qwen_merge_group_pool(
                state.hidden_states, grid_thw, merge_size=merge_size
            )
            for (
                row,
                visual,
                grid,
                native_cap_plan,
                native_pixel_plan,
                continuous_plan,
                patch_exchange_plan,
            ) in zip(
                batch_rows,
                pooled,
                grid_thw.detach().cpu().tolist(),
                native_cap_plans,
                native_pixel_plans,
                continuous_global_plans,
                patch_exchange_global_plans,
            ):
                temporal, height, width = [int(value) for value in grid]
                buffers[point].append(
                    {
                        "id": str(row["id"]),
                        "label_index": int(row.get("label_index", -1)),
                        "image": str(row["image"]),
                        "question": clean_question(str(row["question"])),
                        "grid_thw": [temporal, height, width],
                        "feature_grid": [
                            height // merge_size,
                            width // merge_size,
                        ],
                        "visual": visual,
                        "native_token_cap_plan": (
                            native_cap_plan.to_dict()
                            if native_cap_plan is not None
                            else None
                        ),
                        "native_pixel_plan": (
                            native_pixel_plan.to_dict()
                            if native_pixel_plan is not None
                            else None
                        ),
                        "continuous_global_plan": (
                            continuous_plan.to_dict()
                            if continuous_plan is not None
                            else None
                        ),
                        "patch_exchange_global_plan": (
                            patch_exchange_plan.to_dict()
                            if patch_exchange_plan is not None
                            else None
                        ),
                    }
                )
        closed: set[int] = set()
        for image in [*images, *source_images]:
            identity = id(image)
            if identity not in closed:
                image.close()
                closed.add(identity)
        processed += len(batch_rows)

        while all(len(buffer) >= args.shard_size for buffer in buffers.values()):
            for point, buffer in buffers.items():
                records = buffer[: args.shard_size]
                del buffer[: args.shard_size]
                path = save_shard(
                    output_dirs[point],
                    shard_index,
                    records,
                    {
                        "version": "evivit_v3_raw_mid_features_v1",
                        "block": point,
                        "hidden_size": hidden_size,
                        "global_token_budget": args.global_token_budget,
                        "global_budget_semantics": (
                            "patch_exchange_native_ceiling"
                            if args.patch_exchange_global_scale
                            else (
                                "continuous_native_pixel"
                                if args.continuous_native_global
                                else (
                                    "native_pixel"
                                    if args.native_pixel_contract
                                    else (
                                        "native_cap"
                                        if args.native_token_caps
                                        else "fixed_target"
                                    )
                                )
                            )
                        ),
                        "spatial_pool": f"qwen_native_{merge_size}x{merge_size}_mean",
                        "channel_projection": "none",
                        "patch_exchange_soft_floor_tokens": (
                            args.patch_exchange_soft_floor_tokens
                            if args.patch_exchange_global_scale
                            else None
                        ),
                        "patch_exchange_balanced_soft_floor": bool(
                            args.patch_exchange_balanced_soft_floor
                        ),
                    },
                )
                saved.append(str(path))
            shard_index += 1
        if processed % max(10, args.batch_size) == 0 or processed == len(pending):
            print(
                json.dumps(
                    {
                        "processed_this_run": processed,
                        "pending_total": len(pending),
                        "elapsed_sec": round(time.time() - started, 1),
                        "peak_reserved_mib": round(
                            torch.cuda.max_memory_reserved(0) / 1024**2, 1
                        ),
                    }
                ),
                flush=True,
            )

    remaining = len(next(iter(buffers.values())))
    if any(len(buffer) != remaining for buffer in buffers.values()):
        raise RuntimeError("block buffers diverged before final save")
    if remaining:
        for point, buffer in buffers.items():
            path = save_shard(
                output_dirs[point],
                shard_index,
                buffer,
                {
                    "version": "evivit_v3_raw_mid_features_v1",
                    "block": point,
                    "hidden_size": hidden_size,
                    "global_token_budget": args.global_token_budget,
                    "global_budget_semantics": (
                        "patch_exchange_native_ceiling"
                        if args.patch_exchange_global_scale
                        else (
                            "continuous_native_pixel"
                            if args.continuous_native_global
                            else (
                                "native_pixel"
                                if args.native_pixel_contract
                                else (
                                    "native_cap"
                                    if args.native_token_caps
                                    else "fixed_target"
                                )
                            )
                        )
                    ),
                    "spatial_pool": f"qwen_native_{merge_size}x{merge_size}_mean",
                    "channel_projection": "none",
                    "patch_exchange_soft_floor_tokens": (
                        args.patch_exchange_soft_floor_tokens
                        if args.patch_exchange_global_scale
                        else None
                    ),
                    "patch_exchange_balanced_soft_floor": bool(
                        args.patch_exchange_balanced_soft_floor
                    ),
                },
            )
            saved.append(str(path))

    summary = {
        "version": "evivit_v3_mid_feature_extract_v1",
        "manifest": str(args.manifest),
        "records_this_run": processed,
        "existing_records": len(completed),
        "stop_points": list(stop_points),
        "hidden_size": hidden_size,
        "global_token_budget": args.global_token_budget,
        "global_budget_semantics": (
            "patch_exchange_native_ceiling"
            if args.patch_exchange_global_scale
            else (
                "continuous_native_pixel"
                if args.continuous_native_global
                else (
                    "native_pixel"
                    if args.native_pixel_contract
                    else (
                        "native_cap" if args.native_token_caps else "fixed_target"
                    )
                )
            )
        ),
        "patch_exchange_global_scale": (
            args.patch_exchange_global_scale
            if args.patch_exchange_global_scale
            else None
        ),
        "patch_exchange_soft_floor_tokens": (
            args.patch_exchange_soft_floor_tokens
            if args.patch_exchange_global_scale
            else None
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
        "global_processor_max_pixels": args.global_processor_max_pixels,
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
        "elapsed_sec": round(time.time() - started, 2),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024**2, 1),
        "saved_shards": saved,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
