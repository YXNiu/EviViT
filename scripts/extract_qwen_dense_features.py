#!/usr/bin/env python3
"""Encode each original image once and preserve its Qwen-ViT spatial grid.

Unlike the legacy crop extractor, this script never materializes or encodes the
128 proposal crops.  It writes resumable shards of compact projected visual
tokens and a question vector for each of the 1,144 trace samples.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evivit_core.gpu_safety import check_gpu
from evivit_core.evivit import grid_for_token_budget


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def clean_question(value: str) -> str:
    return " ".join(str(value).replace("<image>", " ").split())


def load_existing(output_dir: Path) -> tuple[set[str], int]:
    completed: set[str] = set()
    maximum_index = -1
    pattern = re.compile(r"dense_features_shard_(\d+)\.pt$")
    for path in sorted(output_dir.glob("dense_features_shard_*.pt")):
        match = pattern.search(path.name)
        if match:
            maximum_index = max(maximum_index, int(match.group(1)))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for row in payload.get("records", []):
            completed.add(str(row["id"]))
    return completed, maximum_index + 1


def resolve_image(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def resize_evivit_global(
    image: Image.Image,
    *,
    token_budget: int,
    merged_patch_stride: int,
) -> Image.Image:
    """Build the exact LANCZOS global view used by EviViT Stage 1."""

    grid_h, grid_w = grid_for_token_budget(image.width, image.height, token_budget)
    target = (grid_w * merged_patch_stride, grid_h * merged_patch_stride)
    if image.size == target:
        return image
    downsample = target[0] < image.width or target[1] < image.height
    return image.resize(
        target,
        Image.Resampling.LANCZOS if downsample else Image.Resampling.BICUBIC,
    )


def build_projection(input_dim: int, output_dim: int, seed: int, device: str) -> torch.Tensor:
    if output_dim <= 0 or output_dim > input_dim:
        raise ValueError(f"projection_dim must be in [1,{input_dim}], got {output_dim}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float32)
    matrix = F.normalize(matrix, dim=0)
    return matrix.to(device)


@torch.inference_mode()
def encode_questions(
    model: Any,
    tokenizer: Any,
    questions: list[str],
    projection: torch.Tensor,
    *,
    device: str,
    max_length: int,
) -> torch.Tensor:
    encoded = tokenizer(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    embeddings = model.get_input_embeddings()(encoded.input_ids).float()
    mask = encoded.attention_mask.unsqueeze(-1).float()
    pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return F.normalize(pooled @ projection, dim=-1)


def split_image_embeddings(
    image_embeds: Any,
    grid_rows: torch.Tensor,
    *,
    merge_size: int,
) -> list[torch.Tensor]:
    expected = [
        int(row[0]) * (int(row[1]) // merge_size) * (int(row[2]) // merge_size)
        for row in grid_rows
    ]
    if isinstance(image_embeds, torch.Tensor):
        if image_embeds.ndim == 3 and image_embeds.shape[0] == len(expected):
            return [image_embeds[index, :count] for index, count in enumerate(expected)]
        if image_embeds.ndim == 2 and int(image_embeds.shape[0]) == sum(expected):
            return list(image_embeds.split(expected, dim=0))
    rows = list(image_embeds)
    if len(rows) != len(expected):
        raise RuntimeError(f"image embedding count mismatch: got {len(rows)}, expected {len(expected)}")
    for index, (embedding, count) in enumerate(zip(rows, expected)):
        if int(embedding.shape[0]) != count:
            raise RuntimeError(
                f"image {index} token mismatch: got {tuple(embedding.shape)}, expected first dim {count}"
            )
    return rows


@torch.inference_mode()
def encode_batch(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    images: list[Image.Image],
    projection: torch.Tensor,
    *,
    device: str,
    merge_size: int,
    max_question_length: int,
) -> list[dict[str, Any]]:
    inputs = processor(
        text=["."] * len(images),
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)
    image_embeds, _ = model.model.get_image_features(
        inputs["pixel_values"], inputs["image_grid_thw"]
    )
    grid_rows = inputs["image_grid_thw"].detach().cpu()
    split_embeds = split_image_embeddings(image_embeds, grid_rows, merge_size=merge_size)
    question_features = encode_questions(
        model,
        processor.tokenizer,
        [clean_question(str(row["question"])) for row in rows],
        projection,
        device=device,
        max_length=max_question_length,
    )
    outputs: list[dict[str, Any]] = []
    for row, embedding, grid, question_feature in zip(
        rows, split_embeds, grid_rows.tolist(), question_features
    ):
        temporal, grid_h, grid_w = [int(value) for value in grid]
        merged_h, merged_w = grid_h // merge_size, grid_w // merge_size
        projected = F.normalize(embedding.float() @ projection, dim=-1)
        if temporal > 1:
            projected = projected.reshape(temporal, merged_h, merged_w, -1).mean(dim=0)
        else:
            projected = projected.reshape(merged_h, merged_w, -1)
        outputs.append(
            {
                "id": str(row["id"]),
                "label_index": int(row.get("label_index", -1)),
                "image": str(row["image"]),
                "question": clean_question(str(row["question"])),
                "grid_thw": [temporal, grid_h, grid_w],
                "feature_grid": [merged_h, merged_w],
                "visual": projected.to(torch.float16).cpu(),
                "question_feature": question_feature.to(torch.float16).cpu(),
            }
        )
    return outputs


def save_shard(
    output_dir: Path,
    shard_index: int,
    records: list[dict[str, Any]],
    meta: dict[str, Any],
) -> Path:
    path = output_dir / f"dense_features_shard_{shard_index:05d}.pt"
    torch.save({"records": records, "meta": {**meta, "records": len(records)}}, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/derived/dense_evigain/trace_labels_v1/manifest.jsonl"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen3-VL-4B-Instruct"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/derived/dense_evigain/qwen_dense_512px_p512_v1"),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--memory-budget-mib", type=int, default=28000)
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument("--min-pixels", type=int, default=64 * 64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--projection-seed", type=int, default=20260715)
    parser.add_argument("--max-question-length", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--resize-protocol",
        choices=("qwen_native", "evivit_lanczos"),
        default="qwen_native",
    )
    parser.add_argument("--global-token-budget", type=int, default=1024)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed, shard_index = (
        (set(), 0) if args.no_resume else load_existing(args.output_dir)
    )
    source_rows = read_jsonl(args.manifest, limit=args.limit)
    pending = [row for row in source_rows if str(row["id"]) not in completed]
    if not pending:
        print(json.dumps({"status": "complete", "existing_records": len(completed)}))
        return 0

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        str(args.model),
        local_files_only=True,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": device},
        local_files_only=True,
    )
    model.eval()
    merge_size = int(model.config.vision_config.spatial_merge_size)
    patch_size = int(model.config.vision_config.patch_size)
    merged_patch_stride = merge_size * patch_size
    hidden_size = int(model.config.text_config.hidden_size)
    projection = build_projection(hidden_size, args.projection_dim, args.projection_seed, device)

    started = time.time()
    buffer: list[dict[str, Any]] = []
    saved_paths: list[str] = []
    token_counts: list[int] = []
    processed = 0
    for start in range(0, len(pending), args.batch_size):
        batch_rows = pending[start : start + args.batch_size]
        source_images = [
            Image.open(resolve_image(args.project_root, str(row["image"]))).convert("RGB")
            for row in batch_rows
        ]
        images = (
            [
                resize_evivit_global(
                    image,
                    token_budget=args.global_token_budget,
                    merged_patch_stride=merged_patch_stride,
                )
                for image in source_images
            ]
            if args.resize_protocol == "evivit_lanczos"
            else source_images
        )
        encoded = encode_batch(
            model,
            processor,
            batch_rows,
            images,
            projection,
            device=device,
            merge_size=merge_size,
            max_question_length=args.max_question_length,
        )
        closed: set[int] = set()
        for image in [*images, *source_images]:
            identity = id(image)
            if identity not in closed:
                image.close()
                closed.add(identity)
        buffer.extend(encoded)
        token_counts.extend(int(row["visual"].shape[0] * row["visual"].shape[1]) for row in encoded)
        processed += len(encoded)
        while len(buffer) >= args.shard_size:
            shard_records = buffer[: args.shard_size]
            del buffer[: args.shard_size]
            path = save_shard(
                args.output_dir,
                shard_index,
                shard_records,
                {
                    "model": str(args.model),
                    "max_pixels": args.max_pixels,
                    "projection_dim": args.projection_dim,
                    "projection_seed": args.projection_seed,
                    "resize_protocol": args.resize_protocol,
                    "global_token_budget": args.global_token_budget,
                },
            )
            saved_paths.append(str(path))
            shard_index += 1
        if processed % max(args.batch_size, 20) == 0 or processed == len(pending):
            print(
                json.dumps(
                    {
                        "processed_this_run": processed,
                        "pending_total": len(pending),
                        "existing": len(completed),
                        "elapsed_sec": round(time.time() - started, 1),
                        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024**2, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    if buffer:
        path = save_shard(
            args.output_dir,
            shard_index,
            buffer,
            {
                "model": str(args.model),
                "max_pixels": args.max_pixels,
                "projection_dim": args.projection_dim,
                "projection_seed": args.projection_seed,
                "resize_protocol": args.resize_protocol,
                "global_token_budget": args.global_token_budget,
            },
        )
        saved_paths.append(str(path))

    elapsed = time.time() - started
    summary = {
        "version": "qwen_dense_features_v1",
        "source_rows": len(source_rows),
        "existing_records": len(completed),
        "processed_this_run": processed,
        "total_records_after_run": len(completed) + processed,
        "model": str(args.model),
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
        "batch_size": args.batch_size,
        "projection_dim": args.projection_dim,
        "projection_seed": args.projection_seed,
        "resize_protocol": args.resize_protocol,
        "global_token_budget": args.global_token_budget,
        "mean_visual_tokens": mean(token_counts) if token_counts else 0.0,
        "max_visual_tokens": max(token_counts) if token_counts else 0,
        "elapsed_sec": round(elapsed, 2),
        "samples_per_sec": processed / elapsed if elapsed > 0 else 0.0,
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024**2, 1),
        "saved_shards": saved_paths,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
