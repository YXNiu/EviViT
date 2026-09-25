#!/usr/bin/env python3
"""Train the token-level Patch--Text Evidence Alignment (PTEA) map head."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_dense_trace_map import (  # noqa: E402
    compose_targets,
    load_features,
    read_jsonl,
    shape_batches,
    stable_fold,
    trace_loss,
)
from evivit_core.dense_evidence import (  # noqa: E402
    decode_top_boxes,
    map_mass,
    policy_iou,
    target_coverage,
)
from evivit_core.patch_text_evidence import PatchTextEvidenceHead  # noqa: E402


def load_question_tokens(path: Path) -> dict[str, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = {str(row["id"]): row for row in payload.get("records", [])}
    if not records:
        raise RuntimeError(f"no question-token records found in {path}")
    return records


def padded_text_batch(
    rows: list[dict[str, Any]],
    token_features: dict[str, dict[str, Any]],
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = [
        torch.as_tensor(token_features[str(row["id"])]["question_tokens"])
        for row in rows
    ]
    maximum = max(int(tensor.shape[0]) for tensor in tensors)
    dimension = int(tensors[0].shape[-1])
    output = torch.zeros(len(rows), maximum, dimension, dtype=torch.float16)
    mask = torch.zeros(len(rows), maximum, dtype=torch.bool)
    for index, tensor in enumerate(tensors):
        length = int(tensor.shape[0])
        output[index, :length] = tensor
        mask[index, :length] = True
    return output.to(device), mask.to(device)


def batch_inputs(
    batch: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    token_features: dict[str, dict[str, Any]],
    maps: Any,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    visual = torch.stack(
        [torch.as_tensor(features[str(row["id"])]["visual"]) for row in batch]
    ).to(device)
    text, text_mask = padded_text_batch(batch, token_features, device=device)
    height, width = visual.shape[1], visual.shape[2]
    targets, failed = zip(
        *[
            compose_targets(maps, int(row["label_index"]), height, width, device)
            for row in batch
        ]
    )
    return visual, text, text_mask, torch.stack(targets), torch.stack(failed)


@torch.inference_mode()
def evaluate(
    model: PatchTextEvidenceHead,
    rows: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    token_features: dict[str, dict[str, Any]],
    maps: Any,
    *,
    device: str,
    decode_scales: tuple[float, ...],
    decode_mass_power: float,
    decode_area_penalty: float,
    batch_size: int,
) -> dict[str, float | int]:
    model.eval()
    cosine_values: list[float] = []
    top1_coverage: list[float] = []
    top5_coverage: list[float] = []
    top1_any_hits: list[float] = []
    top1_half_hits: list[float] = []
    top1_full_hits: list[float] = []
    top5_any_hits: list[float] = []
    top5_half_hits: list[float] = []
    top5_full_hits: list[float] = []
    top1_iou: list[float] = []
    top1_last_zoom_coverage: list[float] = []
    top5_last_zoom_coverage: list[float] = []
    top1_last_zoom_iou: list[float] = []
    evidence_recall: list[float] = []
    oracle_recall: list[float] = []
    area_ratios: list[float] = []
    failed_mass: list[float] = []
    point_hits: list[float] = []
    for batch in shape_batches(rows, features, batch_size=batch_size, shuffle=False):
        visual, text, text_mask, targets, failed_only = batch_inputs(
            batch, features, token_features, maps, device=device
        )
        logits_batch = model(visual, text, text_mask)
        probability_batch = torch.softmax(logits_batch.flatten(1), dim=1).reshape_as(logits_batch)
        cosine_values.extend(
            float(value)
            for value in F.cosine_similarity(
                probability_batch.flatten(1), targets.flatten(1), dim=1
            ).cpu().tolist()
        )
        failed_mass.extend(
            float(value)
            for value in (probability_batch * failed_only).sum(dim=(1, 2)).cpu().tolist()
        )
        for row, probability in zip(batch, probability_batch):
            probability_np = probability.cpu().numpy()
            target_np = maps["evidence"][int(row["label_index"])]
            decoded = decode_top_boxes(
                probability_np,
                scales=decode_scales,
                topk=5,
                mass_power=decode_mass_power,
                area_penalty=decode_area_penalty,
            )
            oracle = decode_top_boxes(
                target_np,
                scales=decode_scales,
                topk=1,
                mass_power=decode_mass_power,
                area_penalty=decode_area_penalty,
            )
            if not decoded:
                continue
            best_box = decoded[0]["bbox"]
            target_boxes = list(row.get("final_boxes_policy") or [])
            if target_boxes:
                top1_value = max(target_coverage(best_box, box) for box in target_boxes)
                top5_value = max(
                    target_coverage(candidate["bbox"], box)
                    for candidate in decoded
                    for box in target_boxes
                )
                top1_coverage.append(top1_value)
                top5_coverage.append(top5_value)
                top1_any_hits.append(float(top1_value > 0.0))
                top1_half_hits.append(float(top1_value >= 0.5))
                top1_full_hits.append(float(top1_value >= 0.9))
                top5_any_hits.append(float(top5_value > 0.0))
                top5_half_hits.append(float(top5_value >= 0.5))
                top5_full_hits.append(float(top5_value >= 0.9))
                top1_iou.append(max(policy_iou(best_box, box) for box in target_boxes))
                point_index = int(torch.argmax(probability).item())
                py, px = divmod(point_index, probability.shape[1])
                point_box = [
                    px / probability.shape[1] * 1000,
                    py / probability.shape[0] * 1000,
                    (px + 1) / probability.shape[1] * 1000,
                    (py + 1) / probability.shape[0] * 1000,
                ]
                point_hits.append(float(any(target_coverage(point_box, box) > 0 for box in target_boxes)))
            last_zoom = row.get("last_zoom_box")
            if isinstance(last_zoom, list) and len(last_zoom) == 4:
                top1_last_zoom_coverage.append(target_coverage(best_box, last_zoom))
                top5_last_zoom_coverage.append(
                    max(target_coverage(candidate["bbox"], last_zoom) for candidate in decoded)
                )
                top1_last_zoom_iou.append(policy_iou(best_box, last_zoom))
            evidence_recall.append(map_mass(target_np, best_box))
            if oracle:
                oracle_recall.append(map_mass(target_np, oracle[0]["bbox"]))
            area_ratios.append(float(decoded[0]["area_ratio"]))
    return {
        "rows": len(rows),
        "mean_map_cosine": mean(cosine_values) if cosine_values else 0.0,
        "top1_final_coverage": mean(top1_coverage) if top1_coverage else 0.0,
        "top5_final_coverage": mean(top5_coverage) if top5_coverage else 0.0,
        "top1_final_any_hit": mean(top1_any_hits) if top1_any_hits else 0.0,
        "top1_final_hit_at_50": mean(top1_half_hits) if top1_half_hits else 0.0,
        "top1_final_hit_at_90": mean(top1_full_hits) if top1_full_hits else 0.0,
        "top5_final_any_hit": mean(top5_any_hits) if top5_any_hits else 0.0,
        "top5_final_hit_at_50": mean(top5_half_hits) if top5_half_hits else 0.0,
        "top5_final_hit_at_90": mean(top5_full_hits) if top5_full_hits else 0.0,
        "top1_final_iou": mean(top1_iou) if top1_iou else 0.0,
        "top1_last_zoom_coverage": (
            mean(top1_last_zoom_coverage) if top1_last_zoom_coverage else 0.0
        ),
        "top5_last_zoom_coverage": (
            mean(top5_last_zoom_coverage) if top5_last_zoom_coverage else 0.0
        ),
        "top1_last_zoom_iou": mean(top1_last_zoom_iou) if top1_last_zoom_iou else 0.0,
        "top1_point_in_final": mean(point_hits) if point_hits else 0.0,
        "top1_evidence_recall": mean(evidence_recall) if evidence_recall else 0.0,
        "oracle_evidence_recall": mean(oracle_recall) if oracle_recall else 0.0,
        "mean_area_ratio": mean(area_ratios) if area_ratios else 0.0,
        "mean_failed_branch_mass": mean(failed_mass) if failed_mass else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--question-tokens", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--text-layers", type=int, default=1)
    parser.add_argument("--text-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--cosine-weight", type=float, default=0.2)
    parser.add_argument("--failed-weight", type=float, default=0.1)
    parser.add_argument("--decode-scales", default="0.2,0.25,0.35,0.5,0.67")
    parser.add_argument("--decode-mass-power", type=float, default=0.75)
    parser.add_argument("--decode-area-penalty", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help=(
            "Initialize the PTEA model weights from an existing compatible "
            "checkpoint while resetting optimizer state."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_started_at_utc = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_jsonl(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    features = load_features(args.features_dir)
    token_features = load_question_tokens(args.question_tokens)
    missing_visual = [str(row["id"]) for row in rows if str(row["id"]) not in features]
    missing_text = [str(row["id"]) for row in rows if str(row["id"]) not in token_features]
    if missing_visual or missing_text:
        raise RuntimeError(
            f"missing visual={len(missing_visual)} text={len(missing_text)}; "
            f"examples={missing_visual[:2]} {missing_text[:2]}"
        )
    with np.load(args.maps) as payload:
        maps = {name: payload[name] for name in payload.files}
    if args.train_all:
        train_rows = list(rows)
        validation_rows = []
    else:
        if not 0 <= args.fold < args.folds:
            raise ValueError("fold must be in [0, folds)")
        train_rows = [row for row in rows if stable_fold(str(row["id"]), args.folds) != args.fold]
        validation_rows = [row for row in rows if stable_fold(str(row["id"]), args.folds) == args.fold]
    visual_input_dim = int(next(iter(features.values()))["visual"].shape[-1])
    text_input_dim = int(
        next(iter(token_features.values()))["question_tokens"].shape[-1]
    )
    model = PatchTextEvidenceHead(
        input_dim=None,
        visual_input_dim=visual_input_dim,
        text_input_dim=text_input_dim,
        hidden_dim=args.hidden_dim,
        text_layers=args.text_layers,
        text_heads=args.text_heads,
        dropout=args.dropout,
    ).to(args.device)
    warm_start_metadata: dict[str, Any] | None = None
    if args.warm_start is not None:
        payload = torch.load(
            args.warm_start,
            map_location="cpu",
            weights_only=False,
        )
        source_config = payload.get("model_config") or {}
        expected_config = {
            "visual_input_dim": visual_input_dim,
            "text_input_dim": text_input_dim,
            "hidden_dim": args.hidden_dim,
            "text_layers": args.text_layers,
            "text_heads": args.text_heads,
            "dropout": args.dropout,
        }
        mismatched = {
            key: (source_config.get(key), value)
            for key, value in expected_config.items()
            if source_config.get(key) != value
        }
        if mismatched:
            raise RuntimeError(
                f"incompatible PTEA warm-start model_config: {mismatched}"
            )
        model.load_state_dict(payload["model"], strict=True)
        warm_start_metadata = {
            "path": str(args.warm_start),
            "source_epoch": payload.get("epoch"),
            "optimizer_state_reused": False,
        }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    decode_scales = tuple(float(value) for value in args.decode_scales.split(",") if value)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    best_score = -math.inf
    history: list[dict[str, Any]] = []
    setup_elapsed_sec = time.perf_counter() - wall_start
    is_cuda = torch.device(args.device).type == "cuda"
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if is_cuda:
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()
        model.train()
        batches = shape_batches(train_rows, features, batch_size=args.batch_size, shuffle=True)
        losses: list[float] = []
        components = {"cross_entropy": [], "cosine_loss": [], "failed_overlap": []}
        for batch in batches:
            visual, text, text_mask, target, failed_only = batch_inputs(
                batch, features, token_features, maps, device=args.device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(visual, text, text_mask)
            loss, detail = trace_loss(
                logits,
                target,
                failed_only,
                cosine_weight=args.cosine_weight,
                failed_weight=args.failed_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for name, value in detail.items():
                components[name].append(value)
        validation = evaluate(
            model,
            validation_rows,
            features,
            token_features,
            maps,
            device=args.device,
            decode_scales=decode_scales,
            decode_mass_power=args.decode_mass_power,
            decode_area_penalty=args.decode_area_penalty,
            batch_size=args.batch_size,
        )
        if is_cuda:
            torch.cuda.synchronize()
        epoch_elapsed_sec = time.perf_counter() - epoch_start
        record = {
            "epoch": epoch,
            "epoch_elapsed_sec": epoch_elapsed_sec,
            "cumulative_training_elapsed_sec": time.perf_counter() - training_start,
            "train_loss": mean(losses),
            **{f"train_{name}": mean(values) for name, values in components.items()},
            "validation": validation,
        }
        if is_cuda:
            record.update(
                {
                    "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
                    "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
                }
            )
        history.append(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        checkpoint = args.output_dir / f"checkpoint_epoch_{epoch:02d}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "config": vars(args),
                "model_config": {
                    "visual_input_dim": visual_input_dim,
                    "text_input_dim": text_input_dim,
                    "hidden_dim": args.hidden_dim,
                    "text_layers": args.text_layers,
                    "text_heads": args.text_heads,
                    "dropout": args.dropout,
                },
                "validation": validation,
            },
            checkpoint,
        )
        score = float(validation["mean_map_cosine"]) + 0.5 * float(
            validation["top1_evidence_recall"]
        )
        if score > best_score:
            best_score = score
            shutil.copyfile(checkpoint, args.output_dir / "best.pt")
        print(json.dumps(record, ensure_ascii=False), flush=True)
    if is_cuda:
        torch.cuda.synchronize()
    training_elapsed_sec = time.perf_counter() - training_start
    wall_elapsed_sec = time.perf_counter() - wall_start
    summary = {
        "version": "patch_text_evidence_map_v2",
        "run_started_at_utc": run_started_at_utc,
        "run_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "setup_elapsed_sec": setup_elapsed_sec,
        "training_elapsed_sec": training_elapsed_sec,
        "wall_elapsed_sec": wall_elapsed_sec,
        "timing_scope": {
            "setup": "data, cached-feature, target, model, and optimizer initialization",
            "training": "all training/evaluation epochs with CUDA synchronization at boundaries",
            "wall": "setup plus training and checkpoint serialization; excludes Python import time",
        },
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "fold": args.fold,
        "folds": args.folds,
        "best_score": best_score,
        "visual_input_dim": visual_input_dim,
        "text_input_dim": text_input_dim,
        "history": history,
        "config": vars(args),
        "warm_start": warm_start_metadata,
    }
    if is_cuda:
        summary.update(
            {
                "gpu_device_name": torch.cuda.get_device_name(),
                "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
                "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
            }
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
