#!/usr/bin/env python3
"""Train and cross-validate direct dense trace-map prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evivit_core.dense_evidence import decode_top_boxes, map_mass, policy_iou, target_coverage
from evivit_core.dense_models import FiLMDenseTraceHead


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_fold(row_id: str, folds: int) -> int:
    digest = hashlib.sha256(row_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def load_features(directory: Path) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("dense_features_shard_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for row in payload.get("records", []):
            features[str(row["id"])] = row
    if not features:
        raise RuntimeError(f"no dense feature shards found in {directory}")
    return features


def resize_distribution(array: np.ndarray, height: int, width: int, device: str) -> torch.Tensor:
    tensor = torch.as_tensor(array, dtype=torch.float32, device=device)[None, None]
    tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    tensor = tensor.clamp_min(0)
    return tensor / tensor.sum().clamp_min(1e-8)


def trace_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    failed_only: torch.Tensor,
    *,
    cosine_weight: float,
    failed_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
        target = target.unsqueeze(0)
        failed_only = failed_only.unsqueeze(0)
    log_probability = F.log_softmax(logits.flatten(1), dim=1)
    probability = log_probability.exp()
    target_flat = target.flatten(1)
    failed_flat = failed_only.flatten(1)
    cross_entropy = (-(target_flat * log_probability).sum(dim=1)).mean()
    cosine = (1.0 - F.cosine_similarity(probability, target_flat, dim=1)).mean()
    failed_overlap = (probability * failed_flat).sum(dim=1).mean()
    loss = cross_entropy + cosine_weight * cosine + failed_weight * failed_overlap
    return loss, {
        "cross_entropy": float(cross_entropy.detach().cpu()),
        "cosine_loss": float(cosine.detach().cpu()),
        "failed_overlap": float(failed_overlap.detach().cpu()),
    }


def compose_targets(
    maps: Any,
    label_index: int,
    height: int,
    width: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = resize_distribution(maps["evidence"][label_index], height, width, device)
    success = resize_distribution(maps["success_trace"][label_index], height, width, device)
    failed = resize_distribution(maps["failed_trace"][label_index], height, width, device)
    failed_only = (failed - success).clamp_min(0)
    failed_only = failed_only / failed_only.sum().clamp_min(1e-8)
    if float(maps["failed_trace"][label_index].sum()) <= 0:
        failed_only.zero_()
    return target, failed_only


def shape_batches(
    rows: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
) -> list[list[dict[str, Any]]]:
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        shape = tuple(int(value) for value in features[str(row["id"])]["feature_grid"])
        buckets.setdefault(shape, []).append(row)
    batches: list[list[dict[str, Any]]] = []
    for bucket_rows in buckets.values():
        if shuffle:
            random.shuffle(bucket_rows)
        for start in range(0, len(bucket_rows), batch_size):
            batches.append(bucket_rows[start : start + batch_size])
    if shuffle:
        random.shuffle(batches)
    return batches


def batch_inputs(
    batch: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    maps: Any,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_rows = [features[str(row["id"])] for row in batch]
    visual = torch.stack([torch.as_tensor(item["visual"]) for item in feature_rows]).to(device)
    question = torch.stack(
        [torch.as_tensor(item["question_feature"]) for item in feature_rows]
    ).to(device)
    height, width = visual.shape[1], visual.shape[2]
    targets, failed = zip(
        *[
            compose_targets(maps, int(row["label_index"]), height, width, device)
            for row in batch
        ]
    )
    return visual, question, torch.stack(targets), torch.stack(failed)


@torch.inference_mode()
def evaluate(
    model: FiLMDenseTraceHead,
    rows: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
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
        visual, question, targets, failed_only = batch_inputs(
            batch, features, maps, device=device
        )
        logits_batch = model(visual, question)
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
            assert isinstance(best_box, list)
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
                point_hits.append(
                    float(any(target_coverage(point_box, box) > 0 for box in target_boxes))
                )
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
    parser.add_argument(
        "--manifest", type=Path, default=Path("datasets/derived/dense_evigain/trace_labels_v1/manifest.jsonl")
    )
    parser.add_argument(
        "--maps", type=Path, default=Path("datasets/derived/dense_evigain/trace_labels_v1/dense_trace_maps.npz")
    )
    parser.add_argument(
        "--features-dir", type=Path, default=Path("datasets/derived/dense_evigain/qwen_dense_512px_p512_v1")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/dense_evigain/dense_trace_map_fold0_v1")
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--cosine-weight", type=float, default=0.2)
    parser.add_argument("--failed-weight", type=float, default=0.1)
    parser.add_argument("--decode-scales", default="0.2,0.25,0.35,0.5,0.67")
    parser.add_argument("--decode-mass-power", type=float, default=0.75)
    parser.add_argument("--decode-area-penalty", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_jsonl(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    features = load_features(args.features_dir)
    missing = [str(row["id"]) for row in rows if str(row["id"]) not in features]
    if missing:
        raise RuntimeError(f"missing {len(missing)} feature rows; first={missing[:5]}")
    with np.load(args.maps) as payload:
        maps = {name: payload[name] for name in payload.files}
    if not args.train_all and not 0 <= args.fold < args.folds:
        raise ValueError("fold must be in [0, folds)")
    if args.train_all:
        train_rows = list(rows)
        validation_rows = []
    else:
        train_rows = [row for row in rows if stable_fold(str(row["id"]), args.folds) != args.fold]
        validation_rows = [row for row in rows if stable_fold(str(row["id"]), args.folds) == args.fold]
    input_dim = int(next(iter(features.values()))["visual"].shape[-1])
    model = FiLMDenseTraceHead(input_dim, args.hidden_dim, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    decode_scales = tuple(float(value) for value in args.decode_scales.split(",") if value)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    best_score = -math.inf
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = shape_batches(
            train_rows, features, batch_size=args.batch_size, shuffle=True
        )
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        components: dict[str, list[float]] = {
            "cross_entropy": [], "cosine_loss": [], "failed_overlap": []
        }
        for index, batch in enumerate(batches, 1):
            visual, question, target, failed_only = batch_inputs(
                batch, features, maps, device=args.device
            )
            logits = model(visual, question)
            loss, detail = trace_loss(
                logits,
                target,
                failed_only,
                cosine_weight=args.cosine_weight,
                failed_weight=args.failed_weight,
            )
            (loss / args.gradient_accumulation).backward()
            if index % args.gradient_accumulation == 0 or index == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            for name, value in detail.items():
                components[name].append(value)
        validation = evaluate(
            model,
            validation_rows,
            features,
            maps,
            device=args.device,
            decode_scales=decode_scales,
            decode_mass_power=args.decode_mass_power,
            decode_area_penalty=args.decode_area_penalty,
            batch_size=args.batch_size,
        )
        record = {
            "epoch": epoch,
            "train_loss": mean(losses),
            **{f"train_{name}": mean(values) for name, values in components.items()},
            "validation": validation,
        }
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
                "validation": validation,
            },
            checkpoint,
        )
        score = float(validation["mean_map_cosine"]) + 0.5 * float(validation["top1_evidence_recall"])
        if score > best_score:
            best_score = score
            shutil.copyfile(checkpoint, args.output_dir / "best.pt")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    summary = {
        "version": "dense_trace_map_v1",
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "fold": args.fold,
        "folds": args.folds,
        "train_all": bool(args.train_all),
        "input_dim": input_dim,
        "history": history,
        "best_score": best_score,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
