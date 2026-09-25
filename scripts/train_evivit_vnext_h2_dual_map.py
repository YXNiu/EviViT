#!/usr/bin/env python3
"""Train leakage-safe decisive/context Mid-PTEA heads on frozen Block16 features."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_dense_trace_map import (  # noqa: E402
    load_features,
    read_jsonl,
    resize_distribution,
    shape_batches,
    stable_fold,
)
from scripts.train_patch_text_evidence_map import (  # noqa: E402
    load_question_tokens,
    padded_text_batch,
)
from evivit_core.dense_evidence import (  # noqa: E402
    decode_top_boxes,
    map_mass,
    normalize_policy_box,
    policy_iou,
    target_coverage,
)
from evivit_core.patch_text_evidence import DualPatchTextEvidenceHead  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_array(array: np.ndarray) -> np.ndarray:
    result = np.clip(np.asarray(array, dtype=np.float32), 0.0, None)
    total = float(result.sum())
    return result / total if total > 0 else result


def context_target_array(maps: dict[str, np.ndarray], index: int) -> np.ndarray:
    exploration = normalize_array(maps["exploration"][index])
    ambiguity = normalize_array(maps["ambiguity"][index])
    if float(ambiguity.sum()) <= 0:
        return exploration
    return normalize_array(0.70 * exploration + 0.30 * ambiguity)


def distribution_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    log_probability = F.log_softmax(logits.flatten(1), dim=1)
    probability = log_probability.exp()
    target_flat = target.flatten(1)
    cross_entropy = (-(target_flat * log_probability).sum(dim=1)).mean()
    cosine = (1.0 - F.cosine_similarity(probability, target_flat, dim=1)).mean()
    return cross_entropy + cosine_weight * cosine, {
        "cross_entropy": float(cross_entropy.detach().cpu()),
        "cosine_loss": float(cosine.detach().cpu()),
    }


def batch_inputs(
    batch: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    question_tokens: dict[str, dict[str, Any]],
    maps: dict[str, np.ndarray],
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    visual = torch.stack(
        [torch.as_tensor(features[str(row["id"])]["visual"]) for row in batch]
    ).to(device)
    text, text_mask = padded_text_batch(batch, question_tokens, device=device)
    height, width = visual.shape[1:3]
    decisive = torch.stack(
        [
            resize_distribution(
                maps["terminal"][int(row["label_index"])],
                height,
                width,
                device,
            )
            for row in batch
        ]
    )
    context = torch.stack(
        [
            resize_distribution(
                context_target_array(maps, int(row["label_index"])),
                height,
                width,
                device,
            )
            for row in batch
        ]
    )
    return visual, text, text_mask, decisive, context


def normalized_entropy(probability: torch.Tensor) -> float:
    flat = probability.float().flatten().clamp_min(1e-12)
    flat = flat / flat.sum()
    return float((-(flat * flat.log()).sum() / math.log(flat.numel())).cpu())


def js_divergence(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().flatten().clamp_min(0)
    right = right.float().flatten().clamp_min(0)
    left = left / left.sum().clamp_min(1e-12)
    right = right / right.sum().clamp_min(1e-12)
    middle = 0.5 * (left + right)
    value = 0.5 * (
        torch.where(left > 0, left * (left / middle.clamp_min(1e-12)).log(), 0).sum()
        + torch.where(
            right > 0,
            right * (right / middle.clamp_min(1e-12)).log(),
            0,
        ).sum()
    ) / math.log(2.0)
    return float(value.cpu())


def box_mask(shape: tuple[int, int], boxes: Sequence[Sequence[float]]) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    for box in boxes:
        x1, y1, x2, y2 = normalize_policy_box(box)
        gx1 = min(width - 1, max(0, int(math.floor(x1 / 1000 * width))))
        gy1 = min(height - 1, max(0, int(math.floor(y1 / 1000 * height))))
        gx2 = min(width, max(gx1 + 1, int(math.ceil(x2 / 1000 * width))))
        gy2 = min(height, max(gy1 + 1, int(math.ceil(y2 / 1000 * height))))
        mask[gy1:gy2, gx1:gx2] = True
    return mask


def union_map_mass(heatmap: np.ndarray, boxes: Sequence[Sequence[float]]) -> float:
    distribution = normalize_array(heatmap)
    return float(distribution[box_mask(distribution.shape, boxes)].sum())


@torch.inference_mode()
def evaluate(
    model: DualPatchTextEvidenceHead,
    rows: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    question_tokens: dict[str, dict[str, Any]],
    maps: dict[str, np.ndarray],
    *,
    device: str,
    batch_size: int,
    decode_scales: tuple[float, ...],
    decode_mass_power: float,
    decode_area_penalty: float,
) -> dict[str, float | int]:
    model.eval()
    values: dict[str, list[float]] = {
        "decisive_map_cosine": [],
        "context_map_cosine": [],
        "predicted_map_js": [],
        "decisive_entropy": [],
        "context_entropy": [],
        "decisive_top1_final_coverage": [],
        "decisive_top1_last_zoom_coverage": [],
        "decisive_top2_terminal_recall": [],
        "decisive_top2_context_recall": [],
        "context_residual_target_recall": [],
        "context_residual_gain": [],
        "combined_terminal_recall": [],
        "context_box_max_iou_with_decisive": [],
    }
    residual_nonempty = 0
    for batch in shape_batches(rows, features, batch_size=batch_size, shuffle=False):
        visual, text, text_mask, decisive_target, context_target = batch_inputs(
            batch, features, question_tokens, maps, device=device
        )
        decisive_logits, context_logits = model(visual, text, text_mask)
        decisive_probability = torch.softmax(decisive_logits.flatten(1), dim=1).reshape_as(
            decisive_logits
        )
        context_probability = torch.softmax(context_logits.flatten(1), dim=1).reshape_as(
            context_logits
        )
        values["decisive_map_cosine"].extend(
            F.cosine_similarity(
                decisive_probability.flatten(1), decisive_target.flatten(1), dim=1
            ).cpu().tolist()
        )
        values["context_map_cosine"].extend(
            F.cosine_similarity(
                context_probability.flatten(1), context_target.flatten(1), dim=1
            ).cpu().tolist()
        )
        for row, decisive, context in zip(
            batch, decisive_probability, context_probability
        ):
            label_index = int(row["label_index"])
            decisive_np = decisive.cpu().numpy()
            context_np = context.cpu().numpy()
            terminal_target = normalize_array(maps["terminal"][label_index])
            broad_target = context_target_array(maps, label_index)
            values["predicted_map_js"].append(js_divergence(decisive, context))
            values["decisive_entropy"].append(normalized_entropy(decisive))
            values["context_entropy"].append(normalized_entropy(context))
            decisive_candidates = decode_top_boxes(
                decisive_np,
                scales=decode_scales,
                topk=2,
                mass_power=decode_mass_power,
                area_penalty=decode_area_penalty,
            )
            if not decisive_candidates:
                continue
            decisive_boxes = [list(candidate["bbox"]) for candidate in decisive_candidates]
            top1 = decisive_boxes[0]
            final_boxes = list(row.get("final_boxes_policy") or [])
            if final_boxes:
                values["decisive_top1_final_coverage"].append(
                    max(target_coverage(top1, box) for box in final_boxes)
                )
            last_zoom = row.get("last_zoom_box")
            if isinstance(last_zoom, list) and len(last_zoom) == 4:
                values["decisive_top1_last_zoom_coverage"].append(
                    target_coverage(top1, last_zoom)
                )
            decisive_terminal = union_map_mass(terminal_target, decisive_boxes)
            decisive_context = union_map_mass(broad_target, decisive_boxes)
            values["decisive_top2_terminal_recall"].append(decisive_terminal)
            values["decisive_top2_context_recall"].append(decisive_context)
            residual = context_np.copy()
            residual[box_mask(residual.shape, decisive_boxes)] = 0.0
            if float(residual.sum()) <= 0:
                continue
            residual /= residual.sum()
            context_candidates = decode_top_boxes(
                residual,
                scales=decode_scales,
                topk=1,
                mass_power=decode_mass_power,
                area_penalty=decode_area_penalty,
            )
            if not context_candidates:
                continue
            residual_nonempty += 1
            context_box = list(context_candidates[0]["bbox"])
            all_boxes = [*decisive_boxes, context_box]
            combined_context = union_map_mass(broad_target, all_boxes)
            values["context_residual_target_recall"].append(
                map_mass(broad_target, context_box)
            )
            values["context_residual_gain"].append(
                max(0.0, combined_context - decisive_context)
            )
            values["combined_terminal_recall"].append(
                union_map_mass(terminal_target, all_boxes)
            )
            values["context_box_max_iou_with_decisive"].append(
                max(policy_iou(context_box, box) for box in decisive_boxes)
            )
    result: dict[str, float | int] = {
        "rows": len(rows),
        "residual_context_nonempty": residual_nonempty,
        "residual_context_nonempty_rate": (
            residual_nonempty / len(rows) if rows else 0.0
        ),
    }
    for name, items in values.items():
        result[f"mean_{name}"] = mean(items) if items else 0.0
        result[f"median_{name}"] = median(items) if items else 0.0
    result["median_context_entropy_margin"] = (
        float(result["median_context_entropy"])
        - float(result["median_decisive_entropy"])
    )
    return result


def save_checkpoint(
    path: Path,
    *,
    model: DualPatchTextEvidenceHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "version": "evivit_vnext_h2_dual_ptea_v1",
            "selector_architecture": "dual_residual_context_ptea",
            "model_config": {
                "visual_input_dim": model.visual_input_dim,
                "text_input_dim": model.text_input_dim,
                "hidden_dim": model.hidden_dim,
                "text_layers": int(config["text_layers"]),
                "text_heads": int(config["text_heads"]),
                "dropout": float(config["dropout"]),
            },
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "validation": validation,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "python_rng_state": random.getstate(),
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--question-tokens", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train on all rows for a pre-registered fixed epoch; no validation selection.",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--text-layers", type=int, default=1)
    parser.add_argument("--text-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--cosine-weight", type=float, default=0.2)
    parser.add_argument("--context-loss-weight", type=float, default=0.5)
    parser.add_argument("--decode-scales", default="0.2,0.25,0.35,0.5,0.67")
    parser.add_argument("--decode-mass-power", type=float, default=0.75)
    parser.add_argument("--decode-area-penalty", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not 0 <= args.fold < args.folds:
        raise ValueError("fold must be in [0, folds)")
    if args.context_loss_weight <= 0:
        raise ValueError("context-loss-weight must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    rows = read_jsonl(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    features = load_features(args.features_dir)
    question_tokens = load_question_tokens(args.question_tokens)
    missing_visual = [str(row["id"]) for row in rows if str(row["id"]) not in features]
    missing_text = [str(row["id"]) for row in rows if str(row["id"]) not in question_tokens]
    if missing_visual or missing_text:
        raise RuntimeError(
            f"missing visual={len(missing_visual)} text={len(missing_text)}"
        )
    with np.load(args.maps) as payload:
        required = {"terminal", "exploration", "ambiguity"}
        missing = required.difference(payload.files)
        if missing:
            raise RuntimeError(f"missing map channels: {sorted(missing)}")
        maps = {name: payload[name] for name in payload.files}
    if args.train_all:
        train_rows = list(rows)
        validation_rows = []
    else:
        train_rows = [
            row for row in rows if stable_fold(str(row["id"]), args.folds) != args.fold
        ]
        validation_rows = [
            row for row in rows if stable_fold(str(row["id"]), args.folds) == args.fold
        ]
    if not train_rows or (not args.train_all and not validation_rows):
        raise RuntimeError("fold split produced an empty train or validation set")

    visual_input_dim = int(next(iter(features.values()))["visual"].shape[-1])
    text_input_dim = int(
        next(iter(question_tokens.values()))["question_tokens"].shape[-1]
    )
    model = DualPatchTextEvidenceHead(
        input_dim=None,
        visual_input_dim=visual_input_dim,
        text_input_dim=text_input_dim,
        hidden_dim=args.hidden_dim,
        text_layers=args.text_layers,
        text_heads=args.text_heads,
        dropout=args.dropout,
    ).to(args.device)
    if args.warm_start is not None:
        warm = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        if "model" not in warm:
            raise RuntimeError("warm-start is not a PTEA checkpoint")
        model.initialize_from_single(warm["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    decode_scales = tuple(float(value) for value in args.decode_scales.split(",") if value)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    if log_path.exists():
        raise RuntimeError(f"refusing to append existing log: {log_path}")
    config = {
        **vars(args),
        "format_version": "evivit_vnext_h2_dual_ptea_train_v1",
        "decisive_target": "maps['terminal']; provenance belongs to the frozen map artifact",
        "context_target": (
            "active-normalized 0.70*maps['exploration']+0.30*maps['ambiguity']; "
            "fallback exploration when ambiguity is empty; provenance belongs to the map artifact"
        ),
        "failed_or_reset_is_negative": False,
        "warm_start_sha256": (
            sha256(args.warm_start) if args.warm_start is not None else None
        ),
        "initialization": (
            "shared_random_seed" if args.warm_start is None else "single_head_warm_start"
        ),
    }
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        losses: list[float] = []
        decisive_losses: list[float] = []
        context_losses: list[float] = []
        for batch in shape_batches(
            train_rows, features, batch_size=args.batch_size, shuffle=True
        ):
            visual, text, text_mask, decisive_target, context_target = batch_inputs(
                batch, features, question_tokens, maps, device=args.device
            )
            optimizer.zero_grad(set_to_none=True)
            decisive_logits, context_logits = model(visual, text, text_mask)
            decisive_loss, _ = distribution_loss(
                decisive_logits, decisive_target, cosine_weight=args.cosine_weight
            )
            context_loss, _ = distribution_loss(
                context_logits, context_target, cosine_weight=args.cosine_weight
            )
            loss = decisive_loss + args.context_loss_weight * context_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            decisive_losses.append(float(decisive_loss.detach().cpu()))
            context_losses.append(float(context_loss.detach().cpu()))
        validation = evaluate(
            model,
            validation_rows,
            features,
            question_tokens,
            maps,
            device=args.device,
            batch_size=args.batch_size,
            decode_scales=decode_scales,
            decode_mass_power=args.decode_mass_power,
            decode_area_penalty=args.decode_area_penalty,
        )
        record = {
            "epoch": epoch,
            "train_loss": mean(losses),
            "train_decisive_loss": mean(decisive_losses),
            "train_context_loss": mean(context_losses),
            "epoch_elapsed_sec": time.perf_counter() - epoch_started,
            "validation": validation,
        }
        history.append(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        checkpoint = args.output_dir / f"checkpoint_epoch_{epoch:02d}.pt"
        save_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            validation=validation,
        )
        score = (
            0.30 * float(validation["mean_decisive_map_cosine"])
            + 0.20 * float(validation["mean_context_map_cosine"])
            + 0.20 * float(validation["mean_decisive_top1_final_coverage"])
            + 0.15 * float(validation["mean_decisive_top1_last_zoom_coverage"])
            + 0.15 * float(validation["mean_context_residual_gain"])
        )
        if score > best_score:
            best_score = score
            shutil.copyfile(checkpoint, args.output_dir / "best.pt")
        print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = {
        "format_version": "evivit_vnext_h2_dual_ptea_summary_v1",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "fold": args.fold,
        "folds": args.folds,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "visual_input_dim": visual_input_dim,
        "text_input_dim": text_input_dim,
        "best_score": best_score,
        "elapsed_sec": time.perf_counter() - started,
        "peak_gpu_mib": (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.device(args.device).type == "cuda"
            else 0.0
        ),
        "history": history,
        "config": config,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
