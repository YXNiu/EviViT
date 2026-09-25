#!/usr/bin/env python3
"""Five-fold box-supervised training for EviViT-v6 AdaptiveBox."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evivit_core.evivit_adaptive_box import (  # noqa: E402
    AdaptiveBoxConfig,
    AdaptiveBoxHead,
    apply_residual,
    box_area,
    generalized_iou,
    target_coverage,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_examples(paths: list[Path]) -> tuple[list[dict[str, Any]], int, bool]:
    records: list[dict[str, Any]] = []
    feature_dim = None
    seen: set[str] = set()
    qa_utility_mode: bool | None = None
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        version = payload.get("version")
        if version not in {
            "evivit_v6_adaptivebox_examples_v1",
            "evivit_v6_qa_utility_examples_v1",
            "evivit_v9_t2_adaptivebox_examples_v1",
        }:
            raise ValueError(f"unsupported examples: {path}")
        current_qa_mode = version == "evivit_v6_qa_utility_examples_v1"
        qa_utility_mode = current_qa_mode if qa_utility_mode is None else qa_utility_mode
        if current_qa_mode != qa_utility_mode:
            raise ValueError("cannot mix geometry and QA-utility example formats")
        current_dim = int(payload["feature_dim"])
        feature_dim = current_dim if feature_dim is None else feature_dim
        if current_dim != feature_dim:
            raise ValueError("feature dimensions differ between example files")
        for row in payload["records"]:
            if str(row["id"]) in seen:
                raise ValueError(f"duplicate sample id: {row['id']}")
            seen.add(str(row["id"]))
            records.append(row)
    if feature_dim is None or not records:
        raise RuntimeError("no AdaptiveBox examples")
    return records, feature_dim, bool(qa_utility_mode)


def flatten_examples(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    features = []
    anchors = []
    targets = []
    residuals = []
    matched = []
    weights = []
    sample_indices = []
    final_targets = []
    has_final = []
    anchor_positions = []
    for sample_index, row in enumerate(records):
        target_box = torch.as_tensor(row["target_box"], dtype=torch.float32)
        final = row.get("final_box")
        final_box = (
            torch.as_tensor(final, dtype=torch.float32)
            if final is not None
            else target_box
        )
        for anchor_position, anchor in enumerate(row["anchors"]):
            qa_mode = "qa_utility_weight" in anchor
            is_matched = bool(anchor.get("qa_utility_active", anchor["matched"]))
            correction_active = bool(anchor.get("correction_active", is_matched))
            anchor_box = torch.tensor(anchor["bbox"], dtype=torch.float32)
            training_target = anchor.get("training_target_box")
            features.append(torch.as_tensor(anchor["features"], dtype=torch.float32))
            anchors.append(anchor_box)
            targets.append(
                torch.as_tensor(training_target, dtype=torch.float32)
                if training_target is not None
                else (target_box if is_matched else anchor_box)
            )
            residuals.append(torch.as_tensor(anchor["target_residual"], dtype=torch.float32))
            matched.append(float(is_matched))
            if qa_mode:
                weights.append(float(anchor["qa_utility_weight"]))
            else:
                weights.append(
                    float(row["sample_weight"])
                    * (1.0 if correction_active else 0.10)
                )
            sample_indices.append(sample_index)
            final_targets.append(final_box)
            has_final.append(float(final is not None))
            anchor_positions.append(anchor_position)
    return {
        "features": torch.stack(features),
        "anchors": torch.stack(anchors),
        "targets": torch.stack(targets),
        "residuals": torch.stack(residuals),
        "matched": torch.tensor(matched, dtype=torch.float32),
        "weights": torch.tensor(weights, dtype=torch.float32),
        "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        "final_targets": torch.stack(final_targets),
        "has_final": torch.tensor(has_final, dtype=torch.float32),
        "anchor_positions": torch.tensor(anchor_positions, dtype=torch.long),
    }


def residual_summary(
    records: list[dict[str, Any]],
    predictions: dict[str, list[list[float]]],
) -> dict[str, float]:
    target_rows = []
    prediction_rows = []
    weights = []
    active = []
    for row in records:
        values = predictions[str(row["id"])]
        for anchor, prediction in zip(row["anchors"], values):
            target_rows.append(torch.as_tensor(anchor["target_residual"], dtype=torch.float32))
            prediction_rows.append(torch.as_tensor(prediction, dtype=torch.float32))
            weights.append(float(anchor.get("qa_utility_weight", 1.0)))
            active.append(float(bool(anchor.get("qa_utility_active", True))))
    target = torch.stack(target_rows)
    prediction = torch.stack(prediction_rows)
    weight = torch.tensor(weights).clamp_min(0)
    mask = torch.tensor(active) > 0
    squared = (prediction - target).square().mean(dim=-1)
    absolute = (prediction - target).abs().mean(dim=-1)
    normalizer = weight.sum().clamp_min(1e-8)
    zero_squared = target.square().mean(dim=-1)
    cosine = F.cosine_similarity(prediction[mask], target[mask], dim=-1) if mask.any() else torch.tensor([])
    return {
        "anchors": len(target),
        "active_anchors": int(mask.sum()),
        "weighted_residual_mse": float((squared * weight).sum() / normalizer),
        "weighted_residual_mae": float((absolute * weight).sum() / normalizer),
        "zero_baseline_weighted_mse": float((zero_squared * weight).sum() / normalizer),
        "active_direction_cosine": float(cosine.mean()) if len(cosine) else float("nan"),
    }


def adaptive_box_loss(
    model: AdaptiveBoxHead,
    batch: dict[str, torch.Tensor],
    index: torch.Tensor,
    *,
    giou_weight: float,
    coverage_weight: float,
    excess_area_weight: float,
    identity_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    features = batch["features"][index].to(next(model.parameters()).device)
    anchors = batch["anchors"][index].to(features.device)
    targets = batch["targets"][index].to(features.device)
    target_residuals = batch["residuals"][index].to(features.device)
    matched = batch["matched"][index].to(features.device)
    weights = batch["weights"][index].to(features.device)
    final_targets = batch["final_targets"][index].to(features.device)

    predicted_residuals = model(features)
    predicted_boxes = apply_residual(
        anchors,
        predicted_residuals,
        minimum_side=model.config.minimum_side,
        maximum_side=model.config.maximum_side,
    )
    regression = F.smooth_l1_loss(
        predicted_residuals, target_residuals, reduction="none", beta=0.15
    ).mean(dim=-1)
    giou = (1.0 - generalized_iou(predicted_boxes, targets)) * matched
    coverage = (1.0 - target_coverage(predicted_boxes, final_targets)) * matched
    target_area = box_area(targets)
    excess = (
        F.relu(box_area(predicted_boxes) / target_area.clamp_min(1e-8) - 1.8)
        .square()
        * matched
    )
    identity = predicted_residuals.square().mean(dim=-1) * (1.0 - matched)
    per_row = (
        regression
        + giou_weight * giou
        + coverage_weight * coverage
        + excess_area_weight * excess
        + identity_weight * identity
    )
    normalizer = weights.sum().clamp_min(1e-6)
    loss = (per_row * weights).sum() / normalizer
    return loss, {
        "regression": float((regression * weights).sum().detach().cpu() / normalizer.cpu()),
        "giou": float((giou * weights).sum().detach().cpu() / normalizer.cpu()),
        "coverage": float((coverage * weights).sum().detach().cpu() / normalizer.cpu()),
        "excess_area": float((excess * weights).sum().detach().cpu() / normalizer.cpu()),
        "identity": float((identity * weights).sum().detach().cpu() / normalizer.cpu()),
    }


def coverage_summary(
    records: list[dict[str, Any]],
    predictions: dict[str, list[list[float]]],
) -> dict[str, float]:
    fixed_target = []
    refined_target = []
    fixed_final = []
    refined_final = []
    fixed_area = []
    refined_area = []
    for row in records:
        anchors = torch.tensor(
            [anchor["bbox"] for anchor in row["anchors"]], dtype=torch.float32
        )
        residuals = torch.tensor(predictions[str(row["id"])], dtype=torch.float32)
        refined = apply_residual(anchors, residuals)
        target = torch.as_tensor(row["target_box"], dtype=torch.float32).repeat(
            len(row["anchors"]), 1
        )
        fixed_target.append(float(target_coverage(anchors, target).max()))
        refined_target.append(float(target_coverage(refined, target).max()))
        final = row.get("final_box")
        if final is not None:
            final_tensor = torch.as_tensor(final, dtype=torch.float32).repeat(
                len(row["anchors"]), 1
            )
            fixed_final.append(float(target_coverage(anchors, final_tensor).max()))
            refined_final.append(float(target_coverage(refined, final_tensor).max()))
        fixed_area.append(float(box_area(anchors).sum()))
        refined_area.append(float(box_area(refined).sum()))
    return {
        "samples": len(records),
        "fixed_target_coverage": statistics.mean(fixed_target),
        "refined_target_coverage": statistics.mean(refined_target),
        "target_coverage_delta": statistics.mean(refined_target) - statistics.mean(fixed_target),
        "fixed_target_coverage_ge_0_8": statistics.mean(value >= 0.8 for value in fixed_target),
        "refined_target_coverage_ge_0_8": statistics.mean(value >= 0.8 for value in refined_target),
        "fixed_final_coverage": statistics.mean(fixed_final) if fixed_final else float("nan"),
        "refined_final_coverage": statistics.mean(refined_final) if refined_final else float("nan"),
        "fixed_total_area": statistics.mean(fixed_area),
        "refined_total_area": statistics.mean(refined_area),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--maximum-center-factor", type=float, default=1.50)
    parser.add_argument("--maximum-log-scale", type=float, default=1.20)
    parser.add_argument("--giou-weight", type=float, default=2.0)
    parser.add_argument("--coverage-weight", type=float, default=0.5)
    parser.add_argument("--excess-area-weight", type=float, default=0.05)
    parser.add_argument("--identity-weight", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shuffle-training-targets", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    records, feature_dim, qa_utility_mode = load_examples(args.examples)
    flat = flatten_examples(records)
    human_indices = [
        index for index, row in enumerate(records)
        if not str(row["source"]).startswith("vlmr1")
    ]
    easy_indices = [index for index in range(len(records)) if index not in set(human_indices)]
    permutation = np.random.default_rng(args.seed).permutation(human_indices)
    fold_samples = [array.tolist() for array in np.array_split(permutation, args.folds)]
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_predictions: dict[str, list[list[float]]] = {}
    fold_reports = []
    best_epochs = []

    for fold, validation_samples in enumerate(fold_samples):
        validation_set = set(validation_samples)
        training_set = set(index for index in human_indices if index not in validation_set)
        training_set.update(easy_indices)
        train_rows = torch.nonzero(
            torch.tensor([int(value) in training_set for value in flat["sample_indices"]]),
            as_tuple=False,
        ).flatten()
        validation_rows = torch.nonzero(
            torch.tensor([int(value) in validation_set for value in flat["sample_indices"]]),
            as_tuple=False,
        ).flatten()
        training_flat = flat
        if args.shuffle_training_targets:
            training_flat = {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in flat.items()
            }
            shuffle_generator = torch.Generator().manual_seed(args.seed + 50000 + fold)
            for anchor_position in sorted(set(flat["anchor_positions"][train_rows].tolist())):
                role_rows = train_rows[
                    flat["anchor_positions"][train_rows] == anchor_position
                ]
                permutation = role_rows[
                    torch.randperm(len(role_rows), generator=shuffle_generator)
                ]
                for key in ("targets", "residuals", "matched", "weights"):
                    training_flat[key][role_rows] = flat[key][permutation]
        config = AdaptiveBoxConfig(
            input_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            bottleneck_dim=args.bottleneck_dim,
            maximum_center_factor=args.maximum_center_factor,
            maximum_absolute_log_scale=args.maximum_log_scale,
        )
        model = AdaptiveBoxHead(config).to(device)
        train_features = flat["features"][train_rows]
        model.set_normalization(
            train_features.mean(dim=0).to(device),
            train_features.std(dim=0, unbiased=False).clamp_min(1e-4).to(device),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        best_state = None
        best_epoch = 0
        best_value = float("inf")
        history = []
        generator = torch.Generator().manual_seed(args.seed + fold)
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = train_rows[torch.randperm(len(train_rows), generator=generator)]
            training_losses = []
            for start in range(0, len(order), args.batch_size):
                index = order[start : start + args.batch_size]
                loss, _ = adaptive_box_loss(
                    model, training_flat, index,
                    giou_weight=args.giou_weight,
                    coverage_weight=args.coverage_weight,
                    excess_area_weight=args.excess_area_weight,
                    identity_weight=args.identity_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                training_losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.inference_mode():
                validation_loss, components = adaptive_box_loss(
                    model, flat, validation_rows,
                    giou_weight=args.giou_weight,
                    coverage_weight=args.coverage_weight,
                    excess_area_weight=args.excess_area_weight,
                    identity_weight=args.identity_weight,
                )
            value = float(validation_loss.cpu())
            history.append({"epoch": epoch, "train_loss": statistics.mean(training_losses), "validation_loss": value, **components})
            if value < best_value:
                best_value = value
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_state is None:
            raise RuntimeError("no fold checkpoint selected")
        model.load_state_dict(best_state)
        model.eval()
        for sample_index in validation_samples:
            row = records[sample_index]
            features = torch.stack([
                torch.as_tensor(anchor["features"], dtype=torch.float32)
                for anchor in row["anchors"]
            ]).to(device)
            with torch.inference_mode():
                values = model(features).cpu().tolist()
            oof_predictions[str(row["id"])] = values
        fold_dir = args.output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": "evivit_v6_adaptivebox_head_v1",
                "state_dict": best_state,
                "model_config": config.to_dict(),
                "fold": fold,
                "best_epoch": best_epoch,
                "best_validation_loss": best_value,
            },
            fold_dir / "head.pt",
        )
        (fold_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        fold_reports.append({"fold": fold, "train_samples": len(training_set), "validation_samples": len(validation_set), "best_epoch": best_epoch, "best_validation_loss": best_value})
        best_epochs.append(best_epoch)

    human_records = [records[index] for index in human_indices]
    oof = coverage_summary(human_records, oof_predictions)
    oof_residual = residual_summary(human_records, oof_predictions)
    full_epochs = max(1, int(round(statistics.median(best_epochs))))
    config = AdaptiveBoxConfig(
        input_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        maximum_center_factor=args.maximum_center_factor,
        maximum_absolute_log_scale=args.maximum_log_scale,
    )
    model = AdaptiveBoxHead(config).to(device)
    model.set_normalization(
        flat["features"].mean(dim=0).to(device),
        flat["features"].std(dim=0, unbiased=False).clamp_min(1e-4).to(device),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    all_rows = torch.arange(len(flat["features"]))
    generator = torch.Generator().manual_seed(args.seed + 10000)
    full_history = []
    for epoch in range(1, full_epochs + 1):
        order = all_rows[torch.randperm(len(all_rows), generator=generator)]
        losses = []
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            loss, _ = adaptive_box_loss(
                model, flat, index,
                giou_weight=args.giou_weight,
                coverage_weight=args.coverage_weight,
                excess_area_weight=args.excess_area_weight,
                identity_weight=args.identity_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        full_history.append({"epoch": epoch, "loss": statistics.mean(losses)})
    checkpoint = {
        "version": "evivit_v6_adaptivebox_head_v1",
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": config.to_dict(),
        "training_config": vars(args),
        "full_training_epochs": full_epochs,
        "fold_best_epochs": best_epochs,
        "examples_sha256": {str(path): sha256(path) for path in args.examples},
    }
    torch.save(checkpoint, args.output_dir / "full_head.pt")
    report = {
        "version": "evivit_v6_adaptivebox_fivefold_v1",
        "samples": len(records),
        "human_samples": len(human_indices),
        "easy_samples": len(easy_indices),
        "feature_dim": feature_dim,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "folds": fold_reports,
        "full_training_epochs": full_epochs,
        "oof_human": oof,
        "oof_residual": oof_residual,
        "qa_utility_mode": qa_utility_mode,
        "shuffle_training_targets": args.shuffle_training_targets,
        "full_history": full_history,
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "TRAINING_COMPLETED").write_text("complete\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
