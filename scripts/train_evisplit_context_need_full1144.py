#!/usr/bin/env python3
"""Train the tiny EviNeed context-budget head on all Trace1144 rows.

Five-fold out-of-fold predictions are diagnostic only.  The deployed
checkpoint is always refit on all 1,144 human-trace records.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
import sys
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evivit_core.evisplit_context_need import (  # noqa: E402
    FEATURE_NAMES,
    ContextNeedHead,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.std() == 0 or right.std() == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    error = prediction - target
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "pearson": pearson(prediction, target),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
    }


def fit_head(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    hidden_dim: int,
    minimum_context_fraction: float,
    maximum_context_fraction: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[ContextNeedHead, list[dict[str, float | int]]]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = ContextNeedHead(
        hidden_dim=hidden_dim,
        minimum_context_fraction=minimum_context_fraction,
        maximum_context_fraction=maximum_context_fraction,
    )
    feature_mean = features.mean(dim=0)
    feature_scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    model.set_normalization(feature_mean, feature_scale)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = nn.functional.mse_loss(prediction, targets)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    return model.eval(), history


def parse_rows(
    rows: list[dict[str, Any]],
    *,
    expected_rows: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(rows) != expected_rows or expected_rows != 1144:
        raise ValueError("formal EviNeed training requires all 1,144 rows")
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != expected_rows:
        raise ValueError("EviNeed records must have 1,144 unique IDs")
    feature_rows = []
    for row in rows:
        values = row.get("features")
        if not isinstance(values, dict):
            raise ValueError(
                "context-need audit must be regenerated with feature records"
            )
        feature_rows.append([float(values[name]) for name in FEATURE_NAMES])
    features = torch.tensor(feature_rows, dtype=torch.float32)
    targets = torch.tensor(
        [float(row["target_context_fraction"]) for row in rows],
        dtype=torch.float32,
    )
    analytic = torch.tensor(
        [float(row["predicted_context_fraction"]) for row in rows],
        dtype=torch.float32,
    )
    if not torch.isfinite(features).all():
        raise ValueError("EviNeed features contain non-finite values")
    return ids, features, targets, analytic


def transform_targets(
    targets: torch.Tensor,
    *,
    clamp_max: float | None,
) -> torch.Tensor:
    """Apply a preregistered one-sided safety target without dropping rows."""

    if clamp_max is None:
        return targets
    if not 0.1 <= clamp_max <= 0.5:
        raise ValueError("target clamp maximum must lie in [0.1, 0.5]")
    return targets.clamp(max=float(clamp_max))


def train(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.records)
    ids, features, raw_targets, analytic = parse_rows(
        rows,
        expected_rows=args.expected_rows,
    )
    targets = transform_targets(
        raw_targets,
        clamp_max=args.target_clamp_max,
    )
    if not (
        0.1
        <= args.minimum_context_fraction
        <= args.maximum_context_fraction
        <= 0.5
    ):
        raise ValueError("context output range must lie in [0.1, 0.5]")
    if args.target_clamp_max is not None and (
        abs(args.target_clamp_max - args.maximum_context_fraction) > 1e-8
    ):
        raise ValueError(
            "target clamp maximum must equal the model output maximum"
        )
    permutation = np.random.default_rng(args.seed).permutation(len(rows))
    fold_assignments = np.empty(len(rows), dtype=np.int64)
    fold_assignments[permutation] = np.arange(len(rows)) % args.folds
    oof = torch.empty_like(targets)
    fold_reports = []
    for fold in range(args.folds):
        validation_mask = torch.from_numpy(fold_assignments == fold)
        training_mask = ~validation_mask
        model, history = fit_head(
            features[training_mask],
            targets[training_mask],
            hidden_dim=args.hidden_dim,
            minimum_context_fraction=args.minimum_context_fraction,
            maximum_context_fraction=args.maximum_context_fraction,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed + fold + 1,
        )
        with torch.inference_mode():
            oof[validation_mask] = model(features[validation_mask])
        fold_reports.append(
            {
                "fold": fold,
                "train_rows": int(training_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "final_loss": history[-1]["loss"],
            }
        )
    deployed, history = fit_head(
        features,
        targets,
        hidden_dim=args.hidden_dim,
        minimum_context_fraction=args.minimum_context_fraction,
        maximum_context_fraction=args.maximum_context_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    with torch.inference_mode():
        fitted = deployed(features)
    parameter_count = sum(
        parameter.numel() for parameter in deployed.parameters()
    )
    checkpoint = {
        "version": "evivit_context_need_head_v1",
        "model": deployed.state_dict(),
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "minimum_context_fraction": args.minimum_context_fraction,
            "maximum_context_fraction": args.maximum_context_fraction,
            "feature_names": list(FEATURE_NAMES),
        },
        "training": {
            "records": str(args.records),
            "rows": len(rows),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "loss": "mean_squared_error",
            "target_clamp_max": args.target_clamp_max,
            "final_refit_uses_all_rows": True,
            "optimizer": "AdamW",
            "optimizer_state": None,
            "rng_state": torch.get_rng_state(),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_dir / "context_need_head.pt")
    target_numpy = targets.numpy()
    fixed = np.full_like(target_numpy, 0.25)
    report = {
        "format_version": "evivit_context_need_full1144_train_v1",
        "rows": len(rows),
        "unique_rows": len(set(ids)),
        "deployed_refit_rows": len(rows),
        "oof_is_diagnostic_only": True,
        "feature_names": list(FEATURE_NAMES),
        "hidden_dim": args.hidden_dim,
        "trainable_parameters": parameter_count,
        "minimum_context_fraction": args.minimum_context_fraction,
        "maximum_context_fraction": args.maximum_context_fraction,
        "target_clamp_max": args.target_clamp_max,
        "target_transform": (
            "identity"
            if args.target_clamp_max is None
            else f"min(human_target,{args.target_clamp_max:.6f})"
        ),
        "original_human_target": {
            "mean": float(raw_targets.mean()),
            "maximum": float(raw_targets.max()),
            "clipped_rows": int(
                (raw_targets > targets + 1e-8).sum().item()
            ),
        },
        "fixed_025": regression_metrics(fixed, target_numpy),
        "analytic_v26": regression_metrics(
            analytic.numpy(),
            target_numpy,
        ),
        "learned_oof": regression_metrics(oof.numpy(), target_numpy),
        "learned_all1144_refit": regression_metrics(
            fitted.numpy(),
            target_numpy,
        ),
        "folds": fold_reports,
        "history": history,
        "checkpoint": str(args.output_dir / "context_need_head.pt"),
    }
    (args.output_dir / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    prediction_rows = [
        {
            "id": sample_id,
            "target_context_fraction": float(target),
            "analytic_context_fraction": float(analytic_value),
            "oof_context_fraction": float(oof_value),
            "deployed_context_fraction": float(fitted_value),
            "fold": int(fold),
        }
        for sample_id, target, analytic_value, oof_value, fitted_value, fold in zip(
            ids,
            targets,
            analytic,
            oof,
            fitted,
            fold_assignments,
        )
    ]
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in prediction_rows
        ),
        encoding="utf-8",
    )
    markdown = "\n".join(
        [
            "# EviNeed Trace1144训练报告",
            "",
            f"- 部署训练数据：完整`{len(rows)}`条；参数量：`{parameter_count}`。",
            "- 五折OOF仅诊断训练标签的可学习性；部署头重新使用全部1,144条训练。",
            "",
            "| Method | MAE ↓ | RMSE ↓ | Pearson ↑ |",
            "|---|---:|---:|---:|",
            f"| Fixed 0.25 | {report['fixed_025']['mae']:.4f} | "
            f"{report['fixed_025']['rmse']:.4f} | "
            f"{report['fixed_025']['pearson']:.4f} |",
            f"| Analytic v26 | {report['analytic_v26']['mae']:.4f} | "
            f"{report['analytic_v26']['rmse']:.4f} | "
            f"{report['analytic_v26']['pearson']:.4f} |",
            f"| EviNeed OOF | {report['learned_oof']['mae']:.4f} | "
            f"{report['learned_oof']['rmse']:.4f} | "
            f"{report['learned_oof']['pearson']:.4f} |",
            f"| EviNeed all-1144 refit | "
            f"{report['learned_all1144_refit']['mae']:.4f} | "
            f"{report['learned_all1144_refit']['rmse']:.4f} | "
            f"{report['learned_all1144_refit']['pearson']:.4f} |",
            "",
            "本报告只验证人类轨迹预算监督的可学习性；QA结论必须来自三个"
            "Trace1144 selector checkpoint的Full515与8B Judge。",
            "",
        ]
    )
    (args.output_dir / "train_report.md").write_text(
        markdown,
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=1144)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--minimum-context-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-context-fraction", type=float, default=0.35)
    parser.add_argument(
        "--target-clamp-max",
        type=float,
        help=(
            "Optional one-sided safety target. Every one of the 1,144 rows "
            "is retained; only targets above this bound are clipped."
        ),
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def main() -> int:
    report = train(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
