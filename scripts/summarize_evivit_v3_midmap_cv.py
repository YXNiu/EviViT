#!/usr/bin/env python3
"""Select one shared Mid-PTEA epoch using human-evidence-first metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import pstdev
from typing import Any


METRIC_WEIGHTS = {
    "mean_map_cosine": 0.30,
    "top1_final_coverage": 0.30,
    "top1_last_zoom_coverage": 0.25,
    "top5_final_hit_at_90": 0.15,
}


def evidence_score(metrics: dict[str, Any]) -> float:
    return sum(float(metrics[name]) * weight for name, weight in METRIC_WEIGHTS.items())


def weighted_metric(records: list[dict[str, Any]], key: str) -> float:
    total = sum(int(record["validation"]["rows"]) for record in records)
    if total <= 0:
        return 0.0
    return sum(
        float(record["validation"][key]) * int(record["validation"]["rows"])
        for record in records
    ) / total


def summarize(run_root: Path, folds: int) -> dict[str, Any]:
    fold_summaries = [
        json.loads((run_root / f"fold{fold}" / "summary.json").read_text())
        for fold in range(folds)
    ]
    histories = [summary["history"] for summary in fold_summaries]
    common_epochs = sorted(
        set.intersection(
            *[
                {int(record["epoch"]) for record in history}
                for history in histories
            ]
        )
    )
    if not common_epochs:
        raise RuntimeError("fold histories have no common epoch")
    numeric_keys = [
        key
        for key, value in histories[0][0]["validation"].items()
        if key != "rows" and isinstance(value, (int, float))
    ]
    epochs: list[dict[str, Any]] = []
    for epoch in common_epochs:
        records = [
            next(record for record in history if int(record["epoch"]) == epoch)
            for history in histories
        ]
        metrics = {key: weighted_metric(records, key) for key in numeric_keys}
        scores = [evidence_score(record["validation"]) for record in records]
        epochs.append(
            {
                "epoch": epoch,
                "evidence_score": evidence_score(metrics),
                "metrics_weighted": metrics,
                "fold_score_std": pstdev(scores),
            }
        )
    selected = max(
        epochs,
        key=lambda row: (
            float(row["evidence_score"]),
            float(row["metrics_weighted"]["top1_final_coverage"]),
            float(row["metrics_weighted"]["mean_map_cosine"]),
            -int(row["epoch"]),
        ),
    )
    dimensions = {
        (int(summary["visual_input_dim"]), int(summary["text_input_dim"]))
        for summary in fold_summaries
    }
    if len(dimensions) != 1:
        raise RuntimeError(f"fold input dimensions disagree: {sorted(dimensions)}")
    visual_dim, text_dim = next(iter(dimensions))
    return {
        "version": "evivit_v3_midmap_cv_shared_epoch_v1",
        "run_root": str(run_root),
        "folds": folds,
        "selection_weights": METRIC_WEIGHTS,
        "selection_rule": (
            "0.30 map cosine + 0.30 Top1 final-box coverage + 0.25 Top1 "
            "last-zoom coverage + 0.15 Top5 final-box >=90% hit"
        ),
        "visual_input_dim": visual_dim,
        "text_input_dim": text_dim,
        "selected_shared_epoch": int(selected["epoch"]),
        "selected": selected,
        "epochs": epochs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(args.run_root, args.folds)
    output = args.output or args.run_root / "shared_epoch_summary_v3.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
