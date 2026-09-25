#!/usr/bin/env python3
"""Audit v26's analytic budget against human Trace1144 evidence targets.

No external QA answer is read.  The target asks how the residual fine-token
    budget should be split between a second committed-evidence mode and a
complementary exploration/ambiguity region.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_dense_trace_map import (  # noqa: E402
    load_features,
    read_jsonl,
    shape_batches,
)
from scripts.train_evivit_vnext_h2_dual_map import (  # noqa: E402
    context_target_array,
    union_map_mass,
)
from scripts.train_patch_text_evidence_map import (  # noqa: E402
    load_question_tokens,
    padded_text_batch,
)
from evivit_core.dense_evidence import map_mass  # noqa: E402
from evivit_core.evivit_v3_online import allocate_trace_split_evidence  # noqa: E402
from evivit_core.evisplit_context_need import (  # noqa: E402
    FEATURE_NAMES,
    context_need_feature_tensor,
)
from evivit_core.patch_text_evidence import DualPatchTextEvidenceHead  # noqa: E402


def target_context_fraction(
    second_committed_mass: float,
    residual_context_gain: float,
    *,
    minimum: float = 0.10,
    maximum: float = 0.35,
) -> float:
    """Convert two human-evidence utilities into an exact bounded quota."""

    second = max(0.0, float(second_committed_mass))
    context = max(0.0, float(residual_context_gain))
    denominator = second + context
    share = context / denominator if denominator > 0 else 0.5
    return minimum + (maximum - minimum) * share


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def pearson(left: list[float], right: list[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_model(path: Path, device: str) -> DualPatchTextEvidenceHead:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    model = DualPatchTextEvidenceHead(
        input_dim=None,
        visual_input_dim=int(config["visual_input_dim"]),
        text_input_dim=int(config["text_input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        text_layers=int(config["text_layers"]),
        text_heads=int(config["text_heads"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def audit(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(args.manifest)
    if len(rows) != args.expected_rows or args.expected_rows != 1144:
        raise ValueError("formal context-need audit requires all 1144 rows")
    features = load_features(args.features_dir)
    questions = load_question_tokens(args.question_tokens)
    with np.load(args.maps) as payload:
        required = {"terminal", "committed", "exploration", "ambiguity"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"missing trace maps: {sorted(missing)}")
        maps = {name: payload[name] for name in payload.files}
    model = load_model(args.selector_checkpoint, args.device)
    records: list[dict[str, Any]] = []
    for batch in shape_batches(
        rows, features, batch_size=args.batch_size, shuffle=False
    ):
        visual = torch.stack(
            [
                torch.as_tensor(features[str(row["id"])]["visual"])
                for row in batch
            ]
        ).to(args.device)
        text, text_mask = padded_text_batch(
            batch, questions, device=args.device
        )
        decisive_logits, context_logits = model(visual, text, text_mask)
        decisive_batch = torch.softmax(
            decisive_logits.flatten(1), dim=1
        ).reshape_as(decisive_logits)
        context_batch = torch.softmax(
            context_logits.flatten(1), dim=1
        ).reshape_as(context_logits)
        for row, decisive, context in zip(
            batch, decisive_batch, context_batch
        ):
            allocation = allocate_trace_split_evidence(
                decisive,
                context,
                fine_token_budget=3072,
                max_regions=3,
                minimum_region_tokens=64,
                budget_mode="competitive",
                min_context_fraction=0.10,
                max_context_fraction=0.35,
            )
            decisive_plans = [
                plan
                for plan in allocation.region_budgets
                if plan.role.startswith("tracesplit_decisive")
            ]
            context_plans = [
                plan
                for plan in allocation.region_budgets
                if plan.role.startswith("tracesplit_context")
            ]
            if len(decisive_plans) not in {1, 2} or len(context_plans) != 1:
                raise RuntimeError("adaptive EviSplit allocation contract failed")
            label_index = int(row["label_index"])
            committed = np.asarray(maps["committed"][label_index])
            broad = context_target_array(maps, label_index)
            decisive_boxes = [list(plan.bbox) for plan in decisive_plans]
            second_mass = (
                map_mass(committed, decisive_boxes[1])
                if len(decisive_boxes) == 2
                else 0.0
            )
            base_context = union_map_mass(broad, decisive_boxes)
            combined_context = union_map_mass(
                broad,
                [*decisive_boxes, list(context_plans[0].bbox)],
            )
            context_gain = max(0.0, combined_context - base_context)
            target_fraction = target_context_fraction(
                second_mass, context_gain
            )
            predicted_fraction = float(
                allocation.trace_split_context_fraction
            )
            feature_tensor = context_need_feature_tensor(
                decisive,
                context,
                decisive_boxes,
                secondary_decisive_share=float(
                    allocation.trace_split_secondary_decisive_share
                ),
            )
            ambiguity_active = bool(
                np.asarray(maps["ambiguity"][label_index]).sum() > 0
            )
            records.append(
                {
                    "id": str(row["id"]),
                    "predicted_context_fraction": predicted_fraction,
                    "target_context_fraction": target_fraction,
                    "absolute_error": abs(
                        predicted_fraction - target_fraction
                    ),
                    "secondary_decisive_share": float(
                        allocation.trace_split_secondary_decisive_share
                    ),
                    "second_committed_mass": float(second_mass),
                    "residual_context_gain": float(context_gain),
                    "ambiguity_active": ambiguity_active,
                    "decisive_modes": len(decisive_plans),
                    "features": {
                        name: float(value)
                        for name, value in zip(
                            FEATURE_NAMES, feature_tensor.cpu().tolist()
                        )
                    },
                }
            )
    predicted = [row["predicted_context_fraction"] for row in records]
    target = [row["target_context_fraction"] for row in records]
    errors = [row["absolute_error"] for row in records]
    by_ambiguity = {}
    for active in (False, True):
        subset = [row for row in records if row["ambiguity_active"] is active]
        by_ambiguity[str(active).lower()] = {
            "rows": len(subset),
            "mae": mean(row["absolute_error"] for row in subset),
            "pearson": pearson(
                [row["predicted_context_fraction"] for row in subset],
                [row["target_context_fraction"] for row in subset],
            ),
        }
    report = {
        "format_version": "evivit_context_need_full1144_audit_v1",
        "rows": len(records),
        "external_qa_read": False,
        "target_definition": (
            "bounded share of human residual context gain versus second "
            "committed-search-mode mass"
        ),
        "predicted_context_fraction": describe(predicted),
        "target_context_fraction": describe(target),
        "absolute_error": describe(errors),
        "predicted_target_pearson": pearson(predicted, target),
        "by_ambiguity": by_ambiguity,
        "single_decisive_rows": sum(
            row["decisive_modes"] == 1 for row in records
        ),
        "decision_hint": (
            "analytic_competition_is_trace_aligned"
            if pearson(predicted, target) >= 0.30 and median(errors) <= 0.08
            else "learn_context_need_scalar_on_trace1144"
        ),
    }
    return report, records


def render_markdown(report: dict[str, Any]) -> str:
    predicted = report["predicted_context_fraction"]
    target = report["target_context_fraction"]
    error = report["absolute_error"]
    return "\n".join(
        [
            "# EviSplit连续预算的人类轨迹一致性审计",
            "",
            f"- 完整训练轨迹：`{report['rows']}`；读取外部QA：`否`。",
            f"- 解析预算—人类目标Pearson：`{report['predicted_target_pearson']:.4f}`。",
            f"- MAE均值/中位数：`{error['mean']:.4f}/{error['median']:.4f}`。",
            f"- 单决定性模式：`{report['single_decisive_rows']}`。",
            f"- 诊断：`{report['decision_hint']}`。",
            "",
            "| Quantity | Mean | P25 | Median | P75 | Range |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Analytic quota | {predicted['mean']:.4f} | "
            f"{predicted['p25']:.4f} | {predicted['median']:.4f} | "
            f"{predicted['p75']:.4f} | {predicted['minimum']:.4f}–"
            f"{predicted['maximum']:.4f} |",
            f"| Human target | {target['mean']:.4f} | "
            f"{target['p25']:.4f} | {target['median']:.4f} | "
            f"{target['p75']:.4f} | {target['minimum']:.4f}–"
            f"{target['maximum']:.4f} |",
            "",
            "该审计只判断解析式配额是否与训练轨迹一致，不证明QA提升。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--question-tokens", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=1144)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report, records = audit(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_jsonl.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in records
        ),
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
