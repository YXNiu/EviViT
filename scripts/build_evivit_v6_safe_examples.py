#!/usr/bin/env python3
"""Convert direct last-zoom regression labels into safe corrective labels."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evivit_core.evivit_adaptive_box import residual_target, target_coverage


def union_box(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.minimum(first[:2], second[:2]), torch.maximum(first[2:], second[2:])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-threshold", type=float, default=0.95)
    args = parser.parse_args()
    if not 0 < args.coverage_threshold <= 1:
        raise ValueError("coverage threshold must lie in (0, 1]")

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    records = copy.deepcopy(payload["records"])
    active = 0
    role_counts = {0: 0, 1: 0, 2: 0}
    before = []
    safe_target_areas = []
    for row in records:
        anchors = torch.tensor([anchor["bbox"] for anchor in row["anchors"]], dtype=torch.float32)
        evidence = row.get("final_box")
        if evidence is None:
            evidence = row["target_box"]
        evidence = torch.as_tensor(evidence, dtype=torch.float32)
        repeated = evidence.repeat(len(anchors), 1)
        coverage = target_coverage(anchors, repeated)
        best = int(torch.argmax(coverage).item())
        best_coverage = float(coverage[best])
        before.append(best_coverage)
        correction = best_coverage < args.coverage_threshold
        if correction:
            active += 1
            role_counts[best] += 1
        desired_view = torch.as_tensor(row["target_box"], dtype=torch.float32)
        for index, anchor in enumerate(row["anchors"]):
            anchor_box = anchors[index]
            training_target = (
                union_box(anchor_box, desired_view)
                if correction and index == best
                else anchor_box
            )
            encoded = residual_target(anchor_box.unsqueeze(0), training_target.unsqueeze(0))[0]
            anchor["matched"] = bool(correction and index == best)
            anchor["correction_active"] = bool(correction and index == best)
            anchor["training_target_box"] = training_target
            anchor["target_residual"] = encoded
            if correction and index == best:
                safe_target_areas.append(float((training_target[2] - training_target[0]) * (training_target[3] - training_target[1])))
        row["safe_correction_active"] = correction
        row["fixed_evidence_coverage"] = best_coverage

    output_payload = {
        **payload,
        "version": payload.get("version", "evivit_v6_adaptivebox_examples_v1"),
        "records": records,
        "safe_target_config": {
            "mode": "coverage_gated_minimum_union",
            "coverage_threshold": args.coverage_threshold,
            "identity_when_covered": True,
            "never_shrink": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)
    summary = {
        "version": str(output_payload["version"]).replace(
            "_examples_v1", "_safe_examples_v1"
        ),
        "rows": len(records),
        "active_corrections": active,
        "identity_rows": len(records) - active,
        "active_fraction": active / len(records),
        "matched_role_counts": role_counts,
        "mean_fixed_evidence_coverage": sum(before) / len(before),
        "mean_safe_target_area": (
            sum(safe_target_areas) / len(safe_target_areas)
            if safe_target_areas else 0.0
        ),
        "coverage_threshold": args.coverage_threshold,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
