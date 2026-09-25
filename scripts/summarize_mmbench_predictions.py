#!/usr/bin/env python3
"""Summarize MMBench predictions with strict CircularEval grouping."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ratio(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    manifest_by_id = {
        str(row["id"]): row for row in read_jsonl(args.manifest)
    }
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    row_correct = 0
    valid_json = 0
    token_total = 0.0
    token_rows = 0

    for row in rows:
        source = manifest_by_id[str(row["id"])]
        metadata = source.get("metadata") or {}
        base_index = metadata.get("base_index")
        if base_index is None:
            sample_id = str(row.get("id", ""))
            base_index = int(sample_id.rsplit("_", 1)[-1]) % 1_000_000
        groups[str(base_index)].append(row)
        row_correct += bool(row.get("exact_correct"))
        valid_json += not bool(row.get("parse_error"))
        visual_tokens = row.get(
            "visual_token_count",
            row.get("visual_tokens", row.get("total_image_tokens")),
        )
        if visual_tokens is not None:
            token_total += float(visual_tokens)
            token_rows += 1

    circular_correct = 0
    category_total: collections.Counter[str] = collections.Counter()
    category_correct: collections.Counter[str] = collections.Counter()
    l2_total: collections.Counter[str] = collections.Counter()
    l2_correct: collections.Counter[str] = collections.Counter()
    group_sizes: collections.Counter[int] = collections.Counter()

    for group_rows in groups.values():
        group_sizes[len(group_rows)] += 1
        base_row = min(
            group_rows,
            key=lambda row: int(
                (
                    manifest_by_id[str(row["id"])].get("metadata")
                    or {}
                ).get("circular_pass", 0)
            ),
        )
        base_source = manifest_by_id[str(base_row["id"])]
        category = str(base_source.get("category", "unknown"))
        l2_category = str(base_source.get("l2_category", "unknown"))
        correct = all(bool(row.get("exact_correct")) for row in group_rows)
        category_total[category] += 1
        l2_total[l2_category] += 1
        category_correct[category] += correct
        l2_correct[l2_category] += correct
        circular_correct += correct

    summary = {
        "format_version": "mmbench_circular_eval_summary_v1",
        "predictions": str(args.predictions),
        "protocol_rows": len(rows),
        "row_exact_correct": row_correct,
        "row_exact_accuracy": ratio(row_correct, len(rows)),
        "base_questions": len(groups),
        "circular_correct": circular_correct,
        "circular_accuracy": ratio(circular_correct, len(groups)),
        "valid_json_rate": ratio(valid_json, len(rows)),
        "mean_visual_tokens": ratio(round(token_total, 6), token_rows),
        "group_size_counts": dict(sorted(group_sizes.items())),
        "category": {
            key: {
                "correct": category_correct[key],
                "total": value,
                "accuracy": ratio(category_correct[key], value),
            }
            for key, value in category_total.most_common()
        },
        "l2_category": {
            key: {
                "correct": l2_correct[key],
                "total": value,
                "accuracy": ratio(l2_correct[key], value),
            }
            for key, value in l2_total.most_common()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
