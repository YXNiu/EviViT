#!/usr/bin/env python3
"""Build hierarchical context/read/core/recovery labels for EviViT-v16.

The output deliberately keeps four spatial roles separate:

* context: successful-branch hover/dwell/intermediate search evidence;
* read: the last human zoom, normally large enough to read the answer;
* core: the final manually drawn answer region;
* failed/recovery: a question-conditional contrastive pair around the last reset.

A pre-reset region is never declared a universal visual negative.  It is only a
lower-preference branch for the same image/question when a valid recovery branch
exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evivit_core.dense_evidence import (
    build_trace_maps,
    map_mass,
    normalize_distribution,
    policy_iou,
    target_coverage,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in materialized:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(materialized)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_entropy(array: np.ndarray) -> float:
    distribution = normalize_distribution(array).reshape(-1)
    positive = distribution[distribution > 0]
    if positive.size <= 1:
        return 0.0
    entropy = -float(np.sum(positive * np.log(positive + 1e-12)))
    return entropy / math.log(distribution.size)


def jensen_shannon(first: np.ndarray, second: np.ndarray) -> float:
    p = normalize_distribution(first).reshape(-1)
    q = normalize_distribution(second).reshape(-1)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    mixture = 0.5 * (p + q)
    p_mask = p > 0
    q_mask = q > 0
    kl_p = float(np.sum(p[p_mask] * np.log((p[p_mask] + 1e-12) / mixture[p_mask])))
    kl_q = float(np.sum(q[q_mask] * np.log((q[q_mask] + 1e-12) / mixture[q_mask])))
    return 0.5 * (kl_p + kl_q)


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def safe_median(values: list[float]) -> float:
    return median(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--typed-trace",
        type=Path,
        default=Path("datasets/derived/evivit_v16/typed_data_v1/trace1144_train.jsonl"),
    )
    parser.add_argument(
        "--raw-trace",
        type=Path,
        default=Path("datasets/human_annotations/visualprobe/train_search_traces.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/derived/evivit_v16/trace_labels_multiscale_v1"),
    )
    parser.add_argument("--map-size", type=int, default=128)
    parser.add_argument("--max-dwell-ms", type=float, default=250.0)
    args = parser.parse_args()

    typed = read_jsonl(args.typed_trace)
    raw_by_sample = {
        str(row.get("sample_id", row["id"])).split("_idx", 1)[0]: row
        for row in read_jsonl(args.raw_trace)
    }
    arrays: dict[str, list[np.ndarray]] = {
        "context": [],
        "read": [],
        "core": [],
        "failed_branch": [],
        "recovery_branch": [],
        "terminal": [],
    }
    manifest: list[dict[str, Any]] = []
    missing: list[str] = []
    reset_rows = 0
    recovery_pairs = 0
    valid_hierarchy = 0
    read_core_coverages: list[float] = []
    read_core_ious: list[float] = []
    failed_recovery_js: list[float] = []
    context_entropies: list[float] = []

    for typed_row in typed:
        sample_id = str(typed_row["id"])
        raw = raw_by_sample.get(sample_id)
        if raw is None:
            missing.append(sample_id)
            continue
        width, height = [int(value) for value in typed_row["image_size"]]
        final_boxes = typed_row.get("final_boxes_policy") or []
        maps = build_trace_maps(
            raw.get("events") or [],
            width=width,
            height=height,
            final_boxes_policy=final_boxes,
            size=args.map_size,
            max_dwell_ms=args.max_dwell_ms,
        )

        # Keep nested targets separate.  The terminal map is only a convenient
        # backward-compatible summary; v16 training consumes the three channels.
        context = np.asarray(maps.success_trace, dtype=np.float32)
        read = np.asarray(maps.last_zoom, dtype=np.float32)
        core = np.asarray(maps.final_box, dtype=np.float32)
        failed = np.asarray(maps.failed_trace, dtype=np.float32)
        recovery = np.asarray(maps.success_trace, dtype=np.float32)
        terminal = 0.15 * context + 0.45 * read + 0.40 * core
        peak = float(terminal.max(initial=0.0))
        if peak > 0:
            terminal = terminal / peak

        label_index = len(manifest)
        channel_values = {
            "context": context,
            "read": read,
            "core": core,
            "failed_branch": failed,
            "recovery_branch": recovery,
            "terminal": terminal,
        }
        for name, array in channel_values.items():
            arrays[name].append(array.astype(np.float16))

        resets = int(maps.stats.get("resets", 0))
        has_failed = float(failed.sum()) > 0
        has_recovery = float(recovery.sum()) > 0
        pair_available = resets > 0 and has_failed and has_recovery
        reset_rows += int(resets > 0)
        recovery_pairs += int(pair_available)

        coverage_values: list[float] = []
        iou_values: list[float] = []
        if maps.last_zoom_box:
            for box in final_boxes:
                coverage_values.append(target_coverage(maps.last_zoom_box, box))
                iou_values.append(policy_iou(maps.last_zoom_box, box))
        read_core_coverage = max(coverage_values, default=0.0)
        read_core_iou = max(iou_values, default=0.0)
        hierarchy_valid = bool(final_boxes and maps.last_zoom_box and read_core_coverage >= 0.80)
        valid_hierarchy += int(hierarchy_valid)
        if maps.last_zoom_box and final_boxes:
            read_core_coverages.append(read_core_coverage)
            read_core_ious.append(read_core_iou)

        js = jensen_shannon(failed, recovery) if pair_available else 0.0
        if pair_available:
            failed_recovery_js.append(js)
        context_entropy = normalized_entropy(context)
        context_entropies.append(context_entropy)
        manifest.append(
            {
                "id": sample_id,
                "label_index": label_index,
                "image": typed_row["image"],
                "image_size": [width, height],
                "question": typed_row["question"],
                "answer": typed_row["answer"],
                "difficulty": typed_row.get("difficulty"),
                "channels": {
                    "context": "post-last-reset successful hover/dwell/zoom search map",
                    "read": "last human zoom box",
                    "core": "final manually drawn answer box",
                    "failed_branch": "pre-last-reset question-conditional branch",
                    "recovery_branch": "post-last-reset successful branch",
                },
                "loss_weights": {"context": 0.15, "read": 0.45, "core": 0.40},
                "last_zoom_box": maps.last_zoom_box,
                "final_boxes_policy": final_boxes,
                "reset_count": resets,
                "recovery_pair_available": pair_available,
                "failed_branch_role": (
                    "relative_downweight_for_same_question"
                    if pair_available
                    else "not_used"
                ),
                "read_core_coverage": read_core_coverage,
                "read_core_iou": read_core_iou,
                "hierarchy_valid_at_coverage_0_80": hierarchy_valid,
                "context_entropy_normalized": context_entropy,
                "failed_recovery_js": js,
                "map_mass": {
                    "context_in_read": (
                        map_mass(context, maps.last_zoom_box)
                        if maps.last_zoom_box
                        else 0.0
                    ),
                    "context_in_core": max(
                        (map_mass(context, box) for box in final_boxes), default=0.0
                    ),
                },
                "trace_stats": maps.stats,
            }
        )

    if missing:
        raise RuntimeError(f"missing {len(missing)} raw traces; first={missing[:5]}")
    if len(manifest) != 1144:
        raise RuntimeError(f"expected 1,144 labels, got {len(manifest)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "multiscale_trace_maps.npz"
    np.savez_compressed(
        npz_path,
        **{name: np.stack(values, axis=0) for name, values in arrays.items()},
    )
    manifest_path = args.output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, manifest)
    audit = {
        "format_version": "evivit_v16_multiscale_trace_labels_v1",
        "rows": len(manifest),
        "map_size": args.map_size,
        "channels": list(arrays),
        "loss_weights": {"context": 0.15, "read": 0.45, "core": 0.40},
        "reset_rows": reset_rows,
        "valid_recovery_pairs": recovery_pairs,
        "recovery_pair_rate_all": recovery_pairs / len(manifest),
        "recovery_pair_rate_reset": recovery_pairs / max(1, reset_rows),
        "hierarchy_valid_rows": valid_hierarchy,
        "hierarchy_valid_rate": valid_hierarchy / len(manifest),
        "read_core_coverage_mean": safe_mean(read_core_coverages),
        "read_core_coverage_median": safe_median(read_core_coverages),
        "read_core_iou_mean": safe_mean(read_core_ious),
        "failed_recovery_js_mean": safe_mean(failed_recovery_js),
        "context_entropy_mean": safe_mean(context_entropies),
        "semantics": {
            "failed_branch_is_universal_negative": False,
            "failed_branch_use": "same-question branch-ranking/recovery auxiliary only",
            "no_reset_rows_use_recovery_loss": False,
        },
        "source_sha256": {
            "typed_trace": sha256(args.typed_trace),
            "raw_trace": sha256(args.raw_trace),
        },
        "outputs": {
            "maps": str(npz_path),
            "manifest": str(manifest_path),
        },
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
