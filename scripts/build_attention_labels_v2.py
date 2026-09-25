#!/usr/bin/env python3
"""Build v2-lite human-search attention labels for all VisualProbe train traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evivit_core.attention_maps_v2 import (
    MAIN_WEIGHT_PROFILES,
    TERMINAL_WEIGHTS,
    build_attention_maps_v2,
)
from evivit_core.dense_evidence import pixels_to_policy_box
from scripts.build_dense_evidence_labels import load_image_sizes_from_sources


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image_sizes(groups_path: Path) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int]] = {}
    with groups_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("depth", -1)) != 0:
                continue
            size = row.get("image_size")
            if not isinstance(size, list) or len(size) != 2:
                raise RuntimeError(f"{groups_path}:{line_number}: missing image_size")
            sizes[str(row["id"])] = (int(size[0]), int(size[1]))
    return sizes


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traces", type=Path,
        default=Path("datasets/human_annotations/visualprobe/train_search_traces.jsonl"),
    )
    parser.add_argument(
        "--ranker-groups", type=Path,
        default=Path("datasets/derived/ad_paz/attention_labels/ranker_groups.jsonl"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--main-profile",
        choices=sorted(MAIN_WEIGHT_PROFILES),
        default="terminal_dominant",
    )
    parser.add_argument("--map-size", type=int, default=128)
    parser.add_argument("--velocity-tau", type=float, default=0.08)
    parser.add_argument("--fast-path-floor", type=float, default=0.10)
    parser.add_argument("--max-interval-ms", type=float, default=500.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path(
            f"datasets/derived/attention_map_v2/labels_v2_{args.main_profile}_128"
        )

    traces = read_jsonl(args.traces, args.limit)
    image_sizes = (
        load_image_sizes(args.ranker_groups)
        if args.ranker_groups.is_file()
        else load_image_sizes_from_sources(traces, args.project_root)
    )
    channel_names = [
        "slow_dwell", "all_zoom", "committed_slow", "committed_zoom",
        "last_zoom", "final_box", "exploration", "committed",
        "terminal", "ambiguity", "main",
    ]
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in channel_names}
    manifest: list[dict[str, Any]] = []
    missing_sizes: list[str] = []
    for row in traces:
        row_id = str(row["id"])
        size = image_sizes.get(row_id)
        if size is None:
            missing_sizes.append(row_id)
            continue
        width, height = size
        final_boxes_policy = [
            pixels_to_policy_box(box, width, height)
            for box in row.get("final_bboxes_pixels", [])
            if isinstance(box, list) and len(box) == 4
        ]
        maps = build_attention_maps_v2(
            row.get("events", []),
            width=width,
            height=height,
            final_boxes_policy=final_boxes_policy,
            size=args.map_size,
            velocity_tau_rel_per_second=args.velocity_tau,
            fast_path_floor=args.fast_path_floor,
            max_interval_ms=args.max_interval_ms,
            main_profile=args.main_profile,
        )
        label_index = len(manifest)
        for name in channel_names:
            arrays[name].append(np.asarray(getattr(maps, name), dtype=np.float16))
        manifest.append(
            {
                "id": row_id,
                "label_index": label_index,
                "image": str(row["image"]),
                "question": " ".join(str(row["question"]).replace("<image>", " ").split()),
                "answer": str(row["answer"]),
                "image_size": [width, height],
                "final_boxes_policy": final_boxes_policy,
                "last_zoom_box": maps.last_zoom_box,
                "difficulty": (row.get("task_evaluation") or {}).get("difficulty_level"),
                "trace_stats": maps.stats,
            }
        )
    if missing_sizes:
        raise RuntimeError(f"missing image sizes for {len(missing_sizes)} traces: {missing_sizes[:5]}")
    if not manifest:
        raise RuntimeError("no attention labels built")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    maps_path = args.output_dir / "attention_maps_v2.npz"
    np.savez_compressed(
        maps_path, **{name: np.stack(values, axis=0) for name, values in arrays.items()}
    )
    manifest_path = args.output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, manifest)
    reset_histogram = Counter(int(row["trace_stats"]["resets"]) for row in manifest)
    audit = {
        "version": f"attention_labels_v2_{args.main_profile}_128",
        "rows": len(manifest),
        "map_size": args.map_size,
        "channels": channel_names,
        "parameters": {
            "velocity_tau_rel_per_second": args.velocity_tau,
            "fast_path_floor": args.fast_path_floor,
            "max_interval_ms": args.max_interval_ms,
            "main_profile": args.main_profile,
            "main_weights": MAIN_WEIGHT_PROFILES[args.main_profile],
            "terminal_weights": TERMINAL_WEIGHTS,
            "committed_weights": {"slow_after_last_reset": 0.60, "zoom_after_last_reset": 0.40},
            "exploration_weights": {"slow_dwell": 2.0 / 3.0, "all_zoom": 1.0 / 3.0},
            "ambiguity_weights": {"slow_before_last_reset": 0.55, "zoom_before_last_reset": 0.45},
            "failed_branch_is_negative": False,
        },
        "trace_sha256": sha256(args.traces),
        "ranker_groups_sha256": sha256(args.ranker_groups),
        "reset_count_histogram": dict(sorted(reset_histogram.items())),
        "rows_with_reset": sum(int(row["trace_stats"]["resets"] > 0) for row in manifest),
        "rows_with_ambiguity": sum(
            int(row["trace_stats"]["ambiguity_mass_present"]) for row in manifest
        ),
        "rows_with_last_zoom": sum(bool(row["last_zoom_box"]) for row in manifest),
        "rows_with_final_box": sum(bool(row["final_boxes_policy"]) for row in manifest),
        "movement_points": sum(int(row["trace_stats"]["movement_points"]) for row in manifest),
        "zoom_episodes": sum(int(row["trace_stats"]["zoom_episodes"]) for row in manifest),
        "out_of_bounds_points": sum(
            int(row["trace_stats"]["out_of_bounds_points"]) for row in manifest
        ),
        "out_of_bounds_boxes": sum(
            int(row["trace_stats"]["out_of_bounds_boxes"]) for row in manifest
        ),
        "mean_main_entropy": mean(
            float(row["trace_stats"]["main_entropy"]) for row in manifest
        ),
        "mean_relative_velocity_median": mean(
            float(row["trace_stats"]["relative_velocity_median"]) for row in manifest
        ),
        "outputs": {"maps": str(maps_path), "manifest": str(manifest_path)},
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
