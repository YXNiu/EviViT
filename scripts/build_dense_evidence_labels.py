#!/usr/bin/env python3
"""Build compact dense evidence labels from all 1,144 VisualProbe traces."""

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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evivit_core.dense_evidence import build_trace_maps, pixels_to_policy_box


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


def load_image_sizes_from_sources(
    traces: list[dict[str, Any]], project_root: Path
) -> dict[str, tuple[int, int]]:
    """Recover exact source dimensions when the old ranker cache is absent.

    The ranker groups were only a convenient size index; image dimensions are
    intrinsic raw-data metadata and can be read losslessly from the retained
    source images.  This fallback prevents a derived ranker cache from becoming
    an undeclared hard dependency of the human-trace labels.
    """

    sizes: dict[str, tuple[int, int]] = {}
    for row in traces:
        image_path = Path(str(row["image"]))
        if not image_path.is_absolute():
            image_path = project_root / image_path
        with Image.open(image_path) as image:
            sizes[str(row["id"])] = (int(image.width), int(image.height))
    return sizes


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("datasets/human_annotations/visualprobe/train_search_traces.jsonl"),
    )
    parser.add_argument(
        "--ranker-groups",
        type=Path,
        default=Path("datasets/derived/ad_paz/attention_labels/ranker_groups.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/derived/dense_evigain/trace_labels_v1"),
    )
    parser.add_argument("--map-size", type=int, default=64)
    parser.add_argument("--max-dwell-ms", type=float, default=250.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()

    traces = read_jsonl(args.traces, limit=args.limit)
    if args.ranker_groups.is_file():
        image_sizes = load_image_sizes(args.ranker_groups)
        image_size_source = "ranker_groups"
    else:
        image_sizes = load_image_sizes_from_sources(traces, args.project_root)
        image_size_source = "retained_source_images"
    arrays: dict[str, list[np.ndarray]] = {
        "all_trace": [],
        "success_trace": [],
        "failed_trace": [],
        "last_zoom": [],
        "final_box": [],
        "evidence": [],
    }
    manifest: list[dict[str, Any]] = []
    missing_sizes: list[str] = []
    for index, row in enumerate(traces):
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
        maps = build_trace_maps(
            row.get("events", []),
            width=width,
            height=height,
            final_boxes_policy=final_boxes_policy,
            size=args.map_size,
            max_dwell_ms=args.max_dwell_ms,
        )
        label_index = len(manifest)
        for name in arrays:
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
        raise RuntimeError(f"missing image sizes for {len(missing_sizes)} traces; first={missing_sizes[:5]}")
    if not manifest:
        raise RuntimeError("no labels built")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / "dense_trace_maps.npz"
    np.savez_compressed(
        npz_path,
        **{name: np.stack(values, axis=0) for name, values in arrays.items()},
    )
    manifest_path = args.output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, manifest)

    reset_counts = Counter(int(row["trace_stats"]["resets"]) for row in manifest)
    audit = {
        "version": "dense_trace_labels_v1",
        "rows": len(manifest),
        "map_size": args.map_size,
        "channels": list(arrays),
        "trace_sha256": sha256(args.traces),
        "ranker_groups_sha256": (
            sha256(args.ranker_groups) if args.ranker_groups.is_file() else None
        ),
        "image_size_source": image_size_source,
        "reset_count_histogram": dict(sorted(reset_counts.items())),
        "rows_with_reset": sum(int(row["trace_stats"]["resets"] > 0) for row in manifest),
        "rows_with_last_zoom": sum(bool(row["last_zoom_box"]) for row in manifest),
        "rows_with_final_box": sum(bool(row["final_boxes_policy"]) for row in manifest),
        "mean_evidence_sum": mean(float(row["trace_stats"]["evidence_sum"]) for row in manifest),
        "mean_failed_trace_sum": mean(
            float(row["trace_stats"]["failed_trace_sum"]) for row in manifest
        ),
        "outputs": {"maps": str(npz_path), "manifest": str(manifest_path)},
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
