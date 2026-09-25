#!/usr/bin/env python3
"""Align canonical trace labels to an existing scale-specific feature cache.

Feature caches are keyed by the manifest ID used at extraction time.  Older
Qwen3-VL-8B caches used legacy IDs containing source indices, while the
canonical Human1144 trace manifest uses stable image IDs.  This utility joins
the two manifests only by an exact normalized (image basename, question)
pair, verifies a bijection and answer agreement, and then emits canonical
trace rows carrying the cache IDs.  It deliberately refuses row-order joins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def key(row: dict[str, Any]) -> tuple[str, str]:
    return Path(str(row["image"])).name, norm_text(row["question"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=1144)
    args = parser.parse_args()

    cache_rows = read_jsonl(args.cache_manifest)
    trace_rows = read_jsonl(args.trace_manifest)
    if len(cache_rows) != args.expected_rows or len(trace_rows) != args.expected_rows:
        raise RuntimeError(
            f"row count mismatch: cache={len(cache_rows)} trace={len(trace_rows)} "
            f"expected={args.expected_rows}"
        )

    trace_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in trace_rows:
        item_key = key(row)
        if item_key in trace_by_key:
            raise RuntimeError(f"duplicate canonical trace key: {item_key}")
        trace_by_key[item_key] = row

    output_rows: list[dict[str, Any]] = []
    used_trace_ids: set[str] = set()
    for cache_row in cache_rows:
        item_key = key(cache_row)
        trace_row = trace_by_key.get(item_key)
        if trace_row is None:
            raise RuntimeError(f"no exact trace match for cache row {cache_row['id']}: {item_key}")
        if norm_text(cache_row.get("answer")) != norm_text(trace_row.get("answer")):
            raise RuntimeError(f"answer mismatch for {cache_row['id']}")
        trace_id = str(trace_row["id"])
        if trace_id in used_trace_ids:
            raise RuntimeError(f"non-bijective trace match for {trace_id}")
        used_trace_ids.add(trace_id)
        aligned = dict(trace_row)
        aligned["canonical_trace_id"] = trace_id
        aligned["id"] = str(cache_row["id"])
        aligned["image"] = str(cache_row["image"])
        output_rows.append(aligned)

    if len(used_trace_ids) != len(trace_rows):
        missing = sorted({str(row["id"]) for row in trace_rows} - used_trace_ids)
        raise RuntimeError(f"trace join is not complete; first missing={missing[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "version": "evivit_scale_aligned_trace_manifest_v1",
        "join": "exact(image_basename, normalized_question) with answer agreement",
        "rows": len(output_rows),
        "unique_cache_ids": len({str(row["id"]) for row in cache_rows}),
        "unique_trace_ids": len(used_trace_ids),
        "cache_manifest": str(args.cache_manifest),
        "cache_manifest_sha256": sha256(args.cache_manifest),
        "trace_manifest": str(args.trace_manifest),
        "trace_manifest_sha256": sha256(args.trace_manifest),
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
