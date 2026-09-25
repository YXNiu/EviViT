#!/usr/bin/env python3
"""Build a fixed-budget multi-role evidence portfolio from one EviMap.

The decoder is training-free.  Instead of spending every slot on the globally
highest overlapping windows, it allocates slots to three residual evidence
modes, pairs each tight evidence crop with a larger context crop, and fills the
remaining budget with broad high-mass context windows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evivit_core.dense_evidence import (
    decode_top_boxes,
    expand_policy_box,
    map_mass,
    normalize_distribution,
    policy_iou,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def suppress_box(distribution: np.ndarray, bbox: list[int], factor: float) -> np.ndarray:
    result = distribution.copy()
    height, width = result.shape
    x1, y1, x2, y2 = bbox
    gx1 = min(width - 1, max(0, int(math.floor(x1 / 1000 * width))))
    gy1 = min(height - 1, max(0, int(math.floor(y1 / 1000 * height))))
    gx2 = min(width, max(gx1 + 1, int(math.ceil(x2 / 1000 * width))))
    gy2 = min(height, max(gy1 + 1, int(math.ceil(y2 / 1000 * height))))
    result[gy1:gy2, gx1:gx2] *= factor
    return normalize_distribution(result)


def scored_candidate(
    distribution: np.ndarray,
    bbox: list[int],
    *,
    role: str,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox
    area_ratio = max(1, x2 - x1) * max(1, y2 - y1) / 1_000_000.0
    mass = map_mass(distribution, bbox)
    return {
        "bbox": bbox,
        "score": mass / max(1e-6, area_ratio**0.75) - 0.05 * area_ratio,
        "mass": mass,
        "area_ratio": area_ratio,
        "scale": math.sqrt(area_ratio),
        "portfolio_role": role,
        "source_bbox": source.get("bbox") if source else None,
    }


def unique_append(
    selected: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    max_iou: float = 0.92,
) -> bool:
    bbox = candidate["bbox"]
    if any(policy_iou(bbox, item["bbox"]) >= max_iou for item in selected):
        return False
    selected.append(candidate)
    return True


def append_deterministic_capacity_fallbacks(
    selected: list[dict[str, Any]],
    distribution: np.ndarray,
    *,
    top_k: int,
) -> None:
    """Fill unused *portfolio* slots without changing decoded focus modes.

    Very small PTEA maps can expose fewer than ``top_k`` distinct sliding
    windows after NMS.  The online EviViT path ultimately consumes the focus
    modes at the front of ``selected``; the remaining portfolio entries are
    only reserve/context candidates.  Failing the whole evaluation because a
    reserve slot is missing is therefore unnecessary.  Deterministic boxes are
    appended only after every evidence-derived fallback has been exhausted, so
    the method's selected focus regions and their ordering stay unchanged.
    """

    existing = {tuple(item["bbox"]) for item in selected}
    # Peak-centred boxes come first, followed by a fixed spatial lattice.  This
    # remains deterministic even for a 1x1 probability map.
    peak_y, peak_x = np.unravel_index(int(np.argmax(distribution)), distribution.shape)
    height, width = distribution.shape
    peak_center = ((peak_x + 0.5) / width, (peak_y + 0.5) / height)
    centers = [
        peak_center,
        (0.25, 0.25),
        (0.75, 0.25),
        (0.25, 0.75),
        (0.75, 0.75),
        (0.50, 0.50),
        (0.50, 0.25),
        (0.50, 0.75),
        (0.25, 0.50),
        (0.75, 0.50),
    ]
    for scale in (0.125, 0.20, 0.33, 0.50, 0.67, 1.0):
        for center_x, center_y in centers:
            half = scale / 2
            x1 = max(0.0, min(1.0 - scale, center_x - half))
            y1 = max(0.0, min(1.0 - scale, center_y - half))
            bbox = [
                int(round(x1 * 1000)),
                int(round(y1 * 1000)),
                int(round((x1 + scale) * 1000)),
                int(round((y1 + scale) * 1000)),
            ]
            key = tuple(bbox)
            if key in existing:
                continue
            existing.add(key)
            selected.append(
                scored_candidate(
                    distribution,
                    bbox,
                    role=f"capacity_fallback_{len(selected) + 1}",
                )
            )
            if len(selected) >= top_k:
                return


def decode_portfolio(
    probability: np.ndarray,
    *,
    top_k: int,
    modes: int,
    context_factor: float,
    residual_suppression: float,
    include_topology_union: bool = False,
    topology_union_factor: float = 1.0,
) -> list[dict[str, Any]]:
    if topology_union_factor < 1.0:
        raise ValueError("topology_union_factor must be at least 1.0")
    distribution = normalize_distribution(probability)
    residual = distribution.copy()
    selected: list[dict[str, Any]] = []
    focus_boxes: list[list[int]] = []

    for mode in range(modes):
        focus_candidates = decode_top_boxes(
            residual,
            scales=(0.2, 0.25),
            topk=1,
            mass_power=0.75,
            area_penalty=0.05,
            nms_iou=0.3,
        )
        if not focus_candidates:
            break
        focus_box = list(focus_candidates[0]["bbox"])
        focus_boxes.append(focus_box)
        focus = scored_candidate(
            distribution, focus_box, role=f"mode_{mode + 1}_focus"
        )
        unique_append(selected, focus)
        if len(selected) >= top_k:
            break
        context_box = expand_policy_box(focus_box, context_factor)
        context = scored_candidate(
            distribution,
            context_box,
            role=f"mode_{mode + 1}_context",
            source=focus,
        )
        unique_append(selected, context)
        residual = suppress_box(residual, context_box, residual_suppression)
        if len(selected) >= top_k:
            break

    if include_topology_union and len(focus_boxes) >= 2 and len(selected) < top_k:
        # Topology is defined by the two strongest complementary foci. Later
        # residual modes are useful for R5 coverage but must not silently make
        # the shared R1--R2 frame expand toward the whole image.
        topology_sources = focus_boxes[:2]
        topology_box = [
            min(box[0] for box in topology_sources),
            min(box[1] for box in topology_sources),
            max(box[2] for box in topology_sources),
            max(box[3] for box in topology_sources),
        ]
        topology_box = expand_policy_box(topology_box, topology_union_factor)
        topology = scored_candidate(
            distribution,
            topology_box,
            role="topology_union",
        )
        # The union has a different semantic role from a focus/context crop:
        # even when it overlaps a broad context window, it must remain present
        # so the topology ablation is guaranteed to compare two tight views
        # against their shared coordinate frame.
        selected.append(topology)

    broad = decode_top_boxes(
        distribution,
        scales=(0.35, 0.5, 0.67),
        topk=max(20, top_k * 4),
        mass_power=0.75,
        area_penalty=0.05,
        nms_iou=0.2,
    )
    for index_, source in enumerate(broad, 1):
        candidate = scored_candidate(
            distribution,
            list(source["bbox"]),
            role=f"broad_context_{index_}",
            source=source,
        )
        if unique_append(selected, candidate, max_iou=0.72) and len(selected) >= top_k:
            break

    if len(selected) < top_k:
        fallback = decode_top_boxes(
            distribution,
            scales=(0.2, 0.25, 0.35, 0.5, 0.67),
            topk=64,
            mass_power=0.75,
            area_penalty=0.05,
            nms_iou=0.15,
        )
        for index_, source in enumerate(fallback, 1):
            candidate = scored_candidate(
                distribution,
                list(source["bbox"]),
                role=f"fallback_{index_}",
                source=source,
            )
            if unique_append(selected, candidate) and len(selected) >= top_k:
                break
    if len(selected) < top_k:
        append_deterministic_capacity_fallbacks(
            selected,
            distribution,
            top_k=top_k,
        )
    if len(selected) != top_k:
        raise RuntimeError(f"portfolio decoder produced {len(selected)} of {top_k} boxes")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-boxes", type=Path, required=True)
    parser.add_argument("--probability-maps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--modes", type=int, default=3)
    parser.add_argument("--context-factor", type=float, default=1.8)
    parser.add_argument("--residual-suppression", type=float, default=0.05)
    parser.add_argument("--include-topology-union", action="store_true")
    parser.add_argument("--topology-union-factor", type=float, default=1.0)
    args = parser.parse_args()
    if args.top_k < 1 or args.modes < 1:
        raise ValueError("top-k and modes must be positive")

    source_rows = read_jsonl(args.source_boxes)
    bundle = torch.load(args.probability_maps, map_location="cpu", weights_only=False)
    probability_maps: dict[str, torch.Tensor] = bundle["maps"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    role_counts: dict[str, int] = {}
    with args.output.open("w", encoding="utf-8") as handle:
        for row in source_rows:
            row_id = str(row["id"])
            candidates = decode_portfolio(
                probability_maps[row_id].float().numpy(),
                top_k=args.top_k,
                modes=args.modes,
                context_factor=args.context_factor,
                residual_suppression=args.residual_suppression,
                include_topology_union=args.include_topology_union,
                topology_union_factor=args.topology_union_factor,
            )
            for candidate in candidates:
                role = str(candidate["portfolio_role"])
                role_counts[role] = role_counts.get(role, 0) + 1
            output = dict(row)
            output.update(
                {
                    "selector_version": (
                        "ptea_a_topology_preserving_portfolio_v1"
                        if args.include_topology_union
                        else "ptea_a_evidence_portfolio_v1"
                    ),
                    "decoder_version": (
                        "topology_preserving_portfolio_v1"
                        if args.include_topology_union
                        else "evidence_portfolio_v1"
                    ),
                    "decode_top_k": args.top_k,
                    "portfolio_modes": args.modes,
                    "portfolio_context_factor": args.context_factor,
                    "portfolio_residual_suppression": args.residual_suppression,
                    "portfolio_topology_union": args.include_topology_union,
                    "topology_union_factor": args.topology_union_factor,
                    "candidates": candidates,
                }
            )
            handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "version": (
            "topology_preserving_portfolio_v1"
            if args.include_topology_union
            else "evidence_portfolio_v1"
        ),
        "rows": len(source_rows),
        "top_k": args.top_k,
        "modes": args.modes,
        "context_factor": args.context_factor,
        "residual_suppression": args.residual_suppression,
        "include_topology_union": args.include_topology_union,
        "role_counts": role_counts,
        "source_boxes": str(args.source_boxes),
        "probability_maps": str(args.probability_maps),
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
