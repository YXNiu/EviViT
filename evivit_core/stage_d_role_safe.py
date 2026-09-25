"""Role-safe Stage-D region decoding on one frozen PTEA probability map."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np

from scripts.build_evimap_portfolio_boxes import decode_portfolio
from evivit_core.dense_evidence import policy_iou
from evivit_core.seeded_residual_basin import (
    BasinDecoderConfig,
    decode_seeded_residual_basins,
)


RoleSafeMode = Literal[
    "confidence_fallback",
    "replace_r3",
    "partial_fill_three",
    "relaxed_basin_three",
]


def _anchor_focus(probability: np.ndarray, max_regions: int) -> list[dict[str, Any]]:
    portfolio = decode_portfolio(
        probability,
        top_k=max(2 * max_regions + 2, 8),
        modes=max_regions,
        context_factor=1.8,
        residual_suppression=0.05,
        include_topology_union=False,
    )
    focus = [
        dict(candidate)
        for candidate in portfolio
        if str(candidate.get("portfolio_role", "")).endswith("_focus")
    ][:max_regions]
    if not focus:
        raise RuntimeError("Stage-D anchor decoder returned no focus role")
    return focus


def decode_role_safe_basins(
    probability: np.ndarray,
    *,
    max_regions: int = 3,
    config: BasinDecoderConfig | None = None,
    mode: RoleSafeMode = "confidence_fallback",
    maximum_anchor_iou: float = 0.35,
) -> list[dict[str, Any]]:
    """Decode a safe hybrid of frozen anchors and seeded evidence basins.

    A basin portfolio is considered confident only if the already-frozen D1
    stopping rule independently discovers all ``max_regions`` basins.  The
    confidence-fallback version switches the complete portfolio.  The R3-safe
    version retains the first two v4 anchor roles and only replaces R3 with a
    sufficiently complementary basin.  No answer label or human box is read.
    """

    if max_regions != 3:
        raise ValueError("Stage-D role-safe decoding currently requires 3 regions")
    if mode not in {
        "confidence_fallback",
        "replace_r3",
        "partial_fill_three",
        "relaxed_basin_three",
    }:
        raise ValueError(f"unknown Stage-D role-safe mode: {mode}")
    if not 0 <= maximum_anchor_iou <= 1:
        raise ValueError("maximum_anchor_iou must lie in [0, 1]")

    anchors = _anchor_focus(probability, max_regions)
    if mode == "relaxed_basin_three":
        # This is a strict region-count ablation, not the proposed safe route.
        # It disables only the two early-stop thresholds and the pairwise-IoU
        # rejection so that weak residual modes can be measured explicitly.
        relaxed = replace(
            config or BasinDecoderConfig(),
            minimum_residual_mass=0.0,
            minimum_seed_ratio=0.0,
            maximum_pairwise_iou=1.0,
        )
        basins = decode_seeded_residual_basins(
            probability,
            max_regions=max_regions,
            config=relaxed,
        )
        # A fully suppressed map can still contain fewer than three basins.
        # Fill only those exceptional empty slots with the least-overlapping
        # frozen anchors, and expose the route in diagnostics.
        selected = [dict(item) for item in basins]
        for candidate in selected:
            candidate["stage_d_route"] = "relaxed_basin_three"
        remaining = [dict(item) for item in anchors]
        while len(selected) < max_regions and remaining:
            anchor = min(
                remaining,
                key=lambda item: max(
                    (policy_iou(item["bbox"], row["bbox"]) for row in selected),
                    default=0.0,
                ),
            )
            remaining.remove(anchor)
            anchor["stage_d_route"] = "relaxed_basin_anchor_fill"
            selected.append(anchor)
        return selected[:max_regions]

    basins = decode_seeded_residual_basins(
        probability,
        max_regions=max_regions,
        config=config,
    )

    if mode == "partial_fill_three":
        # Keep every independently valid D1 basin, then fill only missing slots
        # with complementary Stage-A anchors.  Scores inherit the frozen anchor
        # role so mixed score scales cannot silently change the Fine budget.
        selected: list[dict[str, Any]] = []
        for index, basin in enumerate(basins):
            candidate = dict(basin)
            candidate["basin_selection_score"] = float(candidate.get("score", 0.0))
            candidate["score"] = float(anchors[min(index, len(anchors) - 1)]["score"])
            candidate["stage_d_route"] = "partial_valid_basin"
            selected.append(candidate)
        remaining = [dict(item) for item in anchors]
        while len(selected) < max_regions and remaining:
            anchor = min(
                remaining,
                key=lambda item: max(
                    (policy_iou(item["bbox"], row["bbox"]) for row in selected),
                    default=0.0,
                ),
            )
            remaining.remove(anchor)
            anchor["stage_d_route"] = "partial_anchor_fill"
            selected.append(anchor)
        return selected[:max_regions]

    confident = len(basins) == max_regions
    if not confident:
        for candidate in anchors:
            candidate["stage_d_route"] = "anchor_fallback"
        return anchors

    if mode == "confidence_fallback":
        for candidate in basins:
            candidate["stage_d_route"] = "basin_confident"
        return basins

    if len(anchors) < max_regions:
        for candidate in anchors:
            candidate["stage_d_route"] = "anchor_missing_r3"
        return anchors

    protected = anchors[:2]
    eligible: list[tuple[float, dict[str, Any], float]] = []
    for basin in basins:
        overlap = max(policy_iou(basin["bbox"], item["bbox"]) for item in protected)
        if overlap <= maximum_anchor_iou:
            utility = float(basin.get("mass", 0.0)) * (1.0 - overlap)
            eligible.append((utility, basin, overlap))
    if not eligible:
        for candidate in anchors:
            candidate["stage_d_route"] = "anchor_no_complement"
        return anchors

    _, replacement, overlap = max(eligible, key=lambda item: item[0])
    replacement = dict(replacement)
    replacement["basin_selection_score"] = float(replacement.get("score", 0.0))
    # Preserve the frozen R3 budget role. Under Fine-Native this also keeps the
    # requested budget metadata comparable even though realized tokens follow
    # the original crop pixels.
    replacement["score"] = float(anchors[2]["score"])
    replacement["portfolio_role"] = "role_safe_dynamic_r3"
    replacement["stage_d_route"] = "replace_r3"
    replacement["maximum_anchor_iou"] = float(overlap)
    for candidate in protected:
        candidate["stage_d_route"] = "protected_anchor"
    return [*protected, replacement]


__all__ = ["decode_role_safe_basins"]
