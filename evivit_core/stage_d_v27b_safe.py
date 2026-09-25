"""v27b-inspired safe native caps for Stage-D anchor-fallback regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from evivit_core.evivit import RegionBudget


@dataclass(frozen=True)
class StageDV27bSafePlan:
    route: str
    context_fraction: float
    region_token_caps: tuple[int, int, int] | None


def _split_integer_exact(
    total: int,
    weights: Sequence[float],
    *,
    minimum: int,
) -> tuple[int, ...]:
    if total < len(weights) * minimum:
        raise ValueError("budget is smaller than the required region floors")
    positive = [max(float(value), 1e-8) for value in weights]
    normalizer = sum(positive)
    remaining = total - len(weights) * minimum
    raw = [remaining * value / normalizer for value in positive]
    extra = [int(value) for value in raw]
    residual = remaining - sum(extra)
    order = sorted(
        range(len(raw)),
        key=lambda index: -(raw[index] - extra[index]),
    )
    for index in order[:residual]:
        extra[index] += 1
    return tuple(minimum + value for value in extra)


def plan_stage_d_v27b_safe_caps(
    regions: Sequence[RegionBudget],
    *,
    total_fine_tokens: int,
    context_fraction: float = 0.2176,
    minimum_region_tokens: int = 64,
) -> StageDV27bSafePlan:
    """Cap only Stage-A fallback regions; preserve confident D1 basins.

    The plan borrows v27b's two-decisive-plus-one-context safety contract but
    applies it as a *native-token ceiling*.  A crop below its ceiling remains
    at native resolution; no region is enlarged to fill unused capacity.
    """

    if not 0.10 <= context_fraction <= 0.25:
        raise ValueError("context fraction must lie in the v27b-safe [0.10, 0.25]")
    # Confidence fallback normally returns three regions, but boundary
    # de-duplication can leave an exceptional sample with fewer valid boxes.
    # Such samples must retain the frozen D2-A behavior instead of being
    # padded with a weak/duplicated region solely to satisfy this ablation.
    if len(regions) != 3:
        return StageDV27bSafePlan(
            route="non_three_region_identity",
            context_fraction=context_fraction,
            region_token_caps=None,
        )
    if total_fine_tokens < 3 * minimum_region_tokens:
        raise ValueError("fine budget cannot satisfy three region floors")

    roles = [str(region.role) for region in regions]
    if any("seeded_basin" in role for role in roles):
        return StageDV27bSafePlan(
            route="basin_confident_identity",
            context_fraction=context_fraction,
            region_token_caps=None,
        )
    if not all("mode_" in role and role.endswith("_focus") for role in roles):
        raise ValueError("fallback regions do not match the frozen Stage-A roles")

    context_tokens = max(
        minimum_region_tokens,
        int(round(total_fine_tokens * context_fraction)),
    )
    decisive_tokens = total_fine_tokens - context_tokens
    decisive_caps = _split_integer_exact(
        decisive_tokens,
        [regions[0].token_budget, regions[1].token_budget],
        minimum=minimum_region_tokens,
    )
    caps = (decisive_caps[0], decisive_caps[1], context_tokens)
    if sum(caps) != total_fine_tokens:
        raise AssertionError("Stage-D v27b-safe caps lost the Fine budget")
    return StageDV27bSafePlan(
        route="anchor_fallback_v27b_safe",
        context_fraction=context_fraction,
        region_token_caps=caps,
    )


__all__ = ["StageDV27bSafePlan", "plan_stage_d_v27b_safe_caps"]
