"""Zero-parameter Native-Cap planning for Simple EviViT.

The frozen EviViT-v4 path treats Global/Fine token budgets as targets and
resizes every view to spend them.  Simple EviViT S1 changes only that budget
semantics: each target becomes a ceiling.  Images that naturally require fewer
Qwen visual tokens keep their native grid, while larger views are capped with
the same aspect-preserving grid utility used by v4.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from evivit_core.evitree import qwen_native_visual_token_geometry
from evivit_core.evivit import grid_for_token_budget


@dataclass(frozen=True)
class NativeTokenCapPlan:
    """Auditable native-grid plan under one hard merged-token ceiling."""

    native_tokens: int
    native_resized_height: int
    native_resized_width: int
    token_cap: int
    minimum_tokens: int
    requested_tokens: int
    grid_height: int
    grid_width: int
    realized_tokens: int
    cap_saturated: bool
    minimum_applied: bool
    minimum_shortfall_tokens: int
    native_grid_preserved: bool
    unused_cap_tokens: int

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class ContinuousFineFloorPlan:
    """Native-relative total Fine floor and its per-region allocation."""

    native_source_tokens: int
    native_ratio: float
    minimum_total_tokens: int
    maximum_total_tokens: int
    requested_total_floor: int
    region_token_caps: tuple[int, ...]
    region_floor_tokens: tuple[int, ...]

    def to_dict(self) -> dict[str, int | float | list[int]]:
        payload = asdict(self)
        payload["region_token_caps"] = list(self.region_token_caps)
        payload["region_floor_tokens"] = list(self.region_floor_tokens)
        return payload


def plan_anchor_fine_floors(
    region_token_caps: list[int] | tuple[int, ...],
    *,
    base_minimum_tokens: int,
    anchor_minimum_tokens: int = 0,
) -> tuple[int, ...]:
    """Build S3 per-region floors without forcing every crop to be large.

    Residual-Focus orders the strongest evidence region first.  S3 therefore
    gives only that first region a stronger readability floor while all
    remaining context regions retain the ordinary Native-Cap floor.  Every
    floor is clipped by its region allocation, so the shared Fine ceiling can
    never be exceeded.
    """

    caps = tuple(int(value) for value in region_token_caps)
    if any(value <= 0 for value in caps):
        raise ValueError("region_token_caps must be positive")
    if base_minimum_tokens < 0 or anchor_minimum_tokens < 0:
        raise ValueError("Native-Cap floors must be non-negative")
    floors = [
        min(cap, int(base_minimum_tokens))
        for cap in caps
    ]
    if floors and anchor_minimum_tokens:
        floors[0] = min(
            caps[0],
            max(floors[0], int(anchor_minimum_tokens)),
        )
    return tuple(floors)


def plan_continuous_fine_floor(
    native_source_tokens: int,
    region_token_caps: list[int] | tuple[int, ...],
    *,
    native_ratio: float,
    minimum_total_tokens: int,
    maximum_total_tokens: int,
) -> ContinuousFineFloorPlan:
    """Split a continuous source-relative Fine floor over evidence regions."""

    caps = tuple(int(value) for value in region_token_caps)
    if native_source_tokens <= 0:
        raise ValueError("native_source_tokens must be positive")
    if not caps or any(value <= 0 for value in caps):
        raise ValueError("region_token_caps must be non-empty and positive")
    if not math.isfinite(native_ratio) or native_ratio < 0:
        raise ValueError("native_ratio must be finite and non-negative")
    if minimum_total_tokens < 0:
        raise ValueError("minimum_total_tokens must be non-negative")
    available = min(int(maximum_total_tokens), sum(caps))
    if available <= 0 or minimum_total_tokens > available:
        raise ValueError("invalid continuous Fine floor interval")

    requested = min(
        available,
        max(
            int(minimum_total_tokens),
            int(round(float(native_ratio) * int(native_source_tokens))),
        ),
    )
    total_cap = sum(caps)
    raw = [requested * value / total_cap for value in caps]
    floors = [min(cap, int(math.floor(value))) for cap, value in zip(caps, raw)]
    remaining = requested - sum(floors)
    order = sorted(
        range(len(caps)),
        key=lambda index: (raw[index] - floors[index], caps[index], -index),
        reverse=True,
    )
    while remaining:
        progressed = False
        for index in order:
            if floors[index] >= caps[index]:
                continue
            floors[index] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise AssertionError("continuous Fine floor allocation stalled")

    if sum(floors) != requested or any(
        floor > cap for floor, cap in zip(floors, caps)
    ):
        raise AssertionError("invalid continuous Fine floor allocation")
    return ContinuousFineFloorPlan(
        native_source_tokens=int(native_source_tokens),
        native_ratio=float(native_ratio),
        minimum_total_tokens=int(minimum_total_tokens),
        maximum_total_tokens=int(maximum_total_tokens),
        requested_total_floor=requested,
        region_token_caps=caps,
        region_floor_tokens=tuple(floors),
    )


def plan_native_token_cap(
    width: int,
    height: int,
    *,
    token_cap: int,
    minimum_tokens: int = 0,
    minimum_pixels: int = 4096,
    maximum_pixels: int = 16 * 1024 * 1024,
    patch_size: int = 16,
    merge_size: int = 2,
) -> NativeTokenCapPlan:
    """Keep the native Qwen grid unless it violates a floor or ceiling.

    ``minimum_tokens=0`` is the protected-Global protocol: a small image is
    never artificially enlarged beyond Qwen's own processor minimum.  Fine
    evidence views may use a small structural floor (64 in S1) so an extremely
    small crop still receives a useful local read.
    """

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if token_cap <= 0:
        raise ValueError("token cap must be positive")
    if minimum_tokens < 0 or minimum_tokens > token_cap:
        raise ValueError("minimum tokens must lie in [0, token cap]")

    native_tokens, resized_height, resized_width = (
        qwen_native_visual_token_geometry(
            width,
            height,
            minimum_pixels=minimum_pixels,
            maximum_pixels=maximum_pixels,
            patch_size=patch_size,
            merge_size=merge_size,
        )
    )
    requested_tokens = min(
        int(token_cap),
        max(int(native_tokens), int(minimum_tokens)),
    )
    minimum_applied = native_tokens < minimum_tokens
    cap_saturated = native_tokens > token_cap
    factor = int(patch_size * merge_size)

    if requested_tokens == native_tokens:
        grid_height = resized_height // factor
        grid_width = resized_width // factor
        native_grid_preserved = True
    else:
        grid_height, grid_width = grid_for_token_budget(
            width,
            height,
            requested_tokens,
        )
        native_grid_preserved = False
        if minimum_applied and grid_height * grid_width < minimum_tokens:
            for candidate_tokens in range(requested_tokens + 1, token_cap + 1):
                candidate_height, candidate_width = grid_for_token_budget(
                    width,
                    height,
                    candidate_tokens,
                )
                if candidate_height * candidate_width >= minimum_tokens:
                    requested_tokens = candidate_tokens
                    grid_height = candidate_height
                    grid_width = candidate_width
                    break

    realized_tokens = int(grid_height * grid_width)
    if realized_tokens > token_cap:
        raise AssertionError("native-cap planner exceeded the hard token ceiling")
    minimum_shortfall_tokens = max(0, int(minimum_tokens) - realized_tokens)
    if minimum_shortfall_tokens and requested_tokens != token_cap:
        raise AssertionError(
            "Fine floor was missed before exhausting the region token cap"
        )
    if native_grid_preserved and realized_tokens != native_tokens:
        raise AssertionError("native Qwen grid was not preserved exactly")

    return NativeTokenCapPlan(
        native_tokens=int(native_tokens),
        native_resized_height=int(resized_height),
        native_resized_width=int(resized_width),
        token_cap=int(token_cap),
        minimum_tokens=int(minimum_tokens),
        requested_tokens=int(requested_tokens),
        grid_height=int(grid_height),
        grid_width=int(grid_width),
        realized_tokens=realized_tokens,
        cap_saturated=bool(cap_saturated),
        minimum_applied=bool(minimum_applied),
        minimum_shortfall_tokens=minimum_shortfall_tokens,
        native_grid_preserved=bool(native_grid_preserved),
        unused_cap_tokens=int(token_cap - realized_tokens),
    )


__all__ = [
    "plan_anchor_fine_floors",
    "ContinuousFineFloorPlan",
    "NativeTokenCapPlan",
    "plan_continuous_fine_floor",
    "plan_native_token_cap",
]
