"""Native-budget global/local patch exchange for EviViT-v5.

The frozen Qwen PatchEmbed remains a 16x16 spatial convolution.  EviViT-v5
changes the *sampling scale* before that convolution: a coarser Global view
acts like a larger patch in original-image coordinates, while crops read from
the true source image at a finer sampling scale.  Every per-sample plan is
bounded by the number of post-merger tokens that unmodified Qwen would use
under the same min/max-pixel contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

from evivit_core.dense_evidence import normalize_policy_box
from evivit_core.evivit import grid_for_token_budget


def continuous_soft_floor_tokens(
    native_tokens: int,
    *,
    minimum_tokens: int,
    maximum_tokens: int,
    ramp_start_tokens: int,
    ramp_end_tokens: int,
) -> int:
    """Continuously interpolate a small-image floor from native demand.

    Native demand at or below ``ramp_start_tokens`` keeps the conservative
    floor.  Demand at or above ``ramp_end_tokens`` receives the high-detail
    floor, with a linear, per-sample transition in between.  This is not a
    dataset router: the only input is the image's Qwen-native token count.
    """

    if native_tokens <= 0:
        raise ValueError("native_tokens must be positive")
    if minimum_tokens <= 0 or maximum_tokens < minimum_tokens:
        raise ValueError("invalid continuous soft-floor interval")
    if ramp_start_tokens < 0 or ramp_end_tokens <= ramp_start_tokens:
        raise ValueError("invalid continuous soft-floor ramp")
    alpha = min(
        1.0,
        max(
            0.0,
            (native_tokens - ramp_start_tokens)
            / float(ramp_end_tokens - ramp_start_tokens),
        ),
    )
    return int(round(minimum_tokens + alpha * (maximum_tokens - minimum_tokens)))


@dataclass(frozen=True)
class PatchExchangeNativePlan:
    """Torch-free reproduction of Qwen's aligned still-image geometry."""

    source_width: int
    source_height: int
    source_pixels: int
    minimum_pixels: int
    maximum_pixels: int
    resized_width: int
    resized_height: int
    resized_pixels: int
    grid_width: int
    grid_height: int
    realized_tokens: int
    native_grid_preserved: bool
    maximum_applied: bool
    minimum_applied: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PatchExchangeGlobalPlan:
    """Pixel-first plan for the B16 reference and the coarser Global view."""

    native_plan: PatchExchangeNativePlan
    global_scale: float
    requested_maximum_pixels: int
    requested_token_ceiling: int
    grid_height: int
    grid_width: int
    realized_tokens: int
    effective_scale: float
    base_realized_tokens: int
    soft_floor_tokens: int
    soft_floor_global_share: float
    budget_basis_tokens: int
    soft_floor_added_global_tokens: int
    preserve_native_global: bool
    native_global_anchor: bool
    native_preservation_weight: float
    native_preserved_target_tokens: int

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = asdict(self)
        payload["native_plan"] = self.native_plan.to_dict()
        return payload


@dataclass(frozen=True)
class PatchExchangeFineViewPlan:
    """One selected original-image crop and its p8-like token ceiling."""

    source_index: int
    bbox: tuple[int, int, int, int]
    source_width: int
    source_height: int
    area_ratio: float
    priority_weight: float
    requested_pixels: int
    desired_token_ceiling: int
    allocated_token_ceiling: int
    grid_height: int
    grid_width: int
    realized_tokens: int
    effective_scale: float
    dropped: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PatchExchangeFinePlan:
    """Fine-view plan after the per-image native-token safety constraint."""

    local_scale: float
    total_cap_ratio: float
    native_global_anchor: bool
    additive_fine_ratio: float
    additive_fine_max_tokens: int
    soft_floor_tokens: int
    shared_fine_token_override: int | None
    budget_basis_tokens: int
    total_token_ceiling: int
    available_fine_tokens: int
    desired_fine_tokens: int
    allocated_fine_token_ceiling: int
    realized_fine_tokens: int
    realized_total_tokens: int
    unused_tokens: int
    views: tuple[PatchExchangeFineViewPlan, ...]

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = asdict(self)
        payload["views"] = [view.to_dict() for view in self.views]
        return payload


def _minimum_merged_tokens(
    minimum_pixels: int,
    *,
    patch_size: int,
    merge_size: int,
) -> int:
    stride = int(patch_size * merge_size)
    return max(1, int(math.ceil(minimum_pixels / float(stride * stride))))


def _plan_qwen_native_view(
    width: int,
    height: int,
    *,
    minimum_pixels: int,
    maximum_pixels: int,
    patch_size: int,
    merge_size: int,
) -> PatchExchangeNativePlan:
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    if minimum_pixels <= 0 or maximum_pixels < minimum_pixels:
        raise ValueError("invalid Qwen pixel interval")
    factor = int(patch_size * merge_size)
    if factor <= 0:
        raise ValueError("patch_size * merge_size must be positive")
    if max(width, height) / min(width, height) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > maximum_pixels:
        beta = math.sqrt((height * width) / maximum_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_height * resized_width < minimum_pixels:
        beta = math.sqrt(minimum_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    source_pixels = int(width * height)
    resized_pixels = int(resized_height * resized_width)
    return PatchExchangeNativePlan(
        source_width=int(width),
        source_height=int(height),
        source_pixels=source_pixels,
        minimum_pixels=int(minimum_pixels),
        maximum_pixels=int(maximum_pixels),
        resized_width=int(resized_width),
        resized_height=int(resized_height),
        resized_pixels=resized_pixels,
        grid_width=int(resized_width // factor),
        grid_height=int(resized_height // factor),
        realized_tokens=int((resized_height // factor) * (resized_width // factor)),
        native_grid_preserved=bool(
            source_pixels <= maximum_pixels and source_pixels >= minimum_pixels
        ),
        maximum_applied=bool(source_pixels > maximum_pixels),
        minimum_applied=bool(source_pixels < minimum_pixels),
    )


def plan_patch_exchange_global_view(
    width: int,
    height: int,
    *,
    global_scale: float = 2.0,
    minimum_pixels: int = 4096,
    maximum_pixels: int = 16 * 1024 * 1024,
    patch_size: int = 16,
    merge_size: int = 2,
    soft_floor_tokens: int = 0,
    soft_floor_global_share: float = 0.0,
    preserve_native_global: bool = False,
    native_global_anchor: bool = False,
) -> PatchExchangeGlobalPlan:
    """Plan a p16-preserving Global view with an effective larger patch.

    ``global_scale=2`` downsamples both aligned spatial dimensions by roughly
    two, so the frozen p16 PatchEmbed behaves like p32 in source coordinates.
    The pixel maximum is derived from the sample's own B16-aligned geometry,
    not from a training-set reference size or a fixed token template.
    """

    if not math.isfinite(global_scale) or global_scale < 1.0:
        raise ValueError("global_scale must be finite and at least 1")
    if soft_floor_tokens < 0:
        raise ValueError("soft_floor_tokens must be non-negative")
    if not 0.0 <= soft_floor_global_share <= 1.0:
        raise ValueError("soft_floor_global_share must be in [0, 1]")
    if preserve_native_global and soft_floor_tokens <= 0:
        raise ValueError(
            "preserving the native Global grid requires a positive soft floor"
        )
    native = _plan_qwen_native_view(
        width,
        height,
        minimum_pixels=minimum_pixels,
        maximum_pixels=maximum_pixels,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    if native_global_anchor:
        # EviViT-v7-P4 NativeAnchor: B16's native Global grid is an identity
        # anchor.  Fine evidence is additive and therefore must never be paid
        # for by deleting or coarsening native Global tokens.
        return PatchExchangeGlobalPlan(
            native_plan=native,
            global_scale=1.0,
            requested_maximum_pixels=int(maximum_pixels),
            requested_token_ceiling=int(native.realized_tokens),
            grid_height=int(native.grid_height),
            grid_width=int(native.grid_width),
            realized_tokens=int(native.realized_tokens),
            effective_scale=1.0,
            base_realized_tokens=int(native.realized_tokens),
            soft_floor_tokens=0,
            soft_floor_global_share=0.0,
            budget_basis_tokens=int(native.realized_tokens),
            soft_floor_added_global_tokens=0,
            preserve_native_global=True,
            native_global_anchor=True,
            native_preservation_weight=1.0,
            native_preserved_target_tokens=int(native.realized_tokens),
        )
    requested_pixels = max(
        int(minimum_pixels),
        int(math.floor(native.resized_pixels / (global_scale * global_scale))),
    )
    view = _plan_qwen_native_view(
        width,
        height,
        minimum_pixels=minimum_pixels,
        maximum_pixels=requested_pixels,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    minimum_tokens = _minimum_merged_tokens(
        minimum_pixels,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    # The manually resized view is passed through Qwen's processor once more.
    # A rare smart-resize boundary (for example an 85x87 image with min=max
    # 4096) can first produce a grid below the minimum, which the second pass
    # would enlarge.  Emit a processor fixed point so planned and encoded
    # grids are identical.
    if view.realized_tokens < minimum_tokens:
        base_grid_height, base_grid_width = grid_for_token_budget(
            width,
            height,
            minimum_tokens,
        )
        base_realized_tokens = int(base_grid_height * base_grid_width)
    else:
        base_grid_height = int(view.grid_height)
        base_grid_width = int(view.grid_width)
        base_realized_tokens = int(view.realized_tokens)

    budget_basis = int(
        round(math.hypot(float(native.realized_tokens), float(soft_floor_tokens)))
    )
    added_budget = max(0, budget_basis - native.realized_tokens)
    requested_added_global = int(round(added_budget * soft_floor_global_share))
    balanced_global_target = min(
        budget_basis,
        base_realized_tokens + requested_added_global,
    )
    native_preservation_weight = 0.0
    native_preserved_target = int(base_realized_tokens)
    if preserve_native_global:
        # EviViT-v7-P2: when native visual demand is small, exchanging an
        # already sparse Global grid for local evidence can remove more
        # context than the Fine path adds.  Preserve the native Global grid
        # continuously for such samples, while recovering v5/v6's coarser
        # Global/Fine exchange as native demand grows.  This is a continuous
        # rational envelope rather than a low/medium/high routing table.
        floor_sq = float(soft_floor_tokens * soft_floor_tokens)
        native_sq = float(native.realized_tokens * native.realized_tokens)
        native_preservation_weight = floor_sq / max(1.0, floor_sq + native_sq)
        native_preserved_target = int(
            round(
                balanced_global_target
                + native_preservation_weight
                * (native.realized_tokens - balanced_global_target)
            )
        )
        requested_global_tokens = min(
            budget_basis,
            # Never take Global capacity away from v6 Balanced SoftFloor.
            # Very small images already receive more Global tokens than their
            # native grid through that floor; preserving context means keeping
            # that gain, not shrinking it back to the tiny native grid.
            max(balanced_global_target, native_preserved_target),
        )
    else:
        requested_global_tokens = balanced_global_target
    if requested_global_tokens > base_realized_tokens:
        grid_height, grid_width = grid_for_token_budget(
            width,
            height,
            requested_global_tokens,
        )
        realized_tokens = int(grid_height * grid_width)
    else:
        grid_height = int(base_grid_height)
        grid_width = int(base_grid_width)
        realized_tokens = int(base_realized_tokens)
    if realized_tokens > budget_basis:
        raise AssertionError("Balanced PatchExchange Global exceeded budget basis")
    effective_scale = math.sqrt(
        native.realized_tokens / float(max(1, realized_tokens))
    )
    return PatchExchangeGlobalPlan(
        native_plan=native,
        global_scale=float(global_scale),
        requested_maximum_pixels=int(requested_pixels),
        requested_token_ceiling=int(
            max(
                _minimum_merged_tokens(
                    minimum_pixels,
                    patch_size=patch_size,
                    merge_size=merge_size,
                ),
                math.floor(
                    native.realized_tokens / (global_scale * global_scale)
                ),
            )
        ),
        grid_height=int(grid_height),
        grid_width=int(grid_width),
        realized_tokens=int(realized_tokens),
        effective_scale=float(effective_scale),
        base_realized_tokens=int(base_realized_tokens),
        soft_floor_tokens=int(soft_floor_tokens),
        soft_floor_global_share=float(soft_floor_global_share),
        budget_basis_tokens=int(budget_basis),
        soft_floor_added_global_tokens=int(
            max(0, realized_tokens - base_realized_tokens)
        ),
        preserve_native_global=bool(preserve_native_global),
        native_global_anchor=False,
        native_preservation_weight=float(native_preservation_weight),
        native_preserved_target_tokens=int(native_preserved_target),
    )


def _weighted_capped_allocation(
    desired: Sequence[int],
    weights: Sequence[float],
    *,
    total: int,
    minimum: int,
) -> list[int]:
    """Allocate integer ceilings deterministically without exceeding demand."""

    if len(desired) != len(weights):
        raise ValueError("desired and weights must have equal length")
    if total <= 0 or not desired:
        return [0] * len(desired)
    desired_values = [max(0, int(value)) for value in desired]
    positive_order = [index for index, value in enumerate(desired_values) if value > 0]
    # Preserve the selector order.  When a tiny image cannot support every
    # legal Fine grid, lower-priority trailing regions are omitted.
    active: list[int] = []
    remaining = int(total)
    for index in positive_order:
        floor = min(desired_values[index], int(minimum))
        if floor <= remaining:
            active.append(index)
            remaining -= floor
    allocated = [0] * len(desired_values)
    for index in active:
        allocated[index] = min(desired_values[index], int(minimum))
    if remaining <= 0 or not active:
        return allocated

    capacity = {
        index: desired_values[index] - allocated[index] for index in active
    }
    active_set = {index for index in active if capacity[index] > 0}
    positive_weights = {
        index: max(float(weights[index]), 1e-8) for index in active
    }
    while remaining > 0 and active_set:
        normalizer = sum(positive_weights[index] for index in active_set)
        raw = {
            index: remaining * positive_weights[index] / normalizer
            for index in active_set
        }
        granted = {
            index: min(capacity[index], int(math.floor(raw[index])))
            for index in active_set
        }
        progress = sum(granted.values())
        if progress == 0:
            order = sorted(
                active_set,
                key=lambda index: (
                    -(raw[index] - math.floor(raw[index])),
                    -positive_weights[index],
                    index,
                ),
            )
            for index in order:
                if remaining <= 0:
                    break
                allocated[index] += 1
                capacity[index] -= 1
                remaining -= 1
                if capacity[index] <= 0:
                    active_set.discard(index)
            continue
        for index, value in granted.items():
            allocated[index] += value
            capacity[index] -= value
            remaining -= value
        active_set = {index for index in active_set if capacity[index] > 0}
    return allocated


def plan_patch_exchange_fine_views(
    source_width: int,
    source_height: int,
    boxes: Sequence[Sequence[float]],
    *,
    global_plan: PatchExchangeGlobalPlan,
    priority_weights: Sequence[float] | None = None,
    local_scale: float = 2.0,
    total_cap_ratio: float = 1.0,
    soft_floor_tokens: int = 0,
    fill_soft_floor: bool = False,
    minimum_view_pixels: int = 4096,
    maximum_view_pixels: int = 16 * 1024 * 1024,
    native_global_anchor: bool = False,
    additive_fine_ratio: float = 0.5,
    additive_fine_max_tokens: int = 2048,
    shared_fine_token_override: int | None = None,
    patch_size: int = 16,
    merge_size: int = 2,
) -> PatchExchangeFinePlan:
    """Plan p8-like original-image rereads under a native-Qwen total cap.

    Desired Fine pixels are computed from the B16-aligned source pixels and
    each box's original-image area.  Crops are later taken from the true source
    image; this function only determines their aligned Qwen token grids.
    """

    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    if not math.isfinite(local_scale) or local_scale < 1.0:
        raise ValueError("local_scale must be finite and at least 1")
    if not math.isfinite(total_cap_ratio) or total_cap_ratio <= 0:
        raise ValueError("total_cap_ratio must be finite and positive")
    if maximum_view_pixels < minimum_view_pixels or minimum_view_pixels <= 0:
        raise ValueError("invalid Fine pixel interval")
    if soft_floor_tokens < 0:
        raise ValueError("soft_floor_tokens must be non-negative")
    if not math.isfinite(additive_fine_ratio) or additive_fine_ratio < 0:
        raise ValueError("additive_fine_ratio must be finite and non-negative")
    if additive_fine_max_tokens < 0:
        raise ValueError("additive_fine_max_tokens must be non-negative")
    if shared_fine_token_override is not None and shared_fine_token_override < 0:
        raise ValueError("shared Fine-token override must be non-negative")
    if native_global_anchor and not global_plan.native_global_anchor:
        raise ValueError("native-global Fine planning requires an anchored Global plan")
    if priority_weights is None:
        priorities = [1.0] * len(boxes)
    else:
        priorities = [float(value) for value in priority_weights]
    if len(priorities) != len(boxes):
        raise ValueError("priority_weights must match boxes")

    stride = int(patch_size * merge_size)
    minimum_tokens = _minimum_merged_tokens(
        minimum_view_pixels,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    normalized_boxes: list[tuple[int, int, int, int]] = []
    crop_sizes: list[tuple[int, int]] = []
    area_ratios: list[float] = []
    requested_pixels: list[int] = []
    desired_tokens: list[int] = []
    for raw_box in boxes:
        x1, y1, x2, y2 = normalize_policy_box(raw_box)
        box = (int(x1), int(y1), int(x2), int(y2))
        normalized_boxes.append(box)
        area = ((x2 - x1) / 1000.0) * ((y2 - y1) / 1000.0)
        area_ratios.append(float(area))
        crop_width = max(1, int(round(source_width * (x2 - x1) / 1000.0)))
        crop_height = max(1, int(round(source_height * (y2 - y1) / 1000.0)))
        crop_sizes.append((crop_width, crop_height))
        target_pixels = int(
            round(
                global_plan.native_plan.resized_pixels
                * area
                * local_scale
                * local_scale
            )
        )
        target_pixels = min(
            int(maximum_view_pixels),
            max(int(minimum_view_pixels), target_pixels),
        )
        requested_pixels.append(target_pixels)
        desired_tokens.append(
            max(minimum_tokens, int(math.floor(target_pixels / (stride * stride))))
        )

    # Optional continuous lower envelope for small images.  At zero this is
    # exactly Stage-A's strict native-B16 cap.  For N << floor it approaches
    # the floor smoothly; for N >> floor its relative overhead vanishes.  The
    # added capacity is assigned only to raw-image Fine rereads, while the
    # p24 Global grid remains unchanged.
    if native_global_anchor:
        additive_fine = min(
            int(additive_fine_max_tokens),
            int(round(global_plan.native_plan.realized_tokens * additive_fine_ratio)),
        )
        budget_basis = int(global_plan.native_plan.realized_tokens + additive_fine)
        total_ceiling = int(global_plan.realized_tokens + additive_fine)
    else:
        budget_basis = int(
            round(
                math.hypot(
                    float(global_plan.native_plan.realized_tokens),
                    float(soft_floor_tokens),
                )
            )
        )
        total_ceiling = max(
            global_plan.realized_tokens,
            int(math.floor(budget_basis * total_cap_ratio)),
        )
    if shared_fine_token_override is not None:
        # Multi-image EviViT keeps every per-image Global plan intact, then
        # assigns this image an explicit share of one cross-image Fine pool.
        # The override changes neither the crop demand nor the Global grid.
        available_fine = int(shared_fine_token_override)
        total_ceiling = int(global_plan.realized_tokens + available_fine)
    else:
        available_fine = max(0, total_ceiling - global_plan.realized_tokens)
    allocation_desired = list(desired_tokens)
    if fill_soft_floor and soft_floor_tokens > 0 and allocation_desired:
        added_basis = max(
            0,
            budget_basis - global_plan.native_plan.realized_tokens,
        )
        added_global = max(
            0,
            global_plan.realized_tokens - global_plan.base_realized_tokens,
        )
        requested_added_fine = max(0, added_basis - added_global)
        maximum_tokens_per_view = max(
            minimum_tokens,
            int(math.floor(maximum_view_pixels / float(stride * stride))),
        )
        extra_capacity = [
            max(0, maximum_tokens_per_view - value)
            for value in allocation_desired
        ]
        extra = _weighted_capped_allocation(
            extra_capacity,
            priorities,
            total=min(requested_added_fine, sum(extra_capacity)),
            minimum=0,
        )
        allocation_desired = [
            value + increment
            for value, increment in zip(allocation_desired, extra)
        ]

    allocated = _weighted_capped_allocation(
        allocation_desired,
        priorities,
        total=available_fine,
        minimum=minimum_tokens,
    )

    views: list[PatchExchangeFineViewPlan] = []
    for index, (box, crop_size, area, pixels, desired, allocation) in enumerate(
        zip(
            normalized_boxes,
            crop_sizes,
            area_ratios,
            requested_pixels,
            desired_tokens,
            allocated,
        )
    ):
        if allocation <= 0:
            grid_h = grid_w = realized = 0
        else:
            grid_h, grid_w = grid_for_token_budget(
                crop_size[0], crop_size[1], allocation
            )
            realized = int(grid_h * grid_w)
        denominator = max(1e-12, global_plan.native_plan.realized_tokens * area)
        effective_scale = math.sqrt(realized / denominator) if realized else 0.0
        views.append(
            PatchExchangeFineViewPlan(
                source_index=index,
                bbox=box,
                source_width=int(crop_size[0]),
                source_height=int(crop_size[1]),
                area_ratio=float(area),
                priority_weight=float(priorities[index]),
                requested_pixels=int(pixels),
                desired_token_ceiling=int(allocation_desired[index]),
                allocated_token_ceiling=int(allocation),
                grid_height=int(grid_h),
                grid_width=int(grid_w),
                realized_tokens=int(realized),
                effective_scale=float(effective_scale),
                dropped=bool(realized == 0),
            )
        )
    realized_fine = sum(view.realized_tokens for view in views)
    realized_total = global_plan.realized_tokens + realized_fine
    if realized_total > total_ceiling:
        raise AssertionError("PatchExchange exceeded its native-token ceiling")
    return PatchExchangeFinePlan(
        local_scale=float(local_scale),
        total_cap_ratio=float(total_cap_ratio),
        native_global_anchor=bool(native_global_anchor),
        additive_fine_ratio=float(additive_fine_ratio),
        additive_fine_max_tokens=int(additive_fine_max_tokens),
        soft_floor_tokens=int(soft_floor_tokens),
        shared_fine_token_override=(
            int(shared_fine_token_override)
            if shared_fine_token_override is not None
            else None
        ),
        budget_basis_tokens=int(budget_basis),
        total_token_ceiling=int(total_ceiling),
        available_fine_tokens=int(available_fine),
        desired_fine_tokens=int(sum(allocation_desired)),
        allocated_fine_token_ceiling=int(sum(allocated)),
        realized_fine_tokens=int(realized_fine),
        realized_total_tokens=int(realized_total),
        unused_tokens=int(total_ceiling - realized_total),
        views=tuple(views),
    )


__all__ = [
    "PatchExchangeFinePlan",
    "PatchExchangeFineViewPlan",
    "PatchExchangeGlobalPlan",
    "PatchExchangeNativePlan",
    "continuous_soft_floor_tokens",
    "plan_patch_exchange_fine_views",
    "plan_patch_exchange_global_view",
]
