"""Core geometry and budget utilities for EviViT.

EviViT consumes one high-resolution source image.  A cheap global view and a
set of question-conditioned foveal views are implementation details inside the
vision encoder, not independent images in the user prompt.  This module keeps
the first prototype auditable: every foveal token has a source box, an explicit
token budget, and a coordinate in the original image.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from evivit_core.dense_evidence import normalize_policy_box


def selector_question_from_prompt(question: str, *, protocol: str) -> str:
    """Return the text that conditions the evidence allocator.

    PTEA was trained to localize evidence for a question, not for a shuffled
    multiple-choice answer list.  ``question_stem`` removes only the explicit
    benchmark choice instruction, so ordinary open-ended QA is unchanged.
    """

    value = str(question).strip()
    if protocol == "full_prompt":
        return value
    if protocol != "question_stem":
        raise ValueError(f"unknown selector question protocol: {protocol}")
    markers = (
        "\nChoose the correct option and answer with only its letter.",
        "\nChoose the correct option",
    )
    positions = [value.find(marker) for marker in markers if value.find(marker) >= 0]
    if positions:
        value = value[: min(positions)].rstrip()
    if not value:
        raise ValueError("selector question stem is empty")
    return value


@dataclass(frozen=True)
class RegionBudget:
    """One original-image region and its merged Qwen visual-token budget."""

    bbox: tuple[int, int, int, int]
    token_budget: int
    score: float
    role: str
    source_rank: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_region_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    policy: str = "portfolio",
    max_regions: int = 8,
) -> list[dict[str, Any]]:
    """Select the internal foveal views before allocating fine tokens.

    ``portfolio`` preserves the frozen EviMap Portfolio8 control.  ``focus_only``
    keeps only compact, residual-suppressed evidence modes; the always-present
    global branch is then solely responsible for broad context.  This is a
    strict redundancy control, not a new evidence selector.
    """

    if max_regions <= 0:
        raise ValueError("max_regions must be positive")
    rows = list(candidates)
    if policy == "portfolio":
        return rows[:max_regions]
    if policy == "focus_only":
        focused = [row for row in rows if str(row.get("portfolio_role", "")).endswith("_focus")]
        if not focused:
            raise ValueError("focus_only policy found no focus candidates")
        return focused[:max_regions]
    raise ValueError(f"unknown candidate policy: {policy}")


def _positive_score(candidate: dict[str, Any], *, mode: str) -> float:
    mass = max(float(candidate.get("mass", 0.0) or 0.0), 1e-8)
    area = max(float(candidate.get("area_ratio", 0.0) or 0.0), 1e-6)
    if mode == "uniform":
        return 1.0
    if mode == "evidence":
        # Mass preserves recall; the weak density term prevents broad context
        # boxes from consuming the entire fine-resolution budget.
        return mass**0.75 / area**0.15
    raise ValueError(f"unknown allocation mode: {mode}")


def allocate_region_budgets(
    candidates: Sequence[dict[str, Any]],
    *,
    total_tokens: int,
    max_regions: int = 8,
    minimum_tokens: int = 32,
    mode: str = "evidence",
) -> list[RegionBudget]:
    """Allocate a fixed fine-token budget across evidence regions.

    Allocation is deterministic and integer exact: the returned budgets sum to
    ``total_tokens``.  The caller remains responsible for the separate global
    view, which is deliberately never removed.
    """

    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")
    if max_regions <= 0 or minimum_tokens <= 0:
        raise ValueError("max_regions and minimum_tokens must be positive")
    selected = list(candidates[:max_regions])
    if not selected:
        return []
    while len(selected) * minimum_tokens > total_tokens:
        selected.pop()
    if not selected:
        raise ValueError("fine-token budget is too small for one region")

    scores = np.asarray([_positive_score(row, mode=mode) for row in selected], dtype=np.float64)
    scores = scores / scores.sum()
    remaining = total_tokens - len(selected) * minimum_tokens
    raw = scores * remaining
    extras = np.floor(raw).astype(np.int64)
    remainder = int(remaining - int(extras.sum()))
    if remainder:
        order = np.argsort(-(raw - extras), kind="stable")
        extras[order[:remainder]] += 1

    budgets: list[RegionBudget] = []
    for index, (candidate, extra, score) in enumerate(zip(selected, extras, scores), 1):
        box = tuple(normalize_policy_box(candidate["bbox"]))
        budgets.append(
            RegionBudget(
                bbox=box,
                token_budget=minimum_tokens + int(extra),
                score=float(score),
                role=str(candidate.get("portfolio_role", f"candidate_{index}")),
                source_rank=index,
            )
        )
    if sum(row.token_budget for row in budgets) != total_tokens:
        raise AssertionError("integer token allocation lost budget")
    return budgets


def grid_for_token_budget(
    width: int,
    height: int,
    token_budget: int,
    *,
    minimum_side_tokens: int = 2,
) -> tuple[int, int]:
    """Choose a near-aspect-preserving merged-token grid within a budget."""

    if width <= 0 or height <= 0 or token_budget <= 0:
        raise ValueError("width, height, and token_budget must be positive")
    if token_budget < minimum_side_tokens**2:
        raise ValueError("token budget is smaller than the minimum grid")
    aspect = width / height
    grid_w = max(minimum_side_tokens, int(round(math.sqrt(token_budget * aspect))))
    grid_h = max(minimum_side_tokens, int(token_budget // grid_w))
    while grid_h * grid_w > token_budget:
        if grid_w >= grid_h:
            grid_w -= 1
        else:
            grid_h -= 1
    # Spend safe residual tokens when another row or column still fits.
    improved = True
    while improved:
        improved = False
        choices = []
        if (grid_w + 1) * grid_h <= token_budget:
            choices.append((grid_h, grid_w + 1))
        if grid_w * (grid_h + 1) <= token_budget:
            choices.append((grid_h + 1, grid_w))
        if choices:
            current_error = abs(math.log((grid_w / grid_h) / aspect))
            best = min(choices, key=lambda item: abs(math.log((item[1] / item[0]) / aspect)))
            best_error = abs(math.log((best[1] / best[0]) / aspect))
            if best_error <= current_error:
                grid_h, grid_w = best
                improved = True
    return grid_h, grid_w


def token_centers_in_original(
    bbox: Sequence[float],
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    """Return N x 2 normalized (x, y) centers for row-major region tokens."""

    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid dimensions must be positive")
    x1, y1, x2, y2 = normalize_policy_box(bbox)
    xs = x1 + (np.arange(grid_w, dtype=np.float64) + 0.5) / grid_w * (x2 - x1)
    ys = y1 + (np.arange(grid_h, dtype=np.float64) + 0.5) / grid_h * (y2 - y1)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1) / 1000.0


def multiresolution_token_metadata(
    grid_shapes: Sequence[Sequence[int]],
    boxes: Sequence[Sequence[float]],
) -> dict[str, np.ndarray]:
    """Map global and foveal token grids back onto the one source image."""

    if len(grid_shapes) != len(boxes):
        raise ValueError("grid_shapes and boxes must have equal length")
    centers: list[np.ndarray] = []
    levels: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    region_ids: list[np.ndarray] = []
    for index, (shape, box) in enumerate(zip(grid_shapes, boxes)):
        if len(shape) != 2:
            raise ValueError("every grid shape must be [height, width]")
        grid_h, grid_w = int(shape[0]), int(shape[1])
        xy = token_centers_in_original(box, grid_h, grid_w)
        x1, y1, x2, y2 = normalize_policy_box(box)
        area = max(1e-8, (x2 - x1) * (y2 - y1) / 1_000_000.0)
        centers.append(xy)
        levels.append(np.full(len(xy), 0 if index == 0 else 1, dtype=np.int64))
        scales.append(np.full(len(xy), math.sqrt(area), dtype=np.float64))
        region_ids.append(np.full(len(xy), index, dtype=np.int64))
    return {
        "centers_xy": np.concatenate(centers, axis=0),
        "levels": np.concatenate(levels, axis=0),
        "scales": np.concatenate(scales, axis=0),
        "region_ids": np.concatenate(region_ids, axis=0),
    }


def multiresolution_fusion_indices(
    metadata: dict[str, np.ndarray],
    boxes: Sequence[Sequence[float]],
    *,
    mode: str = "append",
) -> np.ndarray:
    """Choose and order tokens for one multiresolution visual span.

    ``append`` reproduces the frozen control: the complete global grid is
    followed by all fine grids. ``parent_interleave`` retains every token but
    groups each fine token after its nearest global parent cell, preserving a
    coarse-to-fine topology in the unified visual span. ``replace_coarse``
    removes global tokens whose centers are covered by any fine region, so
    local detail replaces rather than duplicates coarse evidence.
    ``replace_raster`` applies the same replacement and then restores
    original-image raster order. These are representation changes, not
    learned gates or post-hoc routers.
    """

    centers = np.asarray(metadata["centers_xy"], dtype=np.float64)
    region_ids = np.asarray(metadata["region_ids"], dtype=np.int64)
    levels = np.asarray(metadata["levels"], dtype=np.int64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers_xy must have shape [tokens, 2]")
    if len(region_ids) != len(centers) or len(levels) != len(centers):
        raise ValueError("metadata arrays must have the same token length")
    if mode == "append":
        return np.arange(len(centers), dtype=np.int64)
    if mode == "parent_interleave":
        global_centers = centers[region_ids == 0]
        if not len(global_centers):
            raise ValueError("parent_interleave requires global tokens")
        global_x = np.unique(global_centers[:, 0])
        global_y = np.unique(global_centers[:, 1])
        if len(global_x) * len(global_y) != len(global_centers):
            raise ValueError("global tokens do not form a complete spatial grid")

        # Assign every token to its nearest global-grid center. Global tokens
        # are exact members of this grid; fine tokens inherit an auditable
        # parent without adding a learned router or changing token count.
        parent_x = np.abs(centers[:, 0, None] - global_x[None, :]).argmin(axis=1)
        parent_y = np.abs(centers[:, 1, None] - global_y[None, :]).argmin(axis=1)
        order = np.lexsort(
            (
                region_ids,
                centers[:, 0],
                centers[:, 1],
                levels,
                parent_x,
                parent_y,
            )
        )
        if len(order) != len(centers) or len(np.unique(order)) != len(centers):
            raise AssertionError("parent_interleave did not produce a permutation")
        return order.astype(np.int64)
    if mode not in {"replace_coarse", "replace_raster"}:
        raise ValueError(f"unknown fusion mode: {mode}")
    if not boxes:
        raise ValueError("at least the global image box is required")

    keep = np.ones(len(centers), dtype=bool)
    global_mask = region_ids == 0
    for box in boxes[1:]:
        x1, y1, x2, y2 = normalize_policy_box(box)
        x1, y1, x2, y2 = (value / 1000.0 for value in (x1, y1, x2, y2))
        covered = (
            (centers[:, 0] >= x1)
            & (centers[:, 0] <= x2)
            & (centers[:, 1] >= y1)
            & (centers[:, 1] <= y2)
        )
        keep &= ~(global_mask & covered)

    indices = np.flatnonzero(keep).astype(np.int64)
    if mode == "replace_raster":
        # Primary order follows the original image. At the same location, keep
        # coarse context before fine detail to preserve a coarse-to-fine cue.
        order = np.lexsort(
            (levels[indices], centers[indices, 0], centers[indices, 1])
        )
        indices = indices[order]
    return indices


def quantize_mrope_coordinates(
    metadata: dict[str, np.ndarray],
    *,
    coordinate_bins: int = 128,
    scale_bins: int = 8,
    mode: str = "original_xy_scale_t",
) -> np.ndarray:
    """Build temporal/height/width coordinates for Qwen MRoPE.

    ``original_xy_scale_t`` is the first Stage-1 prototype: it repurposes the
    temporal axis to distinguish global and foveal scales.  ``original_xy_zero_t``
    is the strict static-image control: all temporal coordinates remain zero and
    only original-image x/y positions are changed.  Comparing the two isolates
    whether out-of-distribution temporal coordinates, rather than evidence
    allocation, are responsible for a quality gap.
    """

    if coordinate_bins < 2 or scale_bins < 2:
        raise ValueError("coordinate_bins and scale_bins must be at least two")
    xy = np.asarray(metadata["centers_xy"], dtype=np.float64)
    levels = np.asarray(metadata["levels"], dtype=np.int64)
    scales = np.asarray(metadata["scales"], dtype=np.float64)
    x = np.rint(np.clip(xy[:, 0], 0, 1) * (coordinate_bins - 1)).astype(np.int64)
    y = np.rint(np.clip(xy[:, 1], 0, 1) * (coordinate_bins - 1)).astype(np.int64)
    # Smaller spatial extent means a finer foveal view and receives a larger
    # scale index.  Global tokens are always level zero.
    scale = np.rint((1.0 - np.clip(scales, 0, 1)) * (scale_bins - 1)).astype(np.int64)
    if mode == "original_xy_scale_t":
        temporal = np.where(levels == 0, 0, 1 + scale)
    elif mode == "original_xy_zero_t":
        temporal = np.zeros_like(levels)
    else:
        raise ValueError(f"unknown coordinate mode: {mode}")
    return np.stack([temporal, y, x], axis=0)


def realized_budget(rows: Iterable[RegionBudget]) -> int:
    return sum(row.token_budget for row in rows)
