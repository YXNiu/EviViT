"""Adaptive geometry decoding for question-conditioned EviViT evidence maps.

The frozen v4 decoder searches only 0.20/0.25 square windows.  This module
instead grows a rectangle from each residual evidence mode until it captures
the mode's probability mass and its boundary becomes weak.  All operations are
performed on the small PTEA probability grid; no candidate image is encoded.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Sequence

import numpy as np

from evivit_core.dense_evidence import (
    expand_policy_box,
    map_mass,
    normalize_distribution,
    normalize_policy_box,
    policy_iou,
)


GridBox = tuple[int, int, int, int]  # x1, y1, x2, y2; x2/y2 are exclusive.


def _integral(array: np.ndarray) -> np.ndarray:
    return np.pad(array.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def _rect_sum(integral: np.ndarray, box: GridBox) -> float:
    x1, y1, x2, y2 = box
    return float(
        integral[y2, x2]
        - integral[y1, x2]
        - integral[y2, x1]
        + integral[y1, x1]
    )


def _grid_to_policy(box: GridBox, *, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    return normalize_policy_box(
        [
            x1 / width * 1000,
            y1 / height * 1000,
            x2 / width * 1000,
            y2 / height * 1000,
        ]
    )


def _policy_to_grid(box: Sequence[float], *, width: int, height: int) -> GridBox:
    x1, y1, x2, y2 = normalize_policy_box(box)
    gx1 = min(width - 1, max(0, int(math.floor(x1 / 1000 * width))))
    gy1 = min(height - 1, max(0, int(math.floor(y1 / 1000 * height))))
    gx2 = min(width, max(gx1 + 1, int(math.ceil(x2 / 1000 * width))))
    gy2 = min(height, max(gy1 + 1, int(math.ceil(y2 / 1000 * height))))
    return gx1, gy1, gx2, gy2


def _connected_basin(
    residual: np.ndarray,
    peak_y: int,
    peak_x: int,
    *,
    relative_floor: float,
) -> np.ndarray:
    """Return the 8-connected high-response basin containing one peak."""

    peak = float(residual[peak_y, peak_x])
    positive = residual[residual > 0]
    background = float(np.mean(positive)) if positive.size else 0.0
    threshold = max(peak * relative_floor, background * 0.25)
    active = residual >= threshold
    active[peak_y, peak_x] = True
    basin = np.zeros_like(active, dtype=bool)
    queue: deque[tuple[int, int]] = deque([(peak_y, peak_x)])
    basin[peak_y, peak_x] = True
    height, width = residual.shape
    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dx or dy):
                    continue
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and active[ny, nx]
                    and not basin[ny, nx]
                ):
                    basin[ny, nx] = True
                    queue.append((ny, nx))
    return basin


def _normalized_entropy(values: np.ndarray) -> float:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[positive > 0]
    if positive.size <= 1:
        return 0.0
    probability = positive / positive.sum()
    entropy = float(-(probability * np.log(probability + 1e-12)).sum())
    return entropy / math.log(positive.size)


def _moves(box: GridBox, *, width: int, height: int) -> list[GridBox]:
    x1, y1, x2, y2 = box
    moves: list[GridBox] = []
    if x1 > 0:
        moves.append((x1 - 1, y1, x2, y2))
    if x2 < width:
        moves.append((x1, y1, x2 + 1, y2))
    if y1 > 0:
        moves.append((x1, y1 - 1, x2, y2))
    if y2 < height:
        moves.append((x1, y1, x2, y2 + 1))
    return moves


def _ring_mass(integral: np.ndarray, box: GridBox, *, width: int, height: int) -> float:
    x1, y1, x2, y2 = box
    outer = (max(0, x1 - 1), max(0, y1 - 1), min(width, x2 + 1), min(height, y2 + 1))
    return max(0.0, _rect_sum(integral, outer) - _rect_sum(integral, box))


def _expand_grid_box(box: GridBox, factor: float, *, width: int, height: int) -> GridBox:
    policy = _grid_to_policy(box, width=width, height=height)
    return _policy_to_grid(
        expand_policy_box(policy, factor),
        width=width,
        height=height,
    )


def _grow_mode(
    residual: np.ndarray,
    peak_y: int,
    peak_x: int,
    *,
    basin_relative_floor: float,
    minimum_basin_coverage: float,
    entropy_coverage_gain: float,
    boundary_density_ratio: float,
    context_factor_min: float,
    context_factor_entropy_gain: float,
    maximum_area_ratio: float,
) -> dict[str, Any]:
    height, width = residual.shape
    integral = _integral(residual)
    basin = _connected_basin(
        residual,
        peak_y,
        peak_x,
        relative_floor=basin_relative_floor,
    )
    basin_mass = float(residual[basin].sum())
    basin_entropy = _normalized_entropy(residual[basin])
    target_coverage = min(
        0.98,
        minimum_basin_coverage + entropy_coverage_gain * basin_entropy,
    )
    box: GridBox = (peak_x, peak_y, peak_x + 1, peak_y + 1)
    peak_density = max(float(residual[peak_y, peak_x]), 1e-12)
    max_cells = max(1, int(math.ceil(maximum_area_ratio * height * width)))

    while True:
        current_mass = _rect_sum(integral, box)
        basin_mass_in_box = float(
            residual[
                box[1] : box[3],
                box[0] : box[2],
            ][
                basin[
                    box[1] : box[3],
                    box[0] : box[2],
                ]
            ].sum()
        )
        coverage = basin_mass_in_box / max(basin_mass, 1e-12)
        candidates = []
        for candidate in _moves(box, width=width, height=height):
            area = (candidate[2] - candidate[0]) * (candidate[3] - candidate[1])
            if area > max_cells:
                continue
            added_cells = area - (box[2] - box[0]) * (box[3] - box[1])
            added_mass = max(0.0, _rect_sum(integral, candidate) - current_mass)
            candidates.append((added_mass / max(1, added_cells), added_mass, candidate))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_density, _, best_box = candidates[0]
        minimum_shape_reached = (box[2] - box[0] >= 2) and (box[3] - box[1] >= 2)
        boundary_is_weak = best_density <= boundary_density_ratio * peak_density
        if minimum_shape_reached and coverage >= target_coverage and boundary_is_weak:
            break
        box = best_box

    core_box = box
    context_factor = context_factor_min + context_factor_entropy_gain * basin_entropy
    final_box = _expand_grid_box(
        core_box,
        context_factor,
        width=width,
        height=height,
    )
    final_mass = _rect_sum(integral, final_box)
    area_cells = (final_box[2] - final_box[0]) * (final_box[3] - final_box[1])
    area_ratio = area_cells / (height * width)
    leakage = _ring_mass(integral, final_box, width=width, height=height) / max(
        final_mass, 1e-12
    )
    basin_mass_in_final = float(
        residual[
            final_box[1] : final_box[3],
            final_box[0] : final_box[2],
        ][
            basin[
                final_box[1] : final_box[3],
                final_box[0] : final_box[2],
            ]
        ].sum()
    )
    return {
        "core_grid_box": core_box,
        "grid_box": final_box,
        "basin_mass": basin_mass,
        "basin_entropy": basin_entropy,
        "basin_coverage": basin_mass_in_final / max(basin_mass, 1e-12),
        "mass": final_mass,
        "area_ratio": area_ratio,
        "mean_density": final_mass / max(area_ratio, 1e-12),
        "boundary_leakage": leakage,
        "context_factor": context_factor,
    }


def _candidate_from_box(
    distribution: np.ndarray,
    bbox: Sequence[float],
    *,
    role: str,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    box = normalize_policy_box(bbox)
    area_ratio = max(1, box[2] - box[0]) * max(1, box[3] - box[1]) / 1_000_000.0
    mass = map_mass(distribution, box)
    candidate: dict[str, Any] = {
        "bbox": box,
        "mass": mass,
        "area_ratio": area_ratio,
        "scale": math.sqrt(area_ratio),
        "mean_density": mass / max(area_ratio, 1e-12),
        "portfolio_role": role,
    }
    if diagnostics:
        candidate.update(diagnostics)
    leakage = float(candidate.get("boundary_leakage", 0.0))
    candidate["score"] = (
        mass**0.75 / max(area_ratio, 1e-8) ** 0.15 - 0.05 * leakage
    )
    return candidate


def decode_adaptive_evidence(
    probability: np.ndarray,
    *,
    max_regions: int = 3,
    include_topology_frame: bool = False,
    basin_relative_floor: float = 0.18,
    minimum_basin_coverage: float = 0.60,
    entropy_coverage_gain: float = 0.15,
    boundary_density_ratio: float = 0.20,
    context_factor_min: float = 1.05,
    context_factor_entropy_gain: float = 0.20,
    core_suppression: float = 0.05,
    maximum_area_ratio: float = 0.40,
) -> list[dict[str, Any]]:
    """Decode up to ``max_regions`` variable-size evidence regions.

    ``include_topology_frame`` reserves the final slot for the shared frame of
    the first two complementary modes.  The topology frame is derived from the
    modes and their entropy; it is not selected from a fixed window scale.
    """

    if max_regions <= 0:
        raise ValueError("max_regions must be positive")
    if include_topology_frame and max_regions < 3:
        raise ValueError("topology frame requires at least three regions")
    if not 0 < basin_relative_floor < 1:
        raise ValueError("basin_relative_floor must be in (0, 1)")
    if not 0 < minimum_basin_coverage <= 1:
        raise ValueError("minimum_basin_coverage must be in (0, 1]")
    if not 0 <= core_suppression < 1:
        raise ValueError("core_suppression must be in [0, 1)")
    if not 0 < maximum_area_ratio <= 1:
        raise ValueError("maximum_area_ratio must be in (0, 1]")

    distribution = normalize_distribution(probability)
    if distribution.ndim != 2 or not float(distribution.sum()) > 0:
        return []
    height, width = distribution.shape
    residual = distribution.copy()
    focus_count = max_regions - 1 if include_topology_frame else max_regions
    foci: list[dict[str, Any]] = []
    attempts = 0
    while len(foci) < focus_count and attempts < focus_count * 4:
        attempts += 1
        peak_y, peak_x = np.unravel_index(int(np.argmax(residual)), residual.shape)
        if float(residual[peak_y, peak_x]) <= 0:
            break
        grown = _grow_mode(
            residual,
            peak_y,
            peak_x,
            basin_relative_floor=basin_relative_floor,
            minimum_basin_coverage=minimum_basin_coverage,
            entropy_coverage_gain=entropy_coverage_gain,
            boundary_density_ratio=boundary_density_ratio,
            context_factor_min=context_factor_min,
            context_factor_entropy_gain=context_factor_entropy_gain,
            maximum_area_ratio=maximum_area_ratio,
        )
        policy_box = _grid_to_policy(
            grown["grid_box"],
            width=width,
            height=height,
        )
        core_policy_box = _grid_to_policy(
            grown["core_grid_box"],
            width=width,
            height=height,
        )
        if any(policy_iou(policy_box, item["bbox"]) >= 0.90 for item in foci):
            core = grown["core_grid_box"]
            residual[core[1] : core[3], core[0] : core[2]] *= core_suppression
            continue
        diagnostics = {
            key: float(value)
            for key, value in grown.items()
            if key not in {"grid_box", "core_grid_box", "mass", "area_ratio"}
        }
        diagnostics["core_bbox"] = core_policy_box
        focus = _candidate_from_box(
            distribution,
            policy_box,
            role=f"mode_{len(foci) + 1}_adaptive_focus",
            diagnostics=diagnostics,
        )
        foci.append(focus)
        core = grown["core_grid_box"]
        residual[core[1] : core[3], core[0] : core[2]] *= core_suppression

    if include_topology_frame and len(foci) >= 2:
        first, second = foci[:2]
        union = [
            min(first["bbox"][0], second["bbox"][0]),
            min(first["bbox"][1], second["bbox"][1]),
            max(first["bbox"][2], second["bbox"][2]),
            max(first["bbox"][3], second["bbox"][3]),
        ]
        mean_entropy = 0.5 * (
            float(first.get("basin_entropy", 0.0))
            + float(second.get("basin_entropy", 0.0))
        )
        union = expand_policy_box(union, 1.0 + 0.20 * mean_entropy)
        topology = _candidate_from_box(
            distribution,
            union,
            role="adaptive_topology_frame",
            diagnostics={
                "source_roles": [
                    first["portfolio_role"],
                    second["portfolio_role"],
                ],
                "basin_entropy": mean_entropy,
            },
        )
        return [first, second, topology]
    return foci[:max_regions]


def _pareto_frontier(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return candidates not dominated in coverage, density, context and cost."""

    frontier: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        dominated = False
        for other_index, other in enumerate(candidates):
            if index == other_index:
                continue
            no_worse = (
                float(other["basin_coverage"]) >= float(candidate["basin_coverage"])
                and float(other["mass"]) >= float(candidate["mass"])
                and float(other["mean_density"]) >= float(candidate["mean_density"])
                and float(other["context_value"]) >= float(candidate["context_value"])
                and float(other["area_ratio"]) <= float(candidate["area_ratio"])
                and float(other["boundary_leakage"])
                <= float(candidate["boundary_leakage"])
            )
            strictly_better = (
                float(other["basin_coverage"]) > float(candidate["basin_coverage"])
                or float(other["mass"]) > float(candidate["mass"])
                or float(other["mean_density"]) > float(candidate["mean_density"])
                or float(other["context_value"]) > float(candidate["context_value"])
                or float(other["area_ratio"]) < float(candidate["area_ratio"])
                or float(other["boundary_leakage"])
                < float(candidate["boundary_leakage"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def _normalise_objective(
    candidates: Sequence[dict[str, Any]], key: str, *, reverse: bool = False
) -> list[float]:
    values = np.asarray([float(item[key]) for item in candidates], dtype=np.float64)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum <= 1e-12:
        result = np.ones(len(candidates), dtype=np.float64)
    else:
        result = (values - minimum) / (maximum - minimum)
    if reverse:
        result = 1.0 - result
    return [float(value) for value in result]


def decode_pareto_evidence(
    probability: np.ndarray,
    *,
    max_regions: int = 3,
    include_topology_frame: bool = False,
    core_suppression: float = 0.05,
    maximum_area_ratio: float = 0.50,
) -> list[dict[str, Any]]:
    """Decode variable-size evidence regions through a Pareto candidate set.

    Each residual evidence mode is grown under several basin/coverage settings.
    The decoder first removes dominated rectangles and only then chooses a
    coverage-density-context trade-off.  Consequently neither the window scale
    nor its aspect ratio is selected from the legacy 0.20/0.25 templates.
    """

    if max_regions <= 0:
        raise ValueError("max_regions must be positive")
    if include_topology_frame and max_regions < 3:
        raise ValueError("topology frame requires at least three regions")
    if not 0 <= core_suppression < 1:
        raise ValueError("core_suppression must lie in [0, 1)")
    if not 0 < maximum_area_ratio <= 1:
        raise ValueError("maximum_area_ratio must lie in (0, 1]")

    distribution = normalize_distribution(probability)
    if distribution.ndim != 2 or float(distribution.sum()) <= 0:
        return []
    height, width = distribution.shape
    residual = distribution.copy()
    focus_count = max_regions - 1 if include_topology_frame else max_regions
    selected: list[dict[str, Any]] = []
    attempts = 0

    while len(selected) < focus_count and attempts < focus_count * 4:
        attempts += 1
        peak_y, peak_x = np.unravel_index(int(np.argmax(residual)), residual.shape)
        if float(residual[peak_y, peak_x]) <= 0:
            break
        proposals: list[dict[str, Any]] = []
        for relative_floor in (0.12, 0.18, 0.28):
            for minimum_coverage in (0.55, 0.70, 0.85):
                grown = _grow_mode(
                    residual,
                    peak_y,
                    peak_x,
                    basin_relative_floor=relative_floor,
                    minimum_basin_coverage=minimum_coverage,
                    entropy_coverage_gain=0.12,
                    boundary_density_ratio=0.20,
                    context_factor_min=1.03,
                    context_factor_entropy_gain=0.28,
                    maximum_area_ratio=maximum_area_ratio,
                )
                policy_box = _grid_to_policy(
                    grown["grid_box"], width=width, height=height
                )
                core_box = _grid_to_policy(
                    grown["core_grid_box"], width=width, height=height
                )
                expanded = expand_policy_box(policy_box, 1.15)
                expanded_mass = map_mass(distribution, expanded)
                expanded_area = (
                    max(1, expanded[2] - expanded[0])
                    * max(1, expanded[3] - expanded[1])
                    / 1_000_000.0
                )
                added_area = max(expanded_area - float(grown["area_ratio"]), 1e-8)
                context_value = max(
                    0.0, expanded_mass - float(grown["mass"])
                ) / added_area
                diagnostics = {
                    key: float(value)
                    for key, value in grown.items()
                    if key not in {"grid_box", "core_grid_box", "mass", "area_ratio"}
                }
                diagnostics.update(
                    {
                        "core_bbox": core_box,
                        "basin_relative_floor": relative_floor,
                        "minimum_basin_coverage": minimum_coverage,
                        "context_value": context_value,
                    }
                )
                proposals.append(
                    _candidate_from_box(
                        distribution,
                        policy_box,
                        role="pareto_proposal",
                        diagnostics=diagnostics,
                    )
                )

        frontier = _pareto_frontier(proposals)
        coverage = _normalise_objective(frontier, "basin_coverage")
        mass = _normalise_objective(frontier, "mass")
        density = _normalise_objective(frontier, "mean_density")
        context = _normalise_objective(frontier, "context_value")
        compact = _normalise_objective(frontier, "area_ratio", reverse=True)
        boundary = _normalise_objective(
            frontier, "boundary_leakage", reverse=True
        )
        mode_entropy = max(
            float(item.get("basin_entropy", 0.0)) for item in frontier
        )
        # Diffuse/multi-context modes value coverage and context; sharp modes
        # value density and compactness.  This interpolation is continuous.
        weights = {
            "coverage": 0.25 + 0.10 * mode_entropy,
            "mass": 0.25,
            "density": 0.20 - 0.08 * mode_entropy,
            "context": 0.08 + 0.08 * mode_entropy,
            "compact": 0.15 - 0.05 * mode_entropy,
            "boundary": 0.07,
        }
        utilities = []
        for index in range(len(frontier)):
            utilities.append(
                weights["coverage"] * coverage[index]
                + weights["mass"] * mass[index]
                + weights["density"] * density[index]
                + weights["context"] * context[index]
                + weights["compact"] * compact[index]
                + weights["boundary"] * boundary[index]
            )
        chosen = dict(frontier[int(np.argmax(utilities))])
        chosen["portfolio_role"] = f"mode_{len(selected) + 1}_pareto_focus"
        chosen["pareto_frontier_size"] = len(frontier)
        chosen["pareto_utility"] = float(max(utilities))
        if any(policy_iou(chosen["bbox"], item["bbox"]) >= 0.90 for item in selected):
            core = _policy_to_grid(chosen["core_bbox"], width=width, height=height)
            residual[core[1] : core[3], core[0] : core[2]] *= core_suppression
            continue
        selected.append(chosen)
        core = _policy_to_grid(chosen["core_bbox"], width=width, height=height)
        residual[core[1] : core[3], core[0] : core[2]] *= core_suppression

    if include_topology_frame and len(selected) >= 2:
        first, second = selected[:2]
        union = [
            min(first["bbox"][0], second["bbox"][0]),
            min(first["bbox"][1], second["bbox"][1]),
            max(first["bbox"][2], second["bbox"][2]),
            max(first["bbox"][3], second["bbox"][3]),
        ]
        topology = _candidate_from_box(
            distribution,
            expand_policy_box(union, 1.08),
            role="pareto_topology_frame",
            diagnostics={
                "context_value": 0.0,
                "source_roles": [
                    first["portfolio_role"],
                    second["portfolio_role"],
                ],
            },
        )
        return [first, second, topology]
    return selected[:max_regions]
