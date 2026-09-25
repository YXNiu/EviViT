"""Seeded residual basin decoding for Stage-D EviViT.

The decoder turns one question-conditioned PTEA probability map into one to
three complementary, variable-size evidence regions.  It deliberately differs
from the earlier adaptive-contour experiments in three ways:

* seeds are maxima of a local 3x3 *density* map, not isolated peak pixels;
* a nested multi-threshold component tree chooses a stable basin;
* the complete selected basin/box is excluded before the next seed is found.

No human box is read at inference time.  Trace labels are used only by the
offline calibration script to freeze the small set of deterministic weights.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from evivit_core.dense_evidence import map_mass, normalize_distribution, policy_iou


@dataclass(frozen=True)
class BasinDecoderConfig:
    seed_kernel: int = 3
    threshold_ratios: tuple[float, ...] = (
        0.90,
        0.82,
        0.74,
        0.66,
        0.58,
        0.50,
        0.42,
        0.34,
        0.26,
        0.18,
        0.12,
    )
    context_margin: float = 0.08
    minimum_residual_mass: float = 0.05
    minimum_seed_ratio: float = 0.18
    maximum_region_area: float = 0.40
    maximum_pairwise_iou: float = 0.55
    opening_minimum_cells: int = 9
    opening_minimum_mass_retention: float = 0.60
    mass_weight: float = 0.34
    density_weight: float = 0.16
    stability_weight: float = 0.28
    boundary_weight: float = 0.12
    compactness_weight: float = 0.10
    aspect_penalty: float = 0.08

    def validate(self) -> None:
        if self.seed_kernel < 1 or self.seed_kernel % 2 != 1:
            raise ValueError("seed_kernel must be a positive odd integer")
        if not self.threshold_ratios or any(
            not 0 < value <= 1 for value in self.threshold_ratios
        ):
            raise ValueError("threshold_ratios must lie in (0, 1]")
        if not 0 <= self.context_margin <= 1:
            raise ValueError("context_margin must lie in [0, 1]")
        if not 0 <= self.minimum_residual_mass < 1:
            raise ValueError("minimum_residual_mass must lie in [0, 1)")
        if not 0 <= self.minimum_seed_ratio <= 1:
            raise ValueError("minimum_seed_ratio must lie in [0, 1]")
        if not 0 < self.maximum_region_area <= 1:
            raise ValueError("maximum_region_area must lie in (0, 1]")
        if not 0 <= self.maximum_pairwise_iou <= 1:
            raise ValueError("maximum_pairwise_iou must lie in [0, 1]")
        if self.opening_minimum_cells < 1:
            raise ValueError("opening_minimum_cells must be positive")
        if not 0 <= self.opening_minimum_mass_retention <= 1:
            raise ValueError("opening_minimum_mass_retention must lie in [0, 1]")


def _box_filter(array: np.ndarray, kernel: int) -> np.ndarray:
    radius = kernel // 2
    padded = np.pad(array, ((radius, radius), (radius, radius)), mode="constant")
    valid = np.pad(
        np.ones_like(array, dtype=np.float64),
        ((radius, radius), (radius, radius)),
        mode="constant",
    )
    height, width = array.shape
    total = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.float64)
    for dy in range(kernel):
        for dx in range(kernel):
            total += padded[dy : dy + height, dx : dx + width]
            counts += valid[dy : dy + height, dx : dx + width]
    return total / np.maximum(counts, 1.0)


def _seed_mask(shape: tuple[int, int], y: int, x: int, kernel: int) -> np.ndarray:
    radius = kernel // 2
    result = np.zeros(shape, dtype=bool)
    result[
        max(0, y - radius) : min(shape[0], y + radius + 1),
        max(0, x - radius) : min(shape[1], x + radius + 1),
    ] = True
    return result


def _dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant")
        expanded = np.zeros_like(result)
        for dy in range(3):
            for dx in range(3):
                expanded |= padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
        result = expanded
    return result


def _erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        contracted = np.ones_like(result)
        for dy in range(3):
            for dx in range(3):
                contracted &= padded[
                    dy : dy + result.shape[0], dx : dx + result.shape[1]
                ]
        result = contracted
    return result


def _component_from_seed(active: np.ndarray, seed: np.ndarray) -> np.ndarray:
    starts = np.argwhere(active & seed)
    component = np.zeros_like(active, dtype=bool)
    if not len(starts):
        return component
    queue: deque[tuple[int, int]] = deque()
    for y, x in starts:
        component[int(y), int(x)] = True
        queue.append((int(y), int(x)))
    height, width = active.shape
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
                    and not component[ny, nx]
                ):
                    component[ny, nx] = True
                    queue.append((ny, nx))
    return component


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / max(int(union), 1))


def _grid_box(mask: np.ndarray, *, margin: float) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cannot box an empty component")
    height, width = mask.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    margin_x = max(1, int(round((x1 - x0) * margin))) if margin else 0
    margin_y = max(1, int(round((y1 - y0) * margin))) if margin else 0
    return (
        max(0, x0 - margin_x),
        max(0, y0 - margin_y),
        min(width, x1 + margin_x),
        min(height, y1 + margin_y),
    )


def _policy_box(
    box: tuple[int, int, int, int], *, width: int, height: int
) -> list[int]:
    x0, y0, x1, y1 = box
    return [
        int(round(x0 / width * 1000)),
        int(round(y0 / height * 1000)),
        int(round(x1 / width * 1000)),
        int(round(y1 / height * 1000)),
    ]


def _normalise(values: Iterable[float], *, reverse: bool = False) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 1 or float(array.max() - array.min()) <= 1e-12:
        result = np.ones_like(array)
    else:
        result = (array - array.min()) / (array.max() - array.min())
    return 1.0 - result if reverse else result


def _boundary_contrast(values: np.ndarray, component: np.ndarray) -> float:
    ring = _dilate(component) & ~component
    inside = float(values[component].mean()) if component.any() else 0.0
    outside = float(values[ring].mean()) if ring.any() else 0.0
    return inside / max(outside, 1e-12)


def _open_if_safe(
    component: np.ndarray,
    values: np.ndarray,
    *,
    minimum_cells: int,
    minimum_mass_retention: float,
) -> np.ndarray:
    if int(component.sum()) < minimum_cells:
        return component
    opened = _dilate(_erode(component))
    if not opened.any():
        return component
    original_mass = float(values[component].sum())
    retained_mass = float(values[opened].sum())
    if retained_mass / max(original_mass, 1e-12) < minimum_mass_retention:
        return component
    return opened


def _nested_basin_candidates(
    residual: np.ndarray,
    seed_y: int,
    seed_x: int,
    *,
    config: BasinDecoderConfig,
) -> list[dict[str, Any]]:
    seed = _seed_mask(residual.shape, seed_y, seed_x, config.seed_kernel)
    seed_peak = float(residual[seed].max())
    candidates: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    max_cells = max(1, int(np.ceil(config.maximum_region_area * residual.size)))
    for ratio in sorted(set(config.threshold_ratios), reverse=True):
        active = residual >= ratio * seed_peak
        component = _component_from_seed(active, seed)
        if not component.any() or int(component.sum()) > max_cells:
            continue
        component = _open_if_safe(
            component,
            residual,
            minimum_cells=config.opening_minimum_cells,
            minimum_mass_retention=config.opening_minimum_mass_retention,
        )
        key = np.packbits(component).tobytes()
        if key in seen:
            continue
        seen.add(key)
        box = _grid_box(component, margin=config.context_margin)
        width = box[2] - box[0]
        height = box[3] - box[1]
        area = width * height / residual.size
        component_mass = float(residual[component].sum())
        density = component_mass / max(int(component.sum()), 1)
        aspect = max(width / max(height, 1), height / max(width, 1))
        candidates.append(
            {
                "threshold_ratio": float(ratio),
                "component": component,
                "grid_box": box,
                "component_mass": component_mass,
                "component_density": density,
                "boundary_contrast": _boundary_contrast(residual, component),
                "area_ratio": float(area),
                "aspect_ratio": float(aspect),
                "stability": 0.0,
            }
        )
    for index, candidate in enumerate(candidates):
        neighbors = []
        if index:
            neighbors.append(_mask_iou(candidate["component"], candidates[index - 1]["component"]))
        if index + 1 < len(candidates):
            neighbors.append(_mask_iou(candidate["component"], candidates[index + 1]["component"]))
        candidate["stability"] = float(np.mean(neighbors)) if neighbors else 1.0
    return candidates


def _select_basin(
    candidates: Sequence[dict[str, Any]], *, config: BasinDecoderConfig
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("cannot select from an empty basin candidate set")
    mass = _normalise(item["component_mass"] for item in candidates)
    density = _normalise(item["component_density"] for item in candidates)
    stability = np.asarray([item["stability"] for item in candidates])
    boundary = _normalise(
        np.log1p(item["boundary_contrast"]) for item in candidates
    )
    compact = _normalise((item["area_ratio"] for item in candidates), reverse=True)
    aspect = np.asarray([np.log(max(item["aspect_ratio"], 1.0)) for item in candidates])
    scores = (
        config.mass_weight * mass
        + config.density_weight * density
        + config.stability_weight * stability
        + config.boundary_weight * boundary
        + config.compactness_weight * compact
        - config.aspect_penalty * aspect
    )
    chosen_index = int(np.argmax(scores))
    chosen = dict(candidates[chosen_index])
    chosen["selection_score"] = float(scores[chosen_index])
    chosen["candidate_count"] = len(candidates)
    return chosen


def decode_seeded_residual_basins(
    probability: np.ndarray,
    *,
    max_regions: int = 3,
    config: BasinDecoderConfig | None = None,
    include_component_cells: bool = False,
) -> list[dict[str, Any]]:
    """Decode one to ``max_regions`` complementary adaptive evidence boxes."""

    if max_regions <= 0:
        raise ValueError("max_regions must be positive")
    config = config or BasinDecoderConfig()
    config.validate()
    distribution = normalize_distribution(np.asarray(probability, dtype=np.float64))
    if distribution.ndim != 2 or not float(distribution.sum()) > 0:
        return []
    residual = distribution.copy()
    forbidden = np.zeros_like(residual, dtype=bool)
    selected: list[dict[str, Any]] = []
    first_seed_density: float | None = None
    initial_mass = float(residual.sum())
    height, width = residual.shape

    for region_index in range(1, max_regions + 1):
        if selected and float(residual.sum()) < config.minimum_residual_mass * initial_mass:
            break
        density_map = _box_filter(residual, config.seed_kernel)
        # The complete seed window must remain outside previously selected boxes.
        invalid_centers = _box_filter(forbidden.astype(np.float64), config.seed_kernel) > 0
        density_map[invalid_centers] = -np.inf
        flat_index = int(np.argmax(density_map))
        seed_y, seed_x = np.unravel_index(flat_index, density_map.shape)
        seed_density = float(density_map[seed_y, seed_x])
        if not np.isfinite(seed_density) or seed_density <= 0:
            break
        if first_seed_density is None:
            first_seed_density = seed_density
        elif seed_density < config.minimum_seed_ratio * first_seed_density:
            break

        basin_candidates = _nested_basin_candidates(
            residual,
            seed_y,
            seed_x,
            config=config,
        )
        if not basin_candidates:
            break
        basin = _select_basin(basin_candidates, config=config)
        bbox = _policy_box(basin["grid_box"], width=width, height=height)
        if any(
            policy_iou(bbox, item["bbox"]) > config.maximum_pairwise_iou
            for item in selected
        ):
            # Reject the whole candidate rather than allowing another centre in
            # the same evidence mode. Its box becomes forbidden before retrying.
            x0, y0, x1, y1 = basin["grid_box"]
            forbidden[y0:y1, x0:x1] = True
            residual[y0:y1, x0:x1] = 0.0
            continue

        component = basin.pop("component")
        x0, y0, x1, y1 = basin.pop("grid_box")
        area_ratio = max(1, x1 - x0) * max(1, y1 - y0) / residual.size
        output: dict[str, Any] = {
            "bbox": bbox,
            "score": float(basin.pop("selection_score")),
            "mass": map_mass(distribution, bbox),
            "area_ratio": float(area_ratio),
            "scale": float(np.sqrt(area_ratio)),
            "portfolio_role": f"seeded_basin_{region_index}",
            "seed_grid": [int(seed_x), int(seed_y)],
            "seed_density": seed_density,
            "component_cells": int(component.sum()),
            **basin,
        }
        if include_component_cells:
            output["component_grid_cells"] = [
                [int(y), int(x)] for y, x in np.argwhere(component)
            ]
        selected.append(output)

        # Suppress the complete output box plus one grid-cell halo. This makes
        # the next 3x3 seed genuinely complementary instead of a nearby pixel
        # from the same diffuse mode.
        exclusion = np.zeros_like(forbidden)
        exclusion[y0:y1, x0:x1] = True
        exclusion = _dilate(exclusion, iterations=1)
        forbidden |= exclusion
        residual[exclusion] = 0.0

    return selected


__all__ = ["BasinDecoderConfig", "decode_seeded_residual_basins"]
