"""Training-free adaptive contour decoding for Simple EviViT S2.

Unlike EviViT-v4's fixed-scale sliding windows, this decoder grows each region
from the connected evidence support around a residual-map peak.  It operates
only on the small PTEA probability grid and introduces no trainable parameter.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from evivit_core.dense_evidence import map_mass, normalize_distribution


def _peak_component(mask: np.ndarray, peak_y: int, peak_x: int) -> np.ndarray:
    """Return the 8-connected binary component containing one peak."""

    if mask.ndim != 2 or not bool(mask[peak_y, peak_x]):
        raise ValueError("the peak must lie inside a two-dimensional mask")
    height, width = mask.shape
    component = np.zeros_like(mask, dtype=bool)
    component[peak_y, peak_x] = True
    queue: deque[tuple[int, int]] = deque([(peak_y, peak_x)])
    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                next_y, next_x = y + dy, x + dx
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and bool(mask[next_y, next_x])
                    and not bool(component[next_y, next_x])
                ):
                    component[next_y, next_x] = True
                    queue.append((next_y, next_x))
    return component


def _expanded_grid_box(
    component: np.ndarray,
    *,
    context_margin: float,
) -> tuple[int, int, int, int]:
    """Return an expanded half-open grid box ``x0, y0, x1, y1``."""

    ys, xs = np.nonzero(component)
    if not len(xs):
        raise ValueError("cannot box an empty component")
    height, width = component.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    margin_x = max(1, int(round((x1 - x0) * context_margin)))
    margin_y = max(1, int(round((y1 - y0) * context_margin)))
    return (
        max(0, x0 - margin_x),
        max(0, y0 - margin_y),
        min(width, x1 + margin_x),
        min(height, y1 + margin_y),
    )


def _policy_box(
    grid_box: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
) -> list[int]:
    x0, y0, x1, y1 = grid_box
    return [
        int(round(x0 / width * 1000)),
        int(round(y0 / height * 1000)),
        int(round(x1 / width * 1000)),
        int(round(y1 / height * 1000)),
    ]


def decode_adaptive_contours(
    heatmap: np.ndarray,
    *,
    max_regions: int = 3,
    relative_threshold: float = 0.30,
    context_margin: float = 0.10,
    minimum_residual_mass: float = 0.05,
) -> list[dict[str, Any]]:
    """Decode complementary variable-shape evidence regions.

    At each round the highest residual peak defines a relative threshold.
    Only the connected supra-threshold component containing that peak is
    boxed, expanded by a small context margin, and suppressed.  Selection
    stops when little probability mass remains, so the decoder need not emit
    all ``max_regions``.
    """

    if max_regions <= 0:
        raise ValueError("max_regions must be positive")
    if not 0.0 < relative_threshold <= 1.0:
        raise ValueError("relative_threshold must lie in (0, 1]")
    if not 0.0 <= context_margin <= 1.0:
        raise ValueError("context_margin must lie in [0, 1]")
    if not 0.0 <= minimum_residual_mass < 1.0:
        raise ValueError("minimum_residual_mass must lie in [0, 1)")

    distribution = normalize_distribution(np.asarray(heatmap, dtype=np.float64))
    if distribution.ndim != 2 or float(distribution.sum()) <= 0:
        return []
    residual = distribution.copy()
    height, width = distribution.shape
    candidates: list[dict[str, Any]] = []
    for region_index in range(1, max_regions + 1):
        residual_mass = float(residual.sum())
        if candidates and residual_mass < minimum_residual_mass:
            break
        flat_peak = int(np.argmax(residual))
        peak_y, peak_x = np.unravel_index(flat_peak, residual.shape)
        peak_value = float(residual[peak_y, peak_x])
        if peak_value <= 0:
            break
        support = residual >= relative_threshold * peak_value
        component = _peak_component(support, peak_y, peak_x)
        grid_box = _expanded_grid_box(
            component,
            context_margin=context_margin,
        )
        bbox = _policy_box(grid_box, width=width, height=height)
        mass = map_mass(distribution, bbox)
        x0, y0, x1, y1 = grid_box
        area_ratio = (x1 - x0) * (y1 - y0) / (height * width)
        candidates.append(
            {
                "bbox": bbox,
                "score": mass / max(area_ratio, 1e-8) ** 0.15,
                "mass": mass,
                "area_ratio": area_ratio,
                "portfolio_role": f"contour_mode_{region_index}",
                "peak_probability": peak_value,
                "component_mass": float(distribution[component].sum()),
                "residual_mass_before": residual_mass,
                "relative_threshold": relative_threshold,
                "context_margin": context_margin,
            }
        )
        residual[y0:y1, x0:x1] = 0.0
    return candidates


__all__ = ["decode_adaptive_contours"]
