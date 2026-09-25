"""v4-anchored adaptive evidence growth for Stage-E EviViT.

The decoder deliberately changes only the *boundary* of the frozen v4
Residual-Focus regions.  v4 still determines the ordered R1/R2/R3 anchors.
Within every anchor, the densest 3x3 location seeds a connected growth path.
Growth stops at a persistent marginal-density cliff and otherwise falls back
to the exact v4 anchor.  No human annotation is read at inference time.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from scripts.build_evimap_portfolio_boxes import decode_portfolio
from evivit_core.dense_evidence import map_mass, normalize_distribution


@dataclass(frozen=True)
class AnchoredGrowthConfig:
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
    marginal_density_ratio: float = 0.25
    low_density_patience: int = 2
    context_margin: float = 0.05
    maximum_region_area: float = 0.40

    def validate(self) -> None:
        if self.seed_kernel < 1 or self.seed_kernel % 2 != 1:
            raise ValueError("seed_kernel must be a positive odd integer")
        if not self.threshold_ratios or any(
            not 0 < value <= 1 for value in self.threshold_ratios
        ):
            raise ValueError("threshold_ratios must lie in (0, 1]")
        if not 0 < self.marginal_density_ratio < 1:
            raise ValueError("marginal_density_ratio must lie in (0, 1)")
        if self.low_density_patience < 1:
            raise ValueError("low_density_patience must be positive")
        if not 0 <= self.context_margin <= 1:
            raise ValueError("context_margin must lie in [0, 1]")
        if not 0 < self.maximum_region_area <= 1:
            raise ValueError("maximum_region_area must lie in (0, 1]")


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


def _policy_to_grid_mask(
    bbox: Sequence[float], shape: tuple[int, int]
) -> np.ndarray:
    height, width = shape
    x0 = max(0, min(width - 1, int(np.floor(float(bbox[0]) / 1000 * width))))
    y0 = max(0, min(height - 1, int(np.floor(float(bbox[1]) / 1000 * height))))
    x1 = max(x0 + 1, min(width, int(np.ceil(float(bbox[2]) / 1000 * width))))
    y1 = max(y0 + 1, min(height, int(np.ceil(float(bbox[3]) / 1000 * height))))
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _seed_core(shape: tuple[int, int], y: int, x: int, kernel: int) -> np.ndarray:
    radius = kernel // 2
    mask = np.zeros(shape, dtype=bool)
    mask[
        max(0, y - radius) : min(shape[0], y + radius + 1),
        max(0, x - radius) : min(shape[1], x + radius + 1),
    ] = True
    return mask


def _component_from_seed(active: np.ndarray, seed: np.ndarray) -> np.ndarray:
    starts = np.argwhere(active & seed)
    component = np.zeros_like(active, dtype=bool)
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


def _grid_box(mask: np.ndarray, margin: float) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cannot box an empty component")
    height, width = mask.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    mx = max(1, int(round((x1 - x0) * margin))) if margin else 0
    my = max(1, int(round((y1 - y0) * margin))) if margin else 0
    return (
        max(0, x0 - mx),
        max(0, y0 - my),
        min(width, x1 + mx),
        min(height, y1 + my),
    )


def _policy_box(
    box: tuple[int, int, int, int], shape: tuple[int, int]
) -> list[int]:
    height, width = shape
    x0, y0, x1, y1 = box
    return [
        int(round(x0 / width * 1000)),
        int(round(y0 / height * 1000)),
        int(round(x1 / width * 1000)),
        int(round(y1 / height * 1000)),
    ]


def _find_anchor_seed(
    values: np.ndarray,
    anchor_bbox: Sequence[float],
    *,
    kernel: int,
) -> tuple[int, int]:
    density = _box_filter(values, kernel)
    density[~_policy_to_grid_mask(anchor_bbox, values.shape)] = -np.inf
    flat = int(np.argmax(density))
    y, x = np.unravel_index(flat, density.shape)
    if not np.isfinite(density[y, x]):
        raise RuntimeError("v4 anchor contains no valid density seed")
    return int(y), int(x)


def _grow_one_anchor(
    values: np.ndarray,
    anchor: dict[str, Any],
    seed: tuple[int, int],
    protected_other_seeds: np.ndarray,
    forbidden: np.ndarray,
    *,
    config: AnchoredGrowthConfig,
) -> dict[str, Any]:
    seed_y, seed_x = seed
    core = _seed_core(values.shape, seed_y, seed_x, config.seed_kernel)
    available = ~(forbidden | protected_other_seeds)
    available |= core
    seed_peak = float(values[core].max())
    max_cells = max(1, int(np.ceil(config.maximum_region_area * values.size)))
    candidates: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for threshold in sorted(set(config.threshold_ratios), reverse=True):
        active = ((values >= threshold * seed_peak) | core) & available
        component = _component_from_seed(active, core)
        cells = int(component.sum())
        if not cells or cells > max_cells:
            continue
        key = np.packbits(component).tobytes()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "threshold_ratio": float(threshold),
                "component": component,
                "cells": cells,
                "mass": float(values[component].sum()),
            }
        )

    anchor_box = list(anchor["bbox"])
    fallback = {
        "bbox": anchor_box,
        "score": float(anchor.get("score", map_mass(values, anchor_box))),
        "mass": map_mass(values, anchor_box),
        "portfolio_role": str(anchor.get("portfolio_role", "anchor_focus")),
        "source_bbox": anchor_box,
        "seed_grid": [seed_x, seed_y],
        "growth_applied": False,
        "growth_stop_reason": "identity_no_persistent_density_cliff",
    }
    if len(candidates) < 2:
        return fallback

    chosen: dict[str, Any] | None = None
    low_run = 0
    before_low: dict[str, Any] | None = None
    stop_ratio: float | None = None
    marginal_ratio: float | None = None
    previous = candidates[0]
    for candidate in candidates[1:]:
        added = candidate["component"] & ~previous["component"]
        if not added.any():
            previous = candidate
            continue
        frontier_density = float(values[added].mean())
        interior_density = float(values[previous["component"]].mean())
        ratio = frontier_density / max(interior_density, 1e-12)
        if ratio < config.marginal_density_ratio:
            if low_run == 0:
                before_low = previous
            low_run += 1
            if low_run >= config.low_density_patience:
                chosen = before_low
                stop_ratio = float(candidate["threshold_ratio"])
                marginal_ratio = float(ratio)
                break
        else:
            low_run = 0
            before_low = None
        previous = candidate

    if chosen is None:
        return fallback
    box = _grid_box(chosen["component"], config.context_margin)
    bbox = _policy_box(box, values.shape)
    return {
        "bbox": bbox,
        "score": float(anchor.get("score", map_mass(values, bbox))),
        "mass": map_mass(values, bbox),
        "area_ratio": (
            max(1, box[2] - box[0]) * max(1, box[3] - box[1]) / values.size
        ),
        "portfolio_role": str(anchor.get("portfolio_role", "anchor_focus")),
        "source_bbox": anchor_box,
        "seed_grid": [seed_x, seed_y],
        "growth_applied": True,
        "growth_stop_reason": "persistent_marginal_density_cliff",
        "growth_threshold_ratio": float(chosen["threshold_ratio"]),
        "growth_stop_threshold_ratio": stop_ratio,
        "growth_stop_marginal_ratio": marginal_ratio,
        "component_cells": int(chosen["cells"]),
    }


def decode_anchored_evidence_growth(
    probability: np.ndarray,
    *,
    max_regions: int = 3,
    config: AnchoredGrowthConfig | None = None,
) -> list[dict[str, Any]]:
    """Return v4-anchored, variable-size Stage-E evidence regions."""

    if max_regions <= 0:
        raise ValueError("max_regions must be positive")
    config = config or AnchoredGrowthConfig()
    config.validate()
    distribution = normalize_distribution(np.asarray(probability, dtype=np.float64))
    portfolio = decode_portfolio(
        distribution,
        top_k=max(2 * max_regions + 2, 8),
        modes=max_regions,
        context_factor=1.8,
        residual_suppression=0.05,
    )
    anchors = [
        dict(row)
        for row in portfolio
        if str(row.get("portfolio_role", "")).endswith("_focus")
    ][:max_regions]
    seeds = [
        _find_anchor_seed(
            distribution,
            anchor["bbox"],
            kernel=config.seed_kernel,
        )
        for anchor in anchors
    ]
    seed_cores = [
        _seed_core(distribution.shape, y, x, config.seed_kernel) for y, x in seeds
    ]
    forbidden = np.zeros_like(distribution, dtype=bool)
    outputs: list[dict[str, Any]] = []
    for index, (anchor, seed) in enumerate(zip(anchors, seeds), 1):
        protected = np.zeros_like(forbidden)
        for other_index, core in enumerate(seed_cores):
            if other_index != index - 1:
                protected |= core
        output = _grow_one_anchor(
            distribution,
            anchor,
            seed,
            protected,
            forbidden,
            config=config,
        )
        output["portfolio_role"] = f"anchored_growth_mode_{index}"
        outputs.append(output)
        forbidden |= _policy_to_grid_mask(output["bbox"], distribution.shape)
    return outputs


def _policy_bbox_area(bbox: Sequence[float]) -> float:
    x0, y0, x1, y1 = (float(value) / 1000.0 for value in bbox)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _policy_bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    a = [float(value) / 1000.0 for value in first]
    b = [float(value) / 1000.0 for value in second]
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    return overlap / max(_policy_bbox_area(first) + _policy_bbox_area(second) - overlap, 1e-12)


def anchored_growth_calibrator_features(
    probability: np.ndarray,
    *,
    max_regions: int = 3,
    probe_betas: Sequence[float] = (0.25, 0.35, 0.45),
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return inference-only per-anchor features and the fixed E1 portfolio.

    The PTEA map is already question conditioned.  Features therefore use only
    the map, frozen v4 anchors and three deterministic probes along the same
    anchored growth curve; no image annotation or answer is available here.
    """

    distribution = normalize_distribution(np.asarray(probability, dtype=np.float64))
    beta_values = tuple(float(value) for value in probe_betas)
    if not beta_values or any(not 0 < value < 1 for value in beta_values):
        raise ValueError("probe betas must lie in (0, 1)")
    probes = {
        beta: decode_anchored_evidence_growth(
            distribution,
            max_regions=max_regions,
            config=AnchoredGrowthConfig(marginal_density_ratio=beta),
        )
        for beta in beta_values
    }
    e1_rows = probes[min(beta_values, key=lambda value: abs(value - 0.35))]
    positive = distribution[distribution > 0]
    global_entropy = (
        float(-(positive * np.log(positive + 1e-12)).sum())
        / max(math.log(distribution.size), 1e-12)
        if positive.size
        else 0.0
    )
    feature_rows: list[list[float]] = []
    for index, e1 in enumerate(e1_rows):
        anchor = list(e1["source_bbox"])
        x0, y0, x1, y1 = (float(value) / 1000.0 for value in anchor)
        anchor_area = _policy_bbox_area(anchor)
        anchor_mass = map_mass(distribution, anchor)
        seed_x, seed_y = e1.get("seed_grid", [0, 0])
        base = [
            float(index + 1) / max(max_regions, 1),
            x0,
            y0,
            x1,
            y1,
            anchor_area,
            anchor_mass,
            anchor_mass / max(anchor_area, 1e-8),
            float(distribution.max()),
            global_entropy,
            float(seed_x) / max(distribution.shape[1] - 1, 1),
            float(seed_y) / max(distribution.shape[0] - 1, 1),
        ]
        for beta in beta_values:
            row = probes[beta][index]
            bbox = row["bbox"]
            area = _policy_bbox_area(bbox)
            mass = map_mass(distribution, bbox)
            base.extend(
                [
                    float(bool(row.get("growth_applied"))),
                    area,
                    mass,
                    mass / max(area, 1e-8),
                    float(row.get("growth_threshold_ratio") or 0.0),
                    float(row.get("growth_stop_marginal_ratio") or 0.0),
                    _policy_bbox_iou(anchor, bbox),
                ]
            )
        feature_rows.append(base)
    return np.asarray(feature_rows, dtype=np.float32), e1_rows


class AnchoredGrowthCalibratorHead(nn.Module):
    """Tiny E1-initialized predictor of stop beta and an enable gate."""

    def __init__(
        self,
        input_dim: int = 33,
        hidden_dim: int = 48,
        beta_center: float = 0.35,
        beta_radius: float = 0.15,
    ) -> None:
        super().__init__()
        if not 0 < beta_center - beta_radius < beta_center + beta_radius < 1:
            raise ValueError("beta interval must lie in (0, 1)")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.beta_center = float(beta_center)
        self.beta_radius = float(beta_radius)
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():
            # Identity at initialization: beta=0.35 and gate below threshold.
            self.network[-1].bias[1] = -2.0

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.network(features)
        beta = self.beta_center + self.beta_radius * torch.tanh(raw[:, 0])
        return beta, raw[:, 1]


def load_anchored_growth_calibrator(
    checkpoint_path: str,
    *,
    device: str | torch.device = "cpu",
) -> tuple[AnchoredGrowthCalibratorHead, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    head = AnchoredGrowthCalibratorHead(
        input_dim=int(config.get("input_dim", 33)),
        hidden_dim=int(config.get("hidden_dim", 48)),
        beta_center=float(config.get("beta_center", 0.35)),
        beta_radius=float(config.get("beta_radius", 0.15)),
    )
    head.load_state_dict(checkpoint["state_dict"])
    head.to(device).eval()
    return head, checkpoint


def decode_calibrated_anchored_evidence_growth(
    probability: np.ndarray,
    head: AnchoredGrowthCalibratorHead,
    *,
    max_regions: int = 3,
    gate_threshold: float = 0.65,
    minimum_area_ratio_to_e1: float = 0.67,
    maximum_area_ratio_to_e1: float = 1.50,
) -> list[dict[str, Any]]:
    """Apply a learned stop beta with E1 and Stage-A identity fallbacks."""

    if not 0 < gate_threshold < 1:
        raise ValueError("gate_threshold must lie in (0, 1)")
    if not 0 < minimum_area_ratio_to_e1 <= 1 <= maximum_area_ratio_to_e1:
        raise ValueError("invalid E1-relative area safety interval")
    distribution = normalize_distribution(np.asarray(probability, dtype=np.float64))
    features, fixed_e1 = anchored_growth_calibrator_features(
        distribution, max_regions=max_regions
    )
    device = next(head.parameters()).device
    with torch.inference_mode():
        betas, gate_logits = head(
            torch.tensor(features, dtype=torch.float32, device=device)
        )
        predicted_betas = betas.float().cpu().tolist()
        gate_probabilities = torch.sigmoid(gate_logits).float().cpu().tolist()

    anchors = [dict(row) for row in fixed_e1]
    seeds = [tuple(int(value) for value in row["seed_grid"][::-1]) for row in anchors]
    seed_cores = [
        _seed_core(distribution.shape, y, x, AnchoredGrowthConfig().seed_kernel)
        for y, x in seeds
    ]
    forbidden = np.zeros_like(distribution, dtype=bool)
    outputs: list[dict[str, Any]] = []
    for index, (anchor_row, seed, beta, gate) in enumerate(
        zip(anchors, seeds, predicted_betas, gate_probabilities), 1
    ):
        source_anchor = {
            "bbox": list(anchor_row["source_bbox"]),
            "score": float(anchor_row.get("score", 0.0)),
            "portfolio_role": f"anchored_growth_mode_{index}",
        }
        protected = np.zeros_like(forbidden)
        for other_index, core in enumerate(seed_cores):
            if other_index != index - 1:
                protected |= core
        e1 = _grow_one_anchor(
            distribution,
            source_anchor,
            seed,
            protected,
            forbidden,
            config=AnchoredGrowthConfig(marginal_density_ratio=0.35),
        )
        chosen = e1
        reason = "gate_fallback_e1"
        if gate >= gate_threshold:
            calibrated = _grow_one_anchor(
                distribution,
                source_anchor,
                seed,
                protected,
                forbidden,
                config=AnchoredGrowthConfig(marginal_density_ratio=float(beta)),
            )
            e1_area = _policy_bbox_area(e1["bbox"])
            calibrated_area = _policy_bbox_area(calibrated["bbox"])
            relative = calibrated_area / max(e1_area, 1e-12)
            if (
                bool(calibrated.get("growth_applied"))
                and minimum_area_ratio_to_e1 <= relative <= maximum_area_ratio_to_e1
            ):
                chosen = calibrated
                reason = "calibrated_beta"
            else:
                reason = "area_or_cliff_fallback_e1"
        chosen = dict(chosen)
        chosen["portfolio_role"] = f"anchored_calibrated_mode_{index}"
        chosen["calibrator_beta"] = float(beta)
        chosen["calibrator_gate"] = float(gate)
        chosen["calibrator_applied"] = reason == "calibrated_beta"
        chosen["calibrator_decision"] = reason
        outputs.append(chosen)
        forbidden |= _policy_to_grid_mask(chosen["bbox"], distribution.shape)
    return outputs


__all__ = [
    "AnchoredGrowthConfig",
    "AnchoredGrowthCalibratorHead",
    "anchored_growth_calibrator_features",
    "decode_calibrated_anchored_evidence_growth",
    "decode_anchored_evidence_growth",
    "load_anchored_growth_calibrator",
]
