"""Utilities for direct multi-scale human-trace supervision of fine tokens."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F


DEFAULT_TRACE_WEIGHTS = {"context": 0.15, "read": 0.45, "core": 0.40}


def combine_trace_maps(
    channels: Mapping[str, torch.Tensor],
    *,
    weights: Mapping[str, float] = DEFAULT_TRACE_WEIGHTS,
) -> torch.Tensor:
    """Combine context/read/core maps into one normalized evidence distribution."""

    missing = set(weights) - set(channels)
    if missing:
        raise ValueError(f"missing trace channels: {sorted(missing)}")
    shape = None
    combined = None
    for name, weight in weights.items():
        value = channels[name].float()
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [height, width]")
        if shape is None:
            shape = value.shape
            combined = torch.zeros_like(value)
        elif value.shape != shape:
            raise ValueError("trace channels must share one spatial shape")
        if not torch.isfinite(value).all() or torch.any(value < 0):
            raise ValueError(f"{name} must contain finite non-negative values")
        if weight < 0:
            raise ValueError("trace weights must be non-negative")
        combined = combined + float(weight) * value
    assert combined is not None
    mass = combined.sum()
    if not torch.isfinite(mass) or float(mass) <= 0:
        raise ValueError("combined trace map must contain positive mass")
    return combined / mass


def sample_trace_distribution(
    probability_map: torch.Tensor,
    centers_xy: torch.Tensor,
    *,
    uniform_floor: float = 1e-4,
) -> torch.Tensor:
    """Bilinearly sample a trace map at fine-token centers and normalize."""

    if probability_map.ndim != 2:
        raise ValueError("probability_map must have shape [height, width]")
    if centers_xy.ndim != 2 or centers_xy.shape[-1] != 2:
        raise ValueError("centers_xy must have shape [tokens, 2]")
    if not 0 <= uniform_floor < 1:
        raise ValueError("uniform_floor must be in [0, 1)")
    if not torch.isfinite(centers_xy).all():
        raise ValueError("centers must be finite")
    if int(centers_xy.shape[0]) == 0:
        return probability_map.new_empty((0,))
    source = probability_map.float()
    source = source / source.sum().clamp_min(1e-12)
    # align_corners=False maps normalized original centers directly to map-cell
    # centers through x_grid = 2*x_original - 1.
    grid = centers_xy.float().clamp(0.0, 1.0)
    grid = (2.0 * grid - 1.0).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        source.reshape(1, 1, *source.shape),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).reshape(-1)
    sampled = sampled.clamp_min(0)
    distribution = sampled + float(uniform_floor) / sampled.numel()
    return distribution / distribution.sum().clamp_min(1e-12)


def soft_distribution_cross_entropy(
    logits: torch.Tensor,
    target_distribution: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy from a normalized human target to predicted child logits."""

    if logits.ndim != 1 or target_distribution.ndim != 1:
        raise ValueError("logits and target_distribution must be one-dimensional")
    if logits.shape != target_distribution.shape or logits.numel() == 0:
        raise ValueError("logits and target_distribution must share a non-empty shape")
    target = target_distribution.float()
    if not torch.isfinite(target).all() or torch.any(target < 0):
        raise ValueError("target_distribution must be finite and non-negative")
    target = target / target.sum().clamp_min(1e-12)
    return -(target * logits.float().log_softmax(dim=0)).sum()


__all__ = [
    "DEFAULT_TRACE_WEIGHTS",
    "combine_trace_maps",
    "sample_trace_distribution",
    "soft_distribution_cross_entropy",
]
