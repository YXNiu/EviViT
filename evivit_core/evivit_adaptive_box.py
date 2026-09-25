"""PTEA-anchored continuous box refinement for EviViT-v6.

The verified EviViT-v5 region portfolio remains the safety anchor.  A tiny,
identity-initialized head predicts bounded residuals in centre and log-size;
zero residuals reproduce v5 exactly.  The tensor geometry in this module is
fully differentiable so a later QA loss can update only the box head through a
``grid_sample`` crop while all Qwen and EviViT weights remain frozen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class AdaptiveBoxConfig:
    input_dim: int
    hidden_dim: int = 256
    bottleneck_dim: int = 128
    maximum_center_factor: float = 1.50
    maximum_absolute_log_scale: float = 1.20
    minimum_side: float = 0.02
    maximum_side: float = 0.95

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class AdaptiveBoxHead(nn.Module):
    """Shared residual MLP for the three PTEA region roles."""

    def __init__(self, config: AdaptiveBoxConfig) -> None:
        super().__init__()
        if config.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if config.hidden_dim <= 0 or config.bottleneck_dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        if config.maximum_center_factor <= 0:
            raise ValueError("maximum_center_factor must be positive")
        if config.maximum_absolute_log_scale <= 0:
            raise ValueError("maximum log scale must be positive")
        if not 0 < config.minimum_side <= config.maximum_side <= 1:
            raise ValueError("invalid side interval")
        self.config = config
        self.register_buffer("feature_mean", torch.zeros(config.input_dim))
        self.register_buffer("feature_scale", torch.ones(config.input_dim))
        self.network = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.bottleneck_dim),
            nn.GELU(),
            nn.Linear(config.bottleneck_dim, 4),
        )
        # Identity initialization is the safety contract: before learning, v6
        # emits exactly the v5 anchors.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def set_normalization(
        self, feature_mean: torch.Tensor, feature_scale: torch.Tensor
    ) -> None:
        if feature_mean.shape != self.feature_mean.shape:
            raise ValueError("feature mean has the wrong shape")
        if feature_scale.shape != self.feature_scale.shape:
            raise ValueError("feature scale has the wrong shape")
        self.feature_mean.copy_(feature_mean)
        self.feature_scale.copy_(feature_scale.clamp_min(1e-6))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features.float() - self.feature_mean) / self.feature_scale
        raw = self.network(normalized)
        centre = self.config.maximum_center_factor * torch.tanh(raw[..., :2])
        scale = self.config.maximum_absolute_log_scale * torch.tanh(raw[..., 2:])
        return torch.cat((centre, scale), dim=-1)


def _box_center_size(boxes: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if boxes.shape[-1] != 4:
        raise ValueError("boxes must end in xyxy dimension four")
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    width = (x2 - x1).clamp_min(1e-6)
    height = (y2 - y1).clamp_min(1e-6)
    return (x1 + x2) / 2, (y1 + y2) / 2, width, height


def residual_target(
    anchors: torch.Tensor,
    targets: torch.Tensor,
    *,
    maximum_center_factor: float = 1.50,
    maximum_absolute_log_scale: float = 1.20,
) -> torch.Tensor:
    """Encode target boxes as the bounded residuals predicted by the head."""

    acx, acy, aw, ah = _box_center_size(anchors)
    tcx, tcy, tw, th = _box_center_size(targets)
    centre = torch.stack(((tcx - acx) / aw, (tcy - acy) / ah), dim=-1)
    centre = centre.clamp(-maximum_center_factor, maximum_center_factor)
    scale = torch.stack((torch.log(tw / aw), torch.log(th / ah)), dim=-1)
    scale = scale.clamp(
        -maximum_absolute_log_scale, maximum_absolute_log_scale
    )
    return torch.cat((centre, scale), dim=-1)


def apply_residual(
    anchors: torch.Tensor,
    residuals: torch.Tensor,
    *,
    minimum_side: float = 0.02,
    maximum_side: float = 0.95,
) -> torch.Tensor:
    """Apply normalized centre/log-size residuals to normalized xyxy boxes."""

    if anchors.shape != residuals.shape:
        raise ValueError("anchors and residuals must have identical [...,4] shapes")
    acx, acy, aw, ah = _box_center_size(anchors)
    dx, dy, log_w, log_h = residuals.unbind(dim=-1)
    cx = acx + dx * aw
    cy = acy + dy * ah
    width = (aw * torch.exp(log_w)).clamp(minimum_side, maximum_side)
    height = (ah * torch.exp(log_h)).clamp(minimum_side, maximum_side)

    # Move rather than shrink a box at the image boundary.  This retains its
    # predicted scale while keeping all coordinates legal.
    cx = cx.clamp(width / 2, 1.0 - width / 2)
    cy = cy.clamp(height / 2, 1.0 - height / 2)
    return torch.stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
        dim=-1,
    )


def apply_residual_box(
    anchor: Sequence[float],
    residual: Sequence[float],
    *,
    minimum_side: float = 0.02,
    maximum_side: float = 0.95,
) -> tuple[float, float, float, float]:
    """Torch-free-facing convenience wrapper used by inference code."""

    box = apply_residual(
        torch.tensor([anchor], dtype=torch.float32),
        torch.tensor([residual], dtype=torch.float32),
        minimum_side=minimum_side,
        maximum_side=maximum_side,
    )[0]
    return tuple(float(value) for value in box.tolist())


def residual_ensemble_uncertainty(predictions: torch.Tensor) -> torch.Tensor:
    """Return one RMS disagreement score per region for an ensemble."""

    if predictions.ndim < 3 or predictions.shape[-1] != 4:
        raise ValueError("predictions must have shape [heads, ..., 4]")
    if predictions.shape[0] < 2:
        raise ValueError("at least two heads are required for uncertainty")
    centered = predictions.float() - predictions.float().mean(dim=0, keepdim=True)
    return centered.square().mean(dim=(0, -1)).sqrt()


def union_boxes(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return the smallest normalized xyxy boxes containing both inputs."""

    if first.shape != second.shape or first.shape[-1] != 4:
        raise ValueError("boxes must have identical [..., 4] shapes")
    return torch.cat(
        (
            torch.minimum(first[..., :2], second[..., :2]),
            torch.maximum(first[..., 2:], second[..., 2:]),
        ),
        dim=-1,
    ).clamp(0.0, 1.0)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    _, _, width, height = _box_center_size(boxes)
    return width * height


def intersection_area(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    left_top = torch.maximum(first[..., :2], second[..., :2])
    right_bottom = torch.minimum(first[..., 2:], second[..., 2:])
    size = (right_bottom - left_top).clamp_min(0)
    return size[..., 0] * size[..., 1]


def generalized_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    overlap = intersection_area(first, second)
    union = box_area(first) + box_area(second) - overlap
    iou = overlap / union.clamp_min(1e-8)
    enclosure_left_top = torch.minimum(first[..., :2], second[..., :2])
    enclosure_right_bottom = torch.maximum(first[..., 2:], second[..., 2:])
    enclosure_size = (enclosure_right_bottom - enclosure_left_top).clamp_min(0)
    enclosure = enclosure_size[..., 0] * enclosure_size[..., 1]
    return iou - (enclosure - union) / enclosure.clamp_min(1e-8)


def target_coverage(regions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return intersection_area(regions, targets) / box_area(targets).clamp_min(1e-8)


def load_adaptive_box_head(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[AdaptiveBoxHead, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("version") != "evivit_v6_adaptivebox_head_v1":
        raise ValueError(f"unsupported AdaptiveBox checkpoint: {checkpoint.get('version')}")
    config = AdaptiveBoxConfig(**checkpoint["model_config"])
    model = AdaptiveBoxHead(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


__all__ = [
    "AdaptiveBoxConfig",
    "AdaptiveBoxHead",
    "apply_residual",
    "apply_residual_box",
    "box_area",
    "generalized_iou",
    "intersection_area",
    "load_adaptive_box_head",
    "residual_ensemble_uncertainty",
    "residual_target",
    "target_coverage",
    "union_boxes",
]
