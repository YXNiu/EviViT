"""Tiny human-trace-supervised context-budget head for EviSplit."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn


FEATURE_NAMES = (
    "secondary_decisive_share",
    "decisive_normalized_entropy",
    "context_normalized_entropy",
    "decisive_context_js",
    "context_mass_outside_decisive",
    "decisive_peak_probability",
)


def _normalized_entropy(probability: torch.Tensor) -> torch.Tensor:
    flat = probability.float().flatten().clamp_min(1e-12)
    flat = flat / flat.sum()
    return -(flat * flat.log()).sum() / math.log(flat.numel())


def _js_divergence(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    left = left.float().flatten().clamp_min(0)
    right = right.float().flatten().clamp_min(0)
    left = left / left.sum().clamp_min(1e-12)
    right = right / right.sum().clamp_min(1e-12)
    middle = 0.5 * (left + right)
    return 0.5 * (
        torch.where(
            left > 0,
            left * (left / middle.clamp_min(1e-12)).log(),
            0,
        ).sum()
        + torch.where(
            right > 0,
            right * (right / middle.clamp_min(1e-12)).log(),
            0,
        ).sum()
    ) / math.log(2.0)


def _box_mask(
    shape: tuple[int, int],
    boxes: Sequence[Sequence[int]],
    *,
    device: torch.device,
) -> torch.Tensor:
    height, width = shape
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    for box in boxes:
        x0, y0, x1, y1 = (int(value) for value in box)
        gx0 = min(width - 1, max(0, int(x0 / 1000 * width)))
        gy0 = min(height - 1, max(0, int(y0 / 1000 * height)))
        gx1 = min(width, max(gx0 + 1, int((x1 * width + 999) / 1000)))
        gy1 = min(height, max(gy0 + 1, int((y1 * height + 999) / 1000)))
        mask[gy0:gy1, gx0:gx1] = True
    return mask


def context_need_feature_tensor(
    decisive_probability: torch.Tensor,
    context_probability: torch.Tensor,
    decisive_boxes: Sequence[Sequence[int]],
    *,
    secondary_decisive_share: float,
) -> torch.Tensor:
    """Extract six inference-available, question-conditioned map statistics."""

    if decisive_probability.ndim != 2 or context_probability.ndim != 2:
        raise ValueError("context-need maps must both have shape [H, W]")
    if decisive_probability.shape != context_probability.shape:
        raise ValueError("context-need maps must have equal shapes")
    decisive = decisive_probability.float().clamp_min(0)
    decisive = decisive / decisive.sum().clamp_min(1e-12)
    context = context_probability.float().clamp_min(0)
    context = context / context.sum().clamp_min(1e-12)
    mask = _box_mask(
        tuple(int(value) for value in decisive.shape),
        decisive_boxes,
        device=decisive.device,
    )
    outside_mass = context[~mask].sum()
    return torch.stack(
        [
            decisive.new_tensor(float(secondary_decisive_share)),
            _normalized_entropy(decisive),
            _normalized_entropy(context),
            _js_divergence(decisive, context),
            outside_mass,
            decisive.max(),
        ]
    )


class ContextNeedHead(nn.Module):
    """A ~100-parameter scalar allocator; maps six features to [q_min, q_max]."""

    def __init__(
        self,
        *,
        hidden_dim: int = 16,
        minimum_context_fraction: float = 0.10,
        maximum_context_fraction: float = 0.35,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not (
            0.1
            <= minimum_context_fraction
            <= maximum_context_fraction
            <= 0.5
        ):
            raise ValueError("context-fraction range must lie in [0.1, 0.5]")
        self.hidden_dim = hidden_dim
        self.minimum_context_fraction = minimum_context_fraction
        self.maximum_context_fraction = maximum_context_fraction
        self.register_buffer(
            "feature_mean", torch.zeros(len(FEATURE_NAMES))
        )
        self.register_buffer(
            "feature_scale", torch.ones(len(FEATURE_NAMES))
        )
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def set_normalization(
        self,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
    ) -> None:
        if feature_mean.shape != self.feature_mean.shape:
            raise ValueError("feature mean has the wrong shape")
        if feature_scale.shape != self.feature_scale.shape:
            raise ValueError("feature scale has the wrong shape")
        self.feature_mean.copy_(feature_mean)
        self.feature_scale.copy_(feature_scale.clamp_min(1e-6))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features.float() - self.feature_mean) / self.feature_scale
        unit = torch.sigmoid(self.network(normalized).squeeze(-1))
        return self.minimum_context_fraction + (
            self.maximum_context_fraction - self.minimum_context_fraction
        ) * unit


def load_context_need_head(
    path: Path,
    *,
    device: str | torch.device,
) -> ContextNeedHead:
    payload: dict[str, Any] = torch.load(
        path, map_location="cpu", weights_only=False
    )
    if payload.get("version") != "evivit_context_need_head_v1":
        raise ValueError("unsupported context-need checkpoint")
    config = payload["model_config"]
    model = ContextNeedHead(
        hidden_dim=int(config["hidden_dim"]),
        minimum_context_fraction=float(
            config["minimum_context_fraction"]
        ),
        maximum_context_fraction=float(
            config["maximum_context_fraction"]
        ),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()
