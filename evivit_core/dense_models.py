"""Lightweight heads for one-pass dense evidence prediction."""

from __future__ import annotations

import torch
from torch import nn


def coordinate_grid(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=0).unsqueeze(0)


class FiLMDenseTraceHead(nn.Module):
    """Question-conditioned dense predictor over frozen Qwen spatial tokens."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.visual_projection = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.question_modulation = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim * 2),
        )
        self.coordinate_projection = nn.Conv2d(2, hidden_dim, kernel_size=1)
        self.spatial = nn.Sequential(
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Dropout2d(dropout),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.trace_head = nn.Sequential(
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, visual: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        if visual.ndim == 3:
            visual = visual.unsqueeze(0)
        if visual.ndim != 4:
            raise ValueError(f"visual must be HxWxD or BxHxWxD, got {tuple(visual.shape)}")
        if question.ndim == 1:
            question = question.unsqueeze(0)
        x = visual.permute(0, 3, 1, 2).float()
        question = question.float()
        base = self.visual_projection(x)
        gamma, beta = self.question_modulation(question).chunk(2, dim=-1)
        gamma = 0.1 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        coords = coordinate_grid(base.shape[-2], base.shape[-1], device=base.device)
        if base.shape[0] > 1:
            coords = coords.expand(base.shape[0], -1, -1, -1)
        conditioned = base * (1.0 + gamma) + beta + self.coordinate_projection(coords)
        hidden = conditioned + self.spatial(conditioned)
        return self.trace_head(hidden).squeeze(1)


class SharedRoiRanker(nn.Module):
    """Listwise scorer for ROI features pooled from one shared image grid."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float()).squeeze(-1)
