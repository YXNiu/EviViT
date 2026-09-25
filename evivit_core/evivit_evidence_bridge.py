"""Sparse mid-ViT global--local evidence interaction for EviViT-v3.

The bridge exchanges information without deleting fine tokens. Fine tokens
attend to a small global neighborhood around their original-image location;
global parent tokens receive a gated aggregate message from their fine tokens.
Both output projections are zero-initialized, making the untrained bridge an
exact identity map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class EvidenceBridgeDiagnostics:
    global_tokens: int
    fine_tokens: int
    covered_global_tokens: int
    mean_valid_global_neighbors: float
    global_relative_residual_l2: float
    fine_relative_residual_l2: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_tokens": self.global_tokens,
            "fine_tokens": self.fine_tokens,
            "covered_global_tokens": self.covered_global_tokens,
            "mean_valid_global_neighbors": self.mean_valid_global_neighbors,
            "global_relative_residual_l2": self.global_relative_residual_l2,
            "fine_relative_residual_l2": self.fine_relative_residual_l2,
        }


def spatial_to_qwen_merge_index(
    rows: torch.Tensor,
    columns: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    merge_size: int,
) -> torch.Tensor:
    """Convert spatial row/column coordinates to Qwen pre-merger order."""

    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("global grid must be divisible by merge_size")
    merged_w = grid_w // merge_size
    block_rows = torch.div(rows, merge_size, rounding_mode="floor")
    block_columns = torch.div(columns, merge_size, rounding_mode="floor")
    intra_rows = rows.remainder(merge_size)
    intra_columns = columns.remainder(merge_size)
    return (
        ((block_rows * merged_w + block_columns) * merge_size + intra_rows)
        * merge_size
        + intra_columns
    )


def qwen_global_neighborhood(
    fine_centers_xy: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    merge_size: int = 2,
    radius: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Qwen indexes, validity, parent indexes, and relative xy offsets."""

    if fine_centers_xy.ndim != 2 or fine_centers_xy.shape[-1] != 2:
        raise ValueError("fine_centers_xy must have shape [F, 2]")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    dtype = fine_centers_xy.dtype
    device = fine_centers_xy.device
    xy = fine_centers_xy.clamp(0.0, 1.0 - torch.finfo(dtype).eps)
    parent_columns = torch.floor(xy[:, 0] * grid_w).to(torch.long)
    parent_rows = torch.floor(xy[:, 1] * grid_h).to(torch.long)
    parent_indexes = spatial_to_qwen_merge_index(
        parent_rows,
        parent_columns,
        grid_h=grid_h,
        grid_w=grid_w,
        merge_size=merge_size,
    )
    offsets = torch.tensor(
        [
            (row_offset, column_offset)
            for row_offset in range(-radius, radius + 1)
            for column_offset in range(-radius, radius + 1)
        ],
        device=device,
        dtype=torch.long,
    )
    rows = parent_rows[:, None] + offsets[None, :, 0]
    columns = parent_columns[:, None] + offsets[None, :, 1]
    valid = (rows >= 0) & (rows < grid_h) & (columns >= 0) & (columns < grid_w)
    safe_rows = rows.clamp(0, grid_h - 1)
    safe_columns = columns.clamp(0, grid_w - 1)
    indexes = spatial_to_qwen_merge_index(
        safe_rows,
        safe_columns,
        grid_h=grid_h,
        grid_w=grid_w,
        merge_size=merge_size,
    )
    center_x = (safe_columns.to(dtype) + 0.5) / grid_w
    center_y = (safe_rows.to(dtype) + 0.5) / grid_h
    relative_xy = torch.stack(
        [
            (center_x - xy[:, None, 0]) * grid_w,
            (center_y - xy[:, None, 1]) * grid_h,
        ],
        dim=-1,
    )
    return indexes, valid, parent_indexes, relative_xy


class SparseGlobalLocalEvidenceBridge(nn.Module):
    """Exchange sparse mid-ViT messages while retaining both token streams."""

    def __init__(
        self,
        hidden_size: int,
        *,
        bridge_dim: int = 256,
        heads: int = 4,
        neighborhood_radius: int = 1,
        spatial_merge_size: int = 2,
        max_relative_residual: float | None = 0.2,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or bridge_dim <= 0:
            raise ValueError("hidden_size and bridge_dim must be positive")
        if bridge_dim % heads:
            raise ValueError("bridge_dim must be divisible by heads")
        if max_relative_residual is not None and not (
            0 < max_relative_residual <= 1
        ):
            raise ValueError("max_relative_residual must be in (0, 1]")
        self.hidden_size = hidden_size
        self.bridge_dim = bridge_dim
        self.heads = heads
        self.head_dim = bridge_dim // heads
        self.neighborhood_radius = neighborhood_radius
        self.spatial_merge_size = spatial_merge_size
        self.max_relative_residual = max_relative_residual

        self.global_norm = nn.LayerNorm(hidden_size)
        self.fine_norm = nn.LayerNorm(hidden_size)
        self.fine_query = nn.Linear(hidden_size, bridge_dim, bias=False)
        self.global_key_value = nn.Linear(hidden_size, bridge_dim * 2, bias=False)
        self.relative_bias = nn.Sequential(
            nn.Linear(3, bridge_dim // 2),
            nn.SiLU(),
            nn.Linear(bridge_dim // 2, heads),
        )
        self.fine_output = nn.Linear(bridge_dim, hidden_size, bias=False)

        self.fine_message = nn.Linear(hidden_size, bridge_dim, bias=False)
        self.fine_geometry = nn.Sequential(
            nn.Linear(3, bridge_dim),
            nn.SiLU(),
            nn.Linear(bridge_dim, bridge_dim),
        )
        self.global_gate_query = nn.Linear(hidden_size, bridge_dim, bias=False)
        self.global_output = nn.Linear(bridge_dim, hidden_size, bias=False)
        nn.init.zeros_(self.fine_output.weight)
        nn.init.zeros_(self.global_output.weight)

    def _bounded_delta(
        self, reference: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        if self.max_relative_residual is None:
            return delta
        scale = reference.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        maximum = self.max_relative_residual * scale
        return (maximum * torch.tanh(delta.float() / maximum)).to(delta.dtype)

    def forward(
        self,
        global_tokens: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_centers_xy: torch.Tensor,
        fine_scales: torch.Tensor,
        *,
        global_grid_h: int,
        global_grid_w: int,
        evidence_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, EvidenceBridgeDiagnostics]:
        if global_tokens.ndim != 2 or global_tokens.shape[-1] != self.hidden_size:
            raise ValueError("global_tokens must have shape [G, hidden_size]")
        expected_global = global_grid_h * global_grid_w
        if int(global_tokens.shape[0]) != expected_global:
            raise ValueError(
                f"global token count {global_tokens.shape[0]} != {expected_global}"
            )
        if fine_tokens.ndim != 2 or fine_tokens.shape[-1] != self.hidden_size:
            raise ValueError("fine_tokens must have shape [F, hidden_size]")
        fine_count = int(fine_tokens.shape[0])
        if fine_centers_xy.shape != (fine_count, 2):
            raise ValueError("fine_centers_xy must have shape [F, 2]")
        if fine_scales.shape != (fine_count,):
            raise ValueError("fine_scales must have shape [F]")
        if evidence_weights is None:
            evidence_weights = fine_scales.new_ones((fine_count,))
        if evidence_weights.shape != (fine_count,):
            raise ValueError("evidence_weights must have shape [F]")
        weights = evidence_weights.clamp(0.0, 1.0).to(fine_tokens.dtype)

        if fine_count == 0:
            diagnostics = EvidenceBridgeDiagnostics(
                global_tokens=int(global_tokens.shape[0]),
                fine_tokens=0,
                covered_global_tokens=0,
                mean_valid_global_neighbors=0.0,
                global_relative_residual_l2=0.0,
                fine_relative_residual_l2=0.0,
            )
            return global_tokens, fine_tokens, diagnostics

        neighbor_indexes, neighbor_valid, parent_indexes, relative_xy = (
            qwen_global_neighborhood(
                fine_centers_xy,
                grid_h=global_grid_h,
                grid_w=global_grid_w,
                merge_size=self.spatial_merge_size,
                radius=self.neighborhood_radius,
            )
        )
        global_normalized = self.global_norm(global_tokens)
        fine_normalized = self.fine_norm(fine_tokens)
        query = self.fine_query(fine_normalized).reshape(
            fine_count, self.heads, self.head_dim
        )
        key_value = self.global_key_value(global_normalized[neighbor_indexes])
        key, value = key_value.chunk(2, dim=-1)
        key = key.reshape(
            fine_count, neighbor_indexes.shape[1], self.heads, self.head_dim
        )
        value = value.reshape_as(key)
        logits = torch.einsum("fhd,fkhd->fhk", query, key) / math.sqrt(
            self.head_dim
        )
        global_cell_scale = (1.0 / (global_grid_h * global_grid_w)) ** 0.5
        log_scale = torch.log2(
            fine_scales.clamp_min(torch.finfo(fine_scales.dtype).eps)
            / global_cell_scale
        ).clamp(-8.0, 8.0)
        geometry = torch.cat(
            [
                relative_xy,
                log_scale[:, None, None].expand(-1, relative_xy.shape[1], -1),
            ],
            dim=-1,
        )
        bias = self.relative_bias(
            geometry.to(self.relative_bias[0].weight.dtype)
        ).permute(0, 2, 1)
        logits = logits + bias.to(logits.dtype)
        logits = logits.masked_fill(~neighbor_valid[:, None, :], -torch.inf)
        attention = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        fine_context = torch.einsum("fhk,fkhd->fhd", attention, value).reshape(
            fine_count, self.bridge_dim
        )
        fine_raw_delta = self.fine_output(fine_context) * weights[:, None]
        fine_delta = self._bounded_delta(fine_tokens, fine_raw_delta)

        parent_columns = torch.floor(
            fine_centers_xy[:, 0].clamp(0, 1 - 1e-7) * global_grid_w
        )
        parent_rows = torch.floor(
            fine_centers_xy[:, 1].clamp(0, 1 - 1e-7) * global_grid_h
        )
        parent_x = (parent_columns + 0.5) / global_grid_w
        parent_y = (parent_rows + 0.5) / global_grid_h
        fine_relative = torch.stack(
            [
                (fine_centers_xy[:, 0] - parent_x) * global_grid_w,
                (fine_centers_xy[:, 1] - parent_y) * global_grid_h,
                log_scale,
            ],
            dim=-1,
        )
        messages = self.fine_message(fine_normalized) + self.fine_geometry(
            fine_relative.to(self.fine_geometry[0].weight.dtype)
        ).to(fine_tokens.dtype)
        weighted_messages = messages * weights[:, None]
        global_context = messages.new_zeros((expected_global, self.bridge_dim))
        global_context.index_add_(0, parent_indexes, weighted_messages)
        denominators = weights.new_zeros((expected_global,))
        denominators.index_add_(0, parent_indexes, weights)
        covered = denominators > 0
        global_context = global_context / denominators.clamp_min(1e-6)[:, None]
        gate_query = self.global_gate_query(global_normalized)
        gates = torch.sigmoid(
            (gate_query * global_context).sum(dim=-1) / math.sqrt(self.bridge_dim)
        )
        global_raw_delta = self.global_output(global_context * gates[:, None])
        global_delta = self._bounded_delta(global_tokens, global_raw_delta)

        global_output = global_tokens + global_delta
        fine_output = fine_tokens + fine_delta
        global_relative = float(
            (
                global_delta.float().pow(2).mean()
                / global_tokens.float().pow(2).mean().clamp_min(1e-6)
            ).detach().cpu()
        )
        fine_relative_l2 = float(
            (
                fine_delta.float().pow(2).mean()
                / fine_tokens.float().pow(2).mean().clamp_min(1e-6)
            ).detach().cpu()
        )
        diagnostics = EvidenceBridgeDiagnostics(
            global_tokens=expected_global,
            fine_tokens=fine_count,
            covered_global_tokens=int(covered.sum().item()),
            mean_valid_global_neighbors=float(
                neighbor_valid.float().sum(dim=1).mean().item()
            ),
            global_relative_residual_l2=global_relative,
            fine_relative_residual_l2=fine_relative_l2,
        )
        return global_output, fine_output, diagnostics
