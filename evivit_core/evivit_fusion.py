"""Coordinate-aware fine-to-global fusion for EviViT-v2.

The module in this file is deliberately small.  It does not crop images, run
Qwen-ViT, or change the number of visual tokens seen by the language model.
Instead, it performs the one operation that distinguishes the first true
EviViT prototype from the earlier multi-view proxy:

1. a frozen Qwen-ViT produces a coarse global token grid;
2. the same frozen Qwen-ViT re-reads selected pixels from the original image;
3. fine tokens are assigned to their parent locations in the global grid;
4. a trainable residual adapter writes fine evidence into those global tokens.

The output projection is zero-initialized.  Before training, the adapter is
therefore an exact identity map and cannot damage the pretrained Qwen path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FusionDiagnostics:
    """Auditable statistics from one fine-to-global fusion call."""

    global_tokens: int
    fine_tokens: int
    covered_global_tokens: int
    mean_fine_tokens_per_covered_parent: float
    mean_effective_fine_weight: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_tokens": self.global_tokens,
            "fine_tokens": self.fine_tokens,
            "covered_global_tokens": self.covered_global_tokens,
            "mean_fine_tokens_per_covered_parent": (
                self.mean_fine_tokens_per_covered_parent
            ),
            "mean_effective_fine_weight": self.mean_effective_fine_weight,
        }


def assign_fine_tokens_to_global_grid(
    fine_centers_xy: torch.Tensor,
    *,
    global_grid_h: int,
    global_grid_w: int,
    token_order: str = "row_major",
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Return row-major parent indices for normalized fine-token centers.

    Args:
        fine_centers_xy: ``[F, 2]`` tensor in normalized original-image
            ``(x, y)`` coordinates.  Values on an image border are clipped to
            the nearest valid global cell.
        global_grid_h: Height of the pre-merger global token grid.
        global_grid_w: Width of the pre-merger global token grid.
    """

    if fine_centers_xy.ndim != 2 or fine_centers_xy.shape[-1] != 2:
        raise ValueError("fine_centers_xy must have shape [F, 2]")
    if global_grid_h <= 0 or global_grid_w <= 0:
        raise ValueError("global grid dimensions must be positive")
    xy = fine_centers_xy.clamp(0.0, 1.0 - torch.finfo(fine_centers_xy.dtype).eps)
    columns = torch.floor(xy[:, 0] * global_grid_w).to(torch.long)
    rows = torch.floor(xy[:, 1] * global_grid_h).to(torch.long)
    if token_order == "row_major":
        return rows * global_grid_w + columns
    if token_order != "qwen_merge":
        raise ValueError(f"unknown token order: {token_order}")
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")
    if global_grid_h % spatial_merge_size or global_grid_w % spatial_merge_size:
        raise ValueError("Qwen grid dimensions must be divisible by spatial_merge_size")
    merged_w = global_grid_w // spatial_merge_size
    merged_rows = torch.div(rows, spatial_merge_size, rounding_mode="floor")
    merged_columns = torch.div(columns, spatial_merge_size, rounding_mode="floor")
    intra_rows = rows.remainder(spatial_merge_size)
    intra_columns = columns.remainder(spatial_merge_size)
    return (
        ((merged_rows * merged_w + merged_columns) * spatial_merge_size + intra_rows)
        * spatial_merge_size
        + intra_columns
    )


def fine_token_geometry(
    fine_centers_xy: torch.Tensor,
    fine_scales: torch.Tensor,
    parent_indices: torch.Tensor,
    *,
    global_grid_h: int,
    global_grid_w: int,
    token_order: str = "row_major",
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Encode fine-token location relative to its global parent.

    The three returned values are relative x, relative y, and log scale.  They
    tell the adapter where the fine observation lies inside its parent cell and
    how much finer it is than a global cell, without repurposing Qwen's temporal
    MRoPE axis.
    """

    if fine_scales.ndim != 1 or fine_scales.shape[0] != fine_centers_xy.shape[0]:
        raise ValueError("fine_scales must have shape [F]")
    if token_order == "row_major":
        parent_rows = torch.div(parent_indices, global_grid_w, rounding_mode="floor")
        parent_columns = parent_indices.remainder(global_grid_w)
    elif token_order == "qwen_merge":
        if global_grid_h % spatial_merge_size or global_grid_w % spatial_merge_size:
            raise ValueError("Qwen grid must be divisible by spatial_merge_size")
        merged_w = global_grid_w // spatial_merge_size
        merge_unit = spatial_merge_size**2
        block_indices = torch.div(parent_indices, merge_unit, rounding_mode="floor")
        within_block = parent_indices.remainder(merge_unit)
        merged_rows = torch.div(block_indices, merged_w, rounding_mode="floor")
        merged_columns = block_indices.remainder(merged_w)
        intra_rows = torch.div(
            within_block, spatial_merge_size, rounding_mode="floor"
        )
        intra_columns = within_block.remainder(spatial_merge_size)
        parent_rows = merged_rows * spatial_merge_size + intra_rows
        parent_columns = merged_columns * spatial_merge_size + intra_columns
    else:
        raise ValueError(f"unknown token order: {token_order}")
    parent_x = (parent_columns.to(fine_centers_xy.dtype) + 0.5) / global_grid_w
    parent_y = (parent_rows.to(fine_centers_xy.dtype) + 0.5) / global_grid_h
    relative_x = (fine_centers_xy[:, 0] - parent_x) * global_grid_w
    relative_y = (fine_centers_xy[:, 1] - parent_y) * global_grid_h
    global_cell_scale = (1.0 / (global_grid_h * global_grid_w)) ** 0.5
    log_relative_scale = torch.log2(
        fine_scales.clamp_min(torch.finfo(fine_scales.dtype).eps)
        / global_cell_scale
    ).clamp(-8.0, 8.0)
    return torch.stack([relative_x, relative_y, log_relative_scale], dim=-1)


class CoordinateAwareEvidenceFusion(nn.Module):
    """Write high-resolution evidence into a fixed global Qwen-ViT grid.

    Only this adapter is trainable in EviViT-v2.0.  The global and fine token
    features are produced by the same frozen Qwen-ViT.  Fine evidence is first
    projected to a small bottleneck, weighted by the frozen EviMap probability
    and a learned content gate, then averaged within its original-image parent
    cell.  A zero-initialized up projection creates a safe residual update.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        bottleneck_size: int = 256,
        gate_temperature: float = 1.0,
        max_relative_residual: float | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or bottleneck_size <= 0:
            raise ValueError("hidden and bottleneck sizes must be positive")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        if max_relative_residual is not None and not (
            0 < max_relative_residual <= 1
        ):
            raise ValueError("max_relative_residual must be in (0, 1]")
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.gate_temperature = gate_temperature
        self.max_relative_residual = max_relative_residual

        self.fine_norm = nn.LayerNorm(hidden_size)
        self.fine_down = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.geometry_projection = nn.Sequential(
            nn.Linear(3, bottleneck_size, bias=False),
            nn.SiLU(),
            nn.Linear(bottleneck_size, bottleneck_size, bias=False),
        )
        self.content_gate = nn.Linear(bottleneck_size, 1)
        self.output_projection = nn.Linear(bottleneck_size, hidden_size, bias=False)
        nn.init.zeros_(self.output_projection.weight)

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
        token_order: str = "row_major",
        spatial_merge_size: int = 2,
    ) -> tuple[torch.Tensor, FusionDiagnostics]:
        """Fuse one image's pre-merger visual tokens.

        ``global_tokens`` and ``fine_tokens`` use Qwen-ViT's pre-merger hidden
        size.  The method currently handles one image at a time so that each
        image keeps an explicit global grid and original-image coordinate
        system.  Batching is performed by the caller across samples.
        """

        expected_global = global_grid_h * global_grid_w
        if global_tokens.ndim != 2 or global_tokens.shape[0] != expected_global:
            raise ValueError(
                "global_tokens must have shape "
                f"[{expected_global}, hidden_size], got {tuple(global_tokens.shape)}"
            )
        if global_tokens.shape[-1] != self.hidden_size:
            raise ValueError("global token hidden size does not match the adapter")
        if fine_tokens.ndim != 2 or fine_tokens.shape[-1] != self.hidden_size:
            raise ValueError("fine_tokens must have shape [F, hidden_size]")
        fine_count = int(fine_tokens.shape[0])
        if fine_centers_xy.shape != (fine_count, 2):
            raise ValueError("fine_centers_xy must have shape [F, 2]")
        if fine_scales.shape != (fine_count,):
            raise ValueError("fine_scales must have shape [F]")

        if fine_count == 0:
            return global_tokens, FusionDiagnostics(
                global_tokens=expected_global,
                fine_tokens=0,
                covered_global_tokens=0,
                mean_fine_tokens_per_covered_parent=0.0,
                mean_effective_fine_weight=0.0,
            )

        if evidence_weights is None:
            evidence_weights = torch.ones(
                fine_count, device=fine_tokens.device, dtype=fine_tokens.dtype
            )
        if evidence_weights.shape != (fine_count,):
            raise ValueError("evidence_weights must have shape [F]")
        if torch.any(evidence_weights < 0):
            raise ValueError("evidence_weights must be non-negative")

        parent_indices = assign_fine_tokens_to_global_grid(
            fine_centers_xy,
            global_grid_h=global_grid_h,
            global_grid_w=global_grid_w,
            token_order=token_order,
            spatial_merge_size=spatial_merge_size,
        )
        geometry = fine_token_geometry(
            fine_centers_xy,
            fine_scales,
            parent_indices,
            global_grid_h=global_grid_h,
            global_grid_w=global_grid_w,
            token_order=token_order,
            spatial_merge_size=spatial_merge_size,
        )
        fine_hidden = self.fine_down(self.fine_norm(fine_tokens))
        fine_hidden = fine_hidden + self.geometry_projection(geometry.to(fine_hidden.dtype))
        learned_gate = torch.sigmoid(
            self.content_gate(F.silu(fine_hidden)).squeeze(-1)
            / self.gate_temperature
        )
        effective_weights = evidence_weights.to(fine_hidden.dtype) * learned_gate

        weighted_hidden = fine_hidden * effective_weights.unsqueeze(-1)
        aggregated = torch.zeros(
            expected_global,
            self.bottleneck_size,
            device=fine_hidden.device,
            dtype=fine_hidden.dtype,
        )
        normalizer = torch.zeros(
            expected_global,
            device=fine_hidden.device,
            dtype=fine_hidden.dtype,
        )
        counts = torch.zeros(
            expected_global,
            device=fine_hidden.device,
            dtype=torch.long,
        )
        aggregated.index_add_(0, parent_indices, weighted_hidden)
        normalizer.index_add_(0, parent_indices, effective_weights)
        counts.index_add_(0, parent_indices, torch.ones_like(parent_indices))
        covered = normalizer > 0
        aggregated = aggregated / normalizer.clamp_min(1e-6).unsqueeze(-1)

        delta = self.output_projection(F.silu(aggregated))
        if self.max_relative_residual is not None:
            # A per-token trust region prevents a small adapter from erasing
            # the frozen representation it is meant to refine.  Since
            # |tanh(.)| <= 1, the residual energy is bounded by alpha^2 times
            # the global-token energy, with alpha=max_relative_residual.
            reference_rms = (
                global_tokens.float()
                .pow(2)
                .mean(dim=-1, keepdim=True)
                .sqrt()
                .clamp_min(1e-6)
                .detach()
            )
            delta = (
                self.max_relative_residual
                * reference_rms
                * torch.tanh(delta.float() / reference_rms)
            )
        delta = delta.to(global_tokens.dtype)
        fused = global_tokens + delta
        covered_count = int(covered.sum().detach().cpu())
        diagnostics = FusionDiagnostics(
            global_tokens=expected_global,
            fine_tokens=fine_count,
            covered_global_tokens=covered_count,
            mean_fine_tokens_per_covered_parent=(
                float(counts[covered].float().mean().detach().cpu())
                if covered_count
                else 0.0
            ),
            mean_effective_fine_weight=float(effective_weights.mean().detach().cpu()),
        )
        return fused, diagnostics
