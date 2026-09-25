"""Parent-preserving hierarchical token interaction for TraceFovea.

The block keeps every global parent and every fine child.  It never pools,
drops, reorders, or replaces either stream.  Each child has exactly one global
parent determined by its original-image center; a parent receives the mean
message from its children, while each child receives its parent and sibling
context.  Zero-initialized output projections make the untrained module an
exact identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from evivit_core.evivit_evidence_bridge import (
    EvidenceBridgeDiagnostics,
    spatial_to_qwen_merge_index,
)


@dataclass(frozen=True)
class TraceFoveaDiagnostics:
    parent_tokens: int
    child_tokens: int
    assigned_children: int
    unique_child_ownership_rate: float
    covered_parents: int
    parent_coverage_rate: float
    maximum_children_per_parent: int
    parent_relative_residual_l2: float
    child_relative_residual_l2: float
    child_gate_mean: float | None = None
    child_gate_std: float | None = None
    child_importance_mean: float | None = None
    child_importance_std: float | None = None
    child_importance_maximum: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_tokens": self.parent_tokens,
            "child_tokens": self.child_tokens,
            "assigned_children": self.assigned_children,
            "unique_child_ownership_rate": self.unique_child_ownership_rate,
            "covered_parents": self.covered_parents,
            "parent_coverage_rate": self.parent_coverage_rate,
            "maximum_children_per_parent": self.maximum_children_per_parent,
            "parent_relative_residual_l2": self.parent_relative_residual_l2,
            "child_relative_residual_l2": self.child_relative_residual_l2,
            "child_gate_mean": self.child_gate_mean,
            "child_gate_std": self.child_gate_std,
            "child_importance_mean": self.child_importance_mean,
            "child_importance_std": self.child_importance_std,
            "child_importance_maximum": self.child_importance_maximum,
        }


def normalized_trace_importance(
    logits: torch.Tensor,
    *,
    minimum: float = 0.25,
    maximum: float = 3.0,
    return_float32: bool = False,
) -> torch.Tensor:
    """Convert trace logits into bounded, shift-invariant residual gains.

    A uniform distribution maps to gain 1. Clamping prevents one imperfect
    evidence prediction from erasing or exploding a fine token. The final
    normalization keeps the mean residual strength unchanged, so the mode
    reallocates a fixed residual budget instead of adding more capacity.
    """

    if logits.ndim != 1 or logits.numel() == 0:
        raise ValueError("logits must be a non-empty one-dimensional tensor")
    if not 0 < minimum <= 1.0:
        raise ValueError("minimum must be in (0, 1]")
    if maximum < 1.0 or maximum < minimum:
        raise ValueError("maximum must be at least 1 and no smaller than minimum")
    probability = logits.float().softmax(dim=0).clamp_min(1e-12)
    probability = probability / probability.sum()
    raw = probability * logits.numel()
    # Project the positive relative weights onto the box-constrained simplex
    # {w: mean(w)=1, minimum<=w_i<=maximum}. A scalar rescaling followed by
    # clipping is monotonic, so bisection gives a deterministic differentiable-
    # almost-everywhere projection without changing token order.
    lower_scale = raw.new_zeros(())
    upper_scale = 1.0 / raw.min().clamp_min(1e-12)
    for _ in range(48):
        middle_scale = (lower_scale + upper_scale) * 0.5
        mean = (raw * middle_scale).clamp(
            min=minimum, max=maximum
        ).mean()
        lower_scale = torch.where(
            mean < 1.0, middle_scale, lower_scale
        )
        upper_scale = torch.where(
            mean >= 1.0, middle_scale, upper_scale
        )
    importance = (raw * ((lower_scale + upper_scale) * 0.5)).clamp(
        min=minimum, max=maximum
    )
    # Importance commonly differs from one by much less than the BF16 unit in
    # the last place around one.  Casting here can therefore turn every gain
    # back into exactly one during the first optimizer updates, removing the
    # very spatial variation that trace supervision is meant to learn.  Keep
    # the historical cast by default so old checkpoints remain bit-compatible;
    # new trace-calibrated paths opt into the numerically safe FP32 result.
    return importance if return_float32 else importance.to(logits.dtype)


def assign_children_to_qwen_parents(
    child_centers_xy: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    spatial_merge_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map every child to one Qwen-ordered global patch and return local offsets.

    Coordinates are normalized original-image `(x, y)` centers.  Returned
    offsets are measured in global-grid cells relative to the selected parent
    center, so they remain in approximately `[-0.5, 0.5]`.
    """

    if child_centers_xy.ndim != 2 or child_centers_xy.shape[-1] != 2:
        raise ValueError("child_centers_xy must have shape [children, 2]")
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("global grid dimensions must be positive")
    if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
        raise ValueError("global grid must be divisible by spatial_merge_size")
    if not torch.isfinite(child_centers_xy).all():
        raise ValueError("child centers must be finite")
    dtype = child_centers_xy.dtype
    epsilon = torch.finfo(dtype).eps
    centers = child_centers_xy.clamp(0.0, 1.0 - epsilon)
    columns = torch.floor(centers[:, 0] * grid_w).to(torch.long)
    rows = torch.floor(centers[:, 1] * grid_h).to(torch.long)
    parent_indexes = spatial_to_qwen_merge_index(
        rows,
        columns,
        grid_h=grid_h,
        grid_w=grid_w,
        merge_size=spatial_merge_size,
    )
    parent_x = (columns.to(dtype) + 0.5) / grid_w
    parent_y = (rows.to(dtype) + 0.5) / grid_h
    offsets = torch.stack(
        [
            (centers[:, 0] - parent_x) * grid_w,
            (centers[:, 1] - parent_y) * grid_h,
        ],
        dim=-1,
    )
    return parent_indexes, offsets


class TraceFoveationBlock(nn.Module):
    """Zero-initialized parent–child–sibling residual interaction."""

    def __init__(
        self,
        hidden_size: int,
        *,
        fovea_dim: int = 256,
        spatial_merge_size: int = 2,
        max_relative_residual: float = 0.20,
        use_child_gate: bool = False,
        modulate_child_residual: bool = False,
        normalized_child_routing: bool = False,
        preserve_float32_importance: bool = False,
        minimum_child_importance: float = 0.25,
        maximum_child_importance: float = 3.0,
        coverage_masked_parent_writeback: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or fovea_dim <= 0:
            raise ValueError("hidden_size and fovea_dim must be positive")
        if spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        if not 0 < max_relative_residual <= 1:
            raise ValueError("max_relative_residual must be in (0, 1]")
        self.hidden_size = int(hidden_size)
        self.fovea_dim = int(fovea_dim)
        self.spatial_merge_size = int(spatial_merge_size)
        self.max_relative_residual = float(max_relative_residual)
        self.use_child_gate = bool(use_child_gate)
        self.modulate_child_residual = bool(modulate_child_residual)
        self.normalized_child_routing = bool(normalized_child_routing)
        self.preserve_float32_importance = bool(
            preserve_float32_importance
        )
        self.minimum_child_importance = float(minimum_child_importance)
        self.maximum_child_importance = float(maximum_child_importance)
        self.coverage_masked_parent_writeback = bool(
            coverage_masked_parent_writeback
        )
        if self.modulate_child_residual and not self.use_child_gate:
            raise ValueError(
                "modulate_child_residual requires use_child_gate=True"
            )
        if self.normalized_child_routing and not self.use_child_gate:
            raise ValueError(
                "normalized_child_routing requires use_child_gate=True"
            )
        if not 0 < self.minimum_child_importance <= 1.0:
            raise ValueError("minimum_child_importance must be in (0, 1]")
        if (
            self.maximum_child_importance < 1.0
            or self.maximum_child_importance < self.minimum_child_importance
        ):
            raise ValueError(
                "maximum_child_importance must be at least 1 and no smaller "
                "than minimum_child_importance"
            )
        self.last_child_gate_logits: torch.Tensor | None = None

        self.parent_norm = nn.LayerNorm(hidden_size)
        self.child_norm = nn.LayerNorm(hidden_size)
        self.parent_value = nn.Linear(hidden_size, fovea_dim, bias=False)
        self.child_value = nn.Linear(hidden_size, fovea_dim, bias=False)
        self.geometry = nn.Sequential(
            nn.Linear(3, fovea_dim),
            nn.SiLU(),
            nn.Linear(fovea_dim, fovea_dim),
        )
        self.child_mixer = nn.Sequential(
            nn.Linear(3 * fovea_dim, fovea_dim),
            nn.SiLU(),
        )
        self.parent_mixer = nn.Sequential(
            nn.Linear(2 * fovea_dim, fovea_dim),
            nn.SiLU(),
        )
        self.child_gate = (
            nn.Sequential(
                nn.Linear(3 * fovea_dim + 1, fovea_dim),
                nn.SiLU(),
                nn.Linear(fovea_dim, 1),
            )
            if self.use_child_gate
            else None
        )
        self.child_output = nn.Linear(fovea_dim, hidden_size, bias=False)
        self.parent_output = nn.Linear(fovea_dim, hidden_size, bias=False)
        nn.init.zeros_(self.child_output.weight)
        nn.init.zeros_(self.parent_output.weight)

    def _bounded_delta(
        self, reference: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        rms = (
            reference.float()
            .pow(2)
            .mean(dim=-1, keepdim=True)
            .sqrt()
            .clamp_min(1e-6)
        )
        maximum = self.max_relative_residual * rms
        return (maximum * torch.tanh(delta.float() / maximum)).to(delta.dtype)

    def forward(
        self,
        parent_tokens: torch.Tensor,
        child_tokens: torch.Tensor,
        child_centers_xy: torch.Tensor,
        child_scales: torch.Tensor,
        *,
        global_grid_h: int,
        global_grid_w: int,
        evidence_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, TraceFoveaDiagnostics]:
        if parent_tokens.ndim != 2 or parent_tokens.shape[-1] != self.hidden_size:
            raise ValueError("parent_tokens must have shape [parents, hidden_size]")
        if child_tokens.ndim != 2 or child_tokens.shape[-1] != self.hidden_size:
            raise ValueError("child_tokens must have shape [children, hidden_size]")
        expected_parents = global_grid_h * global_grid_w
        if int(parent_tokens.shape[0]) != expected_parents:
            raise ValueError(
                f"parent token count {parent_tokens.shape[0]} != {expected_parents}"
            )
        children = int(child_tokens.shape[0])
        if child_centers_xy.shape != (children, 2):
            raise ValueError("child_centers_xy must have shape [children, 2]")
        if child_scales.shape != (children,):
            raise ValueError("child_scales must have shape [children]")
        if children == 0:
            self.last_child_gate_logits = None
            diagnostics = TraceFoveaDiagnostics(
                parent_tokens=expected_parents,
                child_tokens=0,
                assigned_children=0,
                unique_child_ownership_rate=1.0,
                covered_parents=0,
                parent_coverage_rate=0.0,
                maximum_children_per_parent=0,
                parent_relative_residual_l2=0.0,
                child_relative_residual_l2=0.0,
            )
            return parent_tokens, child_tokens, diagnostics

        parent_indexes, relative_xy = assign_children_to_qwen_parents(
            child_centers_xy,
            grid_h=global_grid_h,
            grid_w=global_grid_w,
            spatial_merge_size=self.spatial_merge_size,
        )
        if int(parent_indexes.min()) < 0 or int(parent_indexes.max()) >= expected_parents:
            raise RuntimeError("child parent assignment is out of bounds")
        parent_features = self.parent_value(self.parent_norm(parent_tokens))
        child_features = self.child_value(self.child_norm(child_tokens))
        geometry = self.geometry(
            torch.cat(
                [
                    relative_xy.to(child_tokens.dtype),
                    child_scales.clamp_min(1e-8).log()[:, None].to(child_tokens.dtype),
                ],
                dim=-1,
            )
        )

        gate_logits: torch.Tensor | None = None
        child_importance: torch.Tensor | None = None
        if self.child_gate is not None:
            if evidence_weights is None:
                evidence_weights = child_scales.new_ones((children,))
            if evidence_weights.shape != (children,):
                raise ValueError("evidence_weights must have shape [children]")
            gate_logits = self.child_gate(
                torch.cat(
                    [
                        child_features,
                        parent_features[parent_indexes],
                        geometry,
                        evidence_weights.to(child_features.dtype)[:, None],
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            if self.modulate_child_residual or self.normalized_child_routing:
                child_importance = normalized_trace_importance(
                    gate_logits,
                    minimum=self.minimum_child_importance,
                    maximum=self.maximum_child_importance,
                    return_float32=self.preserve_float32_importance,
                )
            gate_values = (
                child_importance
                if self.normalized_child_routing
                else gate_logits.sigmoid()
            )
            assert gate_values is not None
            self.last_child_gate_logits = gate_logits
        else:
            gate_values = child_features.new_ones((children,))
            self.last_child_gate_logits = None

        if self.normalized_child_routing and self.preserve_float32_importance:
            # Preserve early differences around gain=1 while aggregating
            # siblings. Casting those gains to BF16 before multiplication
            # recreates the uniform-routing dead zone fixed in v20.
            sibling_sum_float = torch.zeros(
                (expected_parents, self.fovea_dim),
                device=child_features.device,
                dtype=torch.float32,
            )
            sibling_sum_float.index_add_(
                0,
                parent_indexes,
                child_features.float() * gate_values.float()[:, None],
            )
            gate_mass_float = torch.zeros(
                (expected_parents,),
                device=child_features.device,
                dtype=torch.float32,
            )
            gate_mass_float.index_add_(
                0, parent_indexes, gate_values.float()
            )
            sibling_mean = (
                sibling_sum_float
                / gate_mass_float.clamp_min(1e-4)[:, None]
            ).to(child_features.dtype)
        else:
            sibling_sum = child_features.new_zeros(
                (expected_parents, self.fovea_dim)
            )
            sibling_sum.index_add_(
                0, parent_indexes, child_features * gate_values[:, None]
            )
            gate_mass = child_features.new_zeros((expected_parents,))
            gate_mass.index_add_(0, parent_indexes, gate_values)
            sibling_mean = sibling_sum / gate_mass.clamp_min(1e-4)[:, None]
        child_counts = child_features.new_zeros((expected_parents,))
        child_counts.index_add_(0, parent_indexes, torch.ones_like(child_scales))

        child_message = self.child_mixer(
            torch.cat(
                [
                    child_features,
                    parent_features[parent_indexes],
                    sibling_mean[parent_indexes] + geometry,
                ],
                dim=-1,
            )
        )
        parent_message = self.parent_mixer(
            torch.cat([parent_features, sibling_mean], dim=-1)
        )
        child_delta = self._bounded_delta(
            child_tokens, self.child_output(child_message)
        )
        if child_importance is not None:
            child_delta = (
                child_delta.float() * child_importance.float()[:, None]
            ).to(child_delta.dtype)
        parent_delta = self._bounded_delta(
            parent_tokens, self.parent_output(parent_message)
        )
        if self.coverage_masked_parent_writeback:
            # Fine evidence is an additive refinement of an always-present
            # native Global canvas. Parents with no assigned Fine child must
            # therefore remain exact identities instead of receiving an MLP
            # residual computed from [parent_feature, zero_sibling]. Covered
            # parents keep the checkpoint's original learned update exactly.
            parent_delta = parent_delta * (child_counts > 0).to(
                parent_delta.dtype
            )[:, None]
        updated_parents = parent_tokens + parent_delta
        updated_children = child_tokens + child_delta
        parent_relative = (
            parent_delta.float().pow(2).mean()
            / parent_tokens.float().pow(2).mean().clamp_min(1e-6)
        )
        child_relative = (
            child_delta.float().pow(2).mean()
            / child_tokens.float().pow(2).mean().clamp_min(1e-6)
        )
        covered = int((child_counts > 0).sum().detach().cpu())
        gate_mean = (
            float(gate_values.detach().float().mean().cpu())
            if gate_logits is not None
            else None
        )
        gate_std = (
            float(gate_values.detach().float().std(unbiased=False).cpu())
            if gate_logits is not None
            else None
        )
        importance_mean = (
            float(child_importance.detach().float().mean().cpu())
            if child_importance is not None
            else None
        )
        importance_std = (
            float(
                child_importance.detach().float().std(unbiased=False).cpu()
            )
            if child_importance is not None
            else None
        )
        importance_maximum = (
            float(child_importance.detach().float().max().cpu())
            if child_importance is not None
            else None
        )
        diagnostics = TraceFoveaDiagnostics(
            parent_tokens=expected_parents,
            child_tokens=children,
            assigned_children=int(parent_indexes.numel()),
            unique_child_ownership_rate=1.0,
            covered_parents=covered,
            parent_coverage_rate=covered / expected_parents,
            maximum_children_per_parent=int(child_counts.max().detach().cpu()),
            parent_relative_residual_l2=float(parent_relative.detach().cpu()),
            child_relative_residual_l2=float(child_relative.detach().cpu()),
            child_gate_mean=gate_mean,
            child_gate_std=gate_std,
            child_importance_mean=importance_mean,
            child_importance_std=importance_std,
            child_importance_maximum=importance_maximum,
        )
        return updated_parents, updated_children, diagnostics


class TraceFoveaBridgeAdapter(nn.Module):
    """Drop-in bridge interface for the existing segmented Qwen-ViT runtime.

    This adapter lets the first fixed-budget experiment change only the fusion
    mechanism.  The PTEA map, decoded regions, token budgets, original-pixel
    rereading, upper frozen Qwen-ViT blocks, merger, MRoPE, and LLM all remain
    byte-for-byte the same as the v4 runtime.
    """

    def __init__(self, block: TraceFoveationBlock) -> None:
        super().__init__()
        self.block = block
        self.hidden_size = block.hidden_size
        self.last_trace_diagnostics: TraceFoveaDiagnostics | None = None
        self.last_child_gate_logits: torch.Tensor | None = None
        self.last_child_centers_xy: torch.Tensor | None = None

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
        parents, children, diagnostics = self.block(
            global_tokens,
            fine_tokens,
            fine_centers_xy,
            fine_scales,
            global_grid_h=global_grid_h,
            global_grid_w=global_grid_w,
            evidence_weights=evidence_weights,
        )
        self.last_trace_diagnostics = diagnostics
        self.last_child_gate_logits = self.block.last_child_gate_logits
        self.last_child_centers_xy = fine_centers_xy
        return parents, children, EvidenceBridgeDiagnostics(
            global_tokens=diagnostics.parent_tokens,
            fine_tokens=diagnostics.child_tokens,
            covered_global_tokens=diagnostics.covered_parents,
            mean_valid_global_neighbors=1.0 if diagnostics.child_tokens else 0.0,
            global_relative_residual_l2=diagnostics.parent_relative_residual_l2,
            fine_relative_residual_l2=diagnostics.child_relative_residual_l2,
        )


__all__ = [
    "TraceFoveaDiagnostics",
    "TraceFoveaBridgeAdapter",
    "TraceFoveationBlock",
    "assign_children_to_qwen_parents",
    "normalized_trace_importance",
]
