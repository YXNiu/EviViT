"""Long-range Global--Fine topology relay for EviViT-v5.

The existing sparse Bridge is deliberately local: every Fine token reads a
3x3 Global neighborhood and writes only to its own Global parent.  This relay
adds the missing long-range path without deleting either stream.  It pools the
complete Global grid into a small coordinate-preserving anchor lattice, builds
one summary per Fine region, lets each region summary attend to all anchors and
other regions with relative-geometry bias, then writes a bounded residual back
to the Fine tokens in that region.

The output projection is zero-initialized, so attaching an untrained relay is
an exact identity operation.  No parameters depend on Global/Fine token count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from evivit_core.evivit_evidence_bridge import spatial_to_qwen_merge_index


@dataclass(frozen=True)
class GlobalAnchorRelayDiagnostics:
    global_tokens: int
    fine_tokens: int
    regions: int
    anchors: int
    fine_relative_residual_l2: float
    gate_mean: float
    gate_std: float
    attention_entropy_normalized: float
    pre_projection_anchor_attention_mass: float
    anchor_attention_mass: float
    region_attention_mass: float
    self_region_attention_mass: float
    anchor_floor_activation_rate: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_tokens": self.global_tokens,
            "fine_tokens": self.fine_tokens,
            "regions": self.regions,
            "anchors": self.anchors,
            "fine_relative_residual_l2": self.fine_relative_residual_l2,
            "gate_mean": self.gate_mean,
            "gate_std": self.gate_std,
            "attention_entropy_normalized": self.attention_entropy_normalized,
            "pre_projection_anchor_attention_mass": (
                self.pre_projection_anchor_attention_mass
            ),
            "anchor_attention_mass": self.anchor_attention_mass,
            "region_attention_mass": self.region_attention_mass,
            "self_region_attention_mass": self.self_region_attention_mass,
            "anchor_floor_activation_rate": self.anchor_floor_activation_rate,
        }


def qwen_anchor_assignments(
    *,
    grid_h: int,
    grid_w: int,
    anchor_h: int,
    anchor_w: int,
    merge_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Qwen token indexes, anchor ids and normalized token centers.

    Rows/columns are first enumerated in ordinary spatial order and then mapped
    to Qwen's pre-merger token order.  This avoids silently pooling unrelated
    positions when the native sequence groups tokens by merge cell.
    """

    if min(grid_h, grid_w, anchor_h, anchor_w, merge_size) <= 0:
        raise ValueError("grid and anchor dimensions must be positive")
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("global grid must be divisible by merge_size")
    rows, columns = torch.meshgrid(
        torch.arange(grid_h, device=device),
        torch.arange(grid_w, device=device),
        indexing="ij",
    )
    flat_rows = rows.flatten()
    flat_columns = columns.flatten()
    qwen_indexes = spatial_to_qwen_merge_index(
        flat_rows,
        flat_columns,
        grid_h=grid_h,
        grid_w=grid_w,
        merge_size=merge_size,
    )
    anchor_rows = torch.div(flat_rows * anchor_h, grid_h, rounding_mode="floor")
    anchor_columns = torch.div(flat_columns * anchor_w, grid_w, rounding_mode="floor")
    anchor_ids = anchor_rows * anchor_w + anchor_columns
    centers = torch.stack(
        [
            (flat_columns.float() + 0.5) / grid_w,
            (flat_rows.float() + 0.5) / grid_h,
        ],
        dim=-1,
    )
    return qwen_indexes, anchor_ids, centers


class GlobalAnchorRelay(nn.Module):
    """Coordinate-aware long-range relay from Global anchors to Fine regions."""

    def __init__(
        self,
        hidden_size: int,
        *,
        relay_dim: int = 256,
        heads: int = 4,
        anchor_grid: tuple[int, int] = (4, 4),
        spatial_merge_size: int = 2,
        max_relative_residual: float = 0.1,
        minimum_anchor_attention_mass: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or relay_dim <= 0 or relay_dim % heads:
            raise ValueError("hidden_size/relay_dim must be positive and divisible by heads")
        if min(anchor_grid) <= 0:
            raise ValueError("anchor_grid must be positive")
        if not 0 < max_relative_residual <= 1:
            raise ValueError("max_relative_residual must be in (0, 1]")
        if not 0 <= minimum_anchor_attention_mass < 1:
            raise ValueError("minimum_anchor_attention_mass must be in [0, 1)")
        self.hidden_size = hidden_size
        self.relay_dim = relay_dim
        self.heads = heads
        self.head_dim = relay_dim // heads
        self.anchor_grid = tuple(int(value) for value in anchor_grid)
        self.spatial_merge_size = spatial_merge_size
        self.max_relative_residual = max_relative_residual
        self.minimum_anchor_attention_mass = float(
            minimum_anchor_attention_mass
        )

        self.global_norm = nn.LayerNorm(hidden_size)
        self.fine_norm = nn.LayerNorm(hidden_size)
        self.region_query = nn.Linear(hidden_size, relay_dim, bias=False)
        self.node_key_value = nn.Linear(hidden_size, relay_dim * 2, bias=False)
        self.geometry_bias = nn.Sequential(
            nn.Linear(4, relay_dim // 2),
            nn.SiLU(),
            nn.Linear(relay_dim // 2, heads),
        )
        self.fine_gate = nn.Linear(hidden_size + relay_dim, 1)
        self.fine_output = nn.Linear(relay_dim, hidden_size, bias=False)
        nn.init.zeros_(self.fine_output.weight)

    def _bounded_delta(self, reference: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        scale = reference.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        maximum = self.max_relative_residual * scale
        return (maximum * torch.tanh(delta.float() / maximum)).to(delta.dtype)

    @staticmethod
    def _weighted_means(
        values: torch.Tensor,
        group_ids: torch.Tensor,
        weights: torch.Tensor,
        groups: int,
    ) -> torch.Tensor:
        output = values.new_zeros((groups, values.shape[-1]))
        output.index_add_(0, group_ids, values * weights[:, None])
        denominator = weights.new_zeros((groups,))
        denominator.index_add_(0, group_ids, weights)
        return output / denominator.clamp_min(1e-6)[:, None]

    def forward(
        self,
        global_tokens: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_centers_xy: torch.Tensor,
        fine_scales: torch.Tensor,
        fine_region_ids: torch.Tensor,
        *,
        global_grid_h: int,
        global_grid_w: int,
        evidence_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, GlobalAnchorRelayDiagnostics]:
        if global_tokens.ndim != 2 or global_tokens.shape[-1] != self.hidden_size:
            raise ValueError("global_tokens must have shape [G, hidden_size]")
        if int(global_tokens.shape[0]) != global_grid_h * global_grid_w:
            raise ValueError("global token count does not match grid")
        if fine_tokens.ndim != 2 or fine_tokens.shape[-1] != self.hidden_size:
            raise ValueError("fine_tokens must have shape [F, hidden_size]")
        fine_count = int(fine_tokens.shape[0])
        if fine_centers_xy.shape != (fine_count, 2):
            raise ValueError("fine_centers_xy must have shape [F, 2]")
        if fine_scales.shape != (fine_count,) or fine_region_ids.shape != (fine_count,):
            raise ValueError("fine scales/region ids do not match fine tokens")
        if fine_count == 0:
            diagnostics = GlobalAnchorRelayDiagnostics(
                global_tokens=int(global_tokens.shape[0]),
                fine_tokens=0,
                regions=0,
                anchors=self.anchor_grid[0] * self.anchor_grid[1],
                fine_relative_residual_l2=0.0,
                gate_mean=0.0,
                gate_std=0.0,
                attention_entropy_normalized=0.0,
                pre_projection_anchor_attention_mass=0.0,
                anchor_attention_mass=0.0,
                region_attention_mass=0.0,
                self_region_attention_mass=0.0,
                anchor_floor_activation_rate=0.0,
            )
            return global_tokens, fine_tokens, diagnostics
        if fine_region_ids.dtype != torch.long or int(fine_region_ids.min()) < 0:
            raise ValueError("fine_region_ids must be non-negative torch.long")
        regions = int(fine_region_ids.max().item()) + 1
        if set(fine_region_ids.detach().cpu().tolist()) != set(range(regions)):
            raise ValueError("fine_region_ids must be contiguous from zero")
        if evidence_weights is None:
            evidence_weights = fine_scales.new_ones((fine_count,))
        if evidence_weights.shape != (fine_count,):
            raise ValueError("evidence_weights must have shape [F]")
        weights = evidence_weights.clamp(0.0, 1.0).to(fine_tokens.dtype)

        global_normalized = self.global_norm(global_tokens)
        fine_normalized = self.fine_norm(fine_tokens)
        anchor_h, anchor_w = self.anchor_grid
        qwen_indexes, anchor_ids, _ = qwen_anchor_assignments(
            grid_h=global_grid_h,
            grid_w=global_grid_w,
            anchor_h=anchor_h,
            anchor_w=anchor_w,
            merge_size=self.spatial_merge_size,
            device=global_tokens.device,
        )
        spatial_global = global_normalized[qwen_indexes]
        anchor_count = anchor_h * anchor_w
        anchors = spatial_global.new_zeros((anchor_count, self.hidden_size))
        anchors.index_add_(0, anchor_ids, spatial_global)
        counts = spatial_global.new_zeros((anchor_count,))
        counts.index_add_(0, anchor_ids, torch.ones_like(anchor_ids, dtype=counts.dtype))
        anchors = anchors / counts.clamp_min(1.0)[:, None]
        anchor_rows, anchor_columns = torch.meshgrid(
            torch.arange(anchor_h, device=global_tokens.device),
            torch.arange(anchor_w, device=global_tokens.device),
            indexing="ij",
        )
        anchor_centers = torch.stack(
            [
                (anchor_columns.flatten().to(fine_centers_xy.dtype) + 0.5) / anchor_w,
                (anchor_rows.flatten().to(fine_centers_xy.dtype) + 0.5) / anchor_h,
            ],
            dim=-1,
        )
        anchor_scales = fine_scales.new_full(
            (anchor_count,),
            math.sqrt(1.0 / anchor_count),
        )

        region_summaries = self._weighted_means(
            fine_normalized,
            fine_region_ids,
            weights,
            regions,
        )
        region_centers = self._weighted_means(
            fine_centers_xy,
            fine_region_ids,
            weights,
            regions,
        )
        region_scales = self._weighted_means(
            fine_scales[:, None],
            fine_region_ids,
            weights,
            regions,
        ).squeeze(-1)

        nodes = torch.cat([anchors, region_summaries], dim=0)
        node_centers = torch.cat([anchor_centers, region_centers], dim=0)
        node_scales = torch.cat([anchor_scales, region_scales], dim=0)
        node_is_region = torch.cat(
            [
                node_scales.new_zeros((anchor_count,)),
                node_scales.new_ones((regions,)),
            ]
        )
        queries = self.region_query(region_summaries).reshape(
            regions, self.heads, self.head_dim
        )
        key, value = self.node_key_value(nodes).chunk(2, dim=-1)
        key = key.reshape(nodes.shape[0], self.heads, self.head_dim)
        value = value.reshape_as(key)
        logits = torch.einsum("rhd,nhd->rhn", queries, key) / math.sqrt(self.head_dim)
        delta_xy = node_centers[None, :, :] - region_centers[:, None, :]
        log_scale = torch.log2(
            node_scales[None, :].clamp_min(1e-6)
            / region_scales[:, None].clamp_min(1e-6)
        ).clamp(-8.0, 8.0)
        geometry = torch.cat(
            [
                delta_xy,
                log_scale[:, :, None],
                node_is_region[None, :, None].expand(regions, -1, -1),
            ],
            dim=-1,
        )
        bias = self.geometry_bias(
            geometry.to(self.geometry_bias[0].weight.dtype)
        ).permute(0, 2, 1)
        attention_float = torch.softmax(
            (logits + bias.to(logits.dtype)).float(),
            dim=-1,
        )
        pre_projection_anchor_mass = attention_float[:, :, :anchor_count].sum(
            dim=-1
        )
        floor_activated = (
            pre_projection_anchor_mass < self.minimum_anchor_attention_mass
        )
        if self.minimum_anchor_attention_mass > 0:
            pre_projection_region_mass = attention_float[
                :, :, anchor_count:
            ].sum(dim=-1)
            target_anchor_mass = pre_projection_anchor_mass.clamp_min(
                self.minimum_anchor_attention_mass
            )
            target_region_mass = 1.0 - target_anchor_mass
            anchor_attention = attention_float[:, :, :anchor_count] * (
                target_anchor_mass
                / pre_projection_anchor_mass.clamp_min(1e-12)
            )[:, :, None]
            region_attention = attention_float[:, :, anchor_count:] * (
                target_region_mass
                / pre_projection_region_mass.clamp_min(1e-12)
            )[:, :, None]
            attention_float = torch.cat(
                [anchor_attention, region_attention],
                dim=-1,
            )
        attention = attention_float.to(logits.dtype)
        region_context = torch.einsum("rhn,nhd->rhd", attention, value).reshape(
            regions, self.relay_dim
        )
        token_context = region_context[fine_region_ids]
        gates = torch.sigmoid(
            self.fine_gate(
                torch.cat([fine_normalized, token_context.to(fine_normalized.dtype)], dim=-1)
            )
        )
        raw_delta = self.fine_output(token_context) * gates * weights[:, None]
        fine_delta = self._bounded_delta(fine_tokens, raw_delta)
        fine_output = fine_tokens + fine_delta
        attention_float = attention_float.detach().clamp_min(1e-12)
        node_count = int(attention_float.shape[-1])
        attention_entropy = -(
            attention_float * attention_float.log()
        ).sum(dim=-1)
        entropy_denominator = math.log(node_count) if node_count > 1 else 1.0
        anchor_attention_mass = attention_float[:, :, :anchor_count].sum(dim=-1)
        region_attention_mass = attention_float[:, :, anchor_count:].sum(dim=-1)
        query_ids = torch.arange(regions, device=attention.device)
        self_region_attention_mass = attention_float[
            query_ids[:, None],
            torch.arange(self.heads, device=attention.device)[None, :],
            anchor_count + query_ids[:, None],
        ]
        residual = float(
            (
                fine_delta.float().pow(2).mean()
                / fine_tokens.float().pow(2).mean().clamp_min(1e-6)
            ).detach().cpu()
        )
        diagnostics = GlobalAnchorRelayDiagnostics(
            global_tokens=int(global_tokens.shape[0]),
            fine_tokens=fine_count,
            regions=regions,
            anchors=anchor_count,
            fine_relative_residual_l2=residual,
            gate_mean=float(gates.detach().float().mean().cpu()),
            gate_std=float(gates.detach().float().std(unbiased=False).cpu()),
            attention_entropy_normalized=float(
                (attention_entropy / entropy_denominator).mean().cpu()
            ),
            pre_projection_anchor_attention_mass=float(
                pre_projection_anchor_mass.detach().mean().cpu()
            ),
            anchor_attention_mass=float(anchor_attention_mass.mean().cpu()),
            region_attention_mass=float(region_attention_mass.mean().cpu()),
            self_region_attention_mass=float(
                self_region_attention_mass.mean().cpu()
            ),
            anchor_floor_activation_rate=float(
                floor_activated.detach().float().mean().cpu()
            ),
        )
        return global_tokens, fine_output, diagnostics
