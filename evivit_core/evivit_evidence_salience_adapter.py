"""Question-conditioned evidence salience residual for EviViT global tokens.

The adapter is deliberately position-wise: it never drops, reorders, pools, or
creates a global token.  A frozen PTEA probability map only scales a small
zero-initialized bottleneck residual at the corresponding native Qwen merge
group.  Consequently the module is an exact identity before training while
retaining the original two-dimensional topology and MRoPE metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from evivit_core.tracefovea import normalized_trace_importance


@dataclass(frozen=True)
class EvidenceSalienceDiagnostics:
    """Auditable statistics for one global-token adapter call."""

    global_tokens: int
    map_height: int
    map_width: int
    gate: float
    active_token_rate: float
    relative_residual_l2: float
    importance_std: float | None = None
    importance_maximum: float | None = None


class EvidenceSalienceGlobalAdapter(nn.Module):
    """Inject a PTEA map into native global tokens without changing topology."""

    def __init__(
        self,
        hidden_size: int,
        *,
        adapter_dim: int = 256,
        spatial_merge_size: int = 2,
        max_relative_residual: float = 0.10,
        trace_refinement: bool = False,
        minimum_importance: float = 0.25,
        maximum_importance: float = 3.0,
        preserve_float32_importance: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or adapter_dim <= 0:
            raise ValueError("hidden_size and adapter_dim must be positive")
        if spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        if not 0 < max_relative_residual <= 1:
            raise ValueError("max_relative_residual must be in (0, 1]")
        if not 0 < minimum_importance <= 1:
            raise ValueError("minimum_importance must be in (0, 1]")
        if maximum_importance < 1 or maximum_importance < minimum_importance:
            raise ValueError("maximum_importance must be at least 1")
        self.hidden_size = int(hidden_size)
        self.adapter_dim = int(adapter_dim)
        self.spatial_merge_size = int(spatial_merge_size)
        self.max_relative_residual = float(max_relative_residual)
        self.trace_refinement = bool(trace_refinement)
        self.minimum_importance = float(minimum_importance)
        self.maximum_importance = float(maximum_importance)
        self.preserve_float32_importance = bool(
            preserve_float32_importance
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, adapter_dim, bias=False)
        self.up = nn.Linear(adapter_dim, hidden_size, bias=False)
        self.activation = nn.SiLU()
        self.gate_logit = nn.Parameter(torch.zeros(()))
        self.importance_head = (
            nn.Linear(adapter_dim + 1, 1)
            if self.trace_refinement
            else None
        )
        if self.importance_head is not None:
            nn.init.zeros_(self.importance_head.weight)
            nn.init.zeros_(self.importance_head.bias)
        nn.init.zeros_(self.up.weight)
        self.last_importance_logits: torch.Tensor | None = None

    def _token_salience(
        self,
        probability_map: torch.Tensor,
        *,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        if probability_map.ndim != 2:
            raise ValueError("probability_map must have shape [H, W]")
        merge = self.spatial_merge_size
        if grid_h % merge or grid_w % merge:
            raise ValueError("global grid must be divisible by spatial_merge_size")
        if tuple(probability_map.shape) == (grid_h, grid_w):
            salience = probability_map
        elif tuple(probability_map.shape) == (grid_h // merge, grid_w // merge):
            # Qwen pre-merger order groups the merge×merge native patches of
            # each post-merger cell contiguously.
            salience = probability_map.flatten().repeat_interleave(merge * merge)
            salience = salience.reshape(grid_h, grid_w)
        else:
            raise ValueError(
                "probability_map must match either the pre- or post-merger grid"
            )
        salience = salience.float().clamp_min(0)
        peak = salience.max().clamp_min(1e-12)
        return (salience / peak).flatten()

    def forward(
        self,
        global_tokens: torch.Tensor,
        probability_map: torch.Tensor,
        *,
        grid_h: int,
        grid_w: int,
    ) -> tuple[torch.Tensor, EvidenceSalienceDiagnostics]:
        if global_tokens.ndim != 2 or global_tokens.shape[-1] != self.hidden_size:
            raise ValueError("global_tokens must have shape [G, hidden_size]")
        if int(global_tokens.shape[0]) != grid_h * grid_w:
            raise ValueError("global token count does not match grid_h × grid_w")
        salience = self._token_salience(
            probability_map, grid_h=grid_h, grid_w=grid_w
        ).to(device=global_tokens.device, dtype=global_tokens.dtype)
        gate = torch.sigmoid(self.gate_logit)
        bottleneck = self.activation(self.down(self.norm(global_tokens)))
        raw_delta = self.up(bottleneck)
        importance = None
        if self.importance_head is not None:
            prior = salience.float().clamp_min(1e-6)
            prior = prior / prior.sum().clamp_min(1e-12)
            refinement = self.importance_head(
                torch.cat(
                    [bottleneck, salience[:, None].to(bottleneck.dtype)],
                    dim=-1,
                )
            ).squeeze(-1)
            logits = prior.log().to(refinement.dtype) + refinement
            importance = normalized_trace_importance(
                logits,
                minimum=self.minimum_importance,
                maximum=self.maximum_importance,
                return_float32=self.preserve_float32_importance,
            )
            modulation = importance
            self.last_importance_logits = logits
        else:
            modulation = salience
            self.last_importance_logits = None
        if importance is not None and self.preserve_float32_importance:
            # Do the evidence reweighting in FP32.  Casting gains close to one
            # to BF16 first creates a dead zone in which early non-uniform
            # trace predictions become exactly uniform.
            raw_delta = (
                raw_delta.float()
                * modulation.float()[:, None]
                * gate.float()
            )
        else:
            raw_delta = (
                raw_delta
                * modulation[:, None].to(raw_delta.dtype)
                * gate
            )

        reference_rms = (
            global_tokens.float()
            .pow(2)
            .mean(dim=-1, keepdim=True)
            .sqrt()
            .clamp_min(1e-6)
        )
        maximum = self.max_relative_residual * reference_rms
        bounded_delta = (
            maximum * torch.tanh(raw_delta.float() / maximum)
        ).to(global_tokens.dtype)
        output = global_tokens + bounded_delta
        relative = (
            bounded_delta.float().pow(2).mean()
            / global_tokens.float().pow(2).mean().clamp_min(1e-6)
        )
        diagnostics = EvidenceSalienceDiagnostics(
            global_tokens=int(global_tokens.shape[0]),
            map_height=int(probability_map.shape[0]),
            map_width=int(probability_map.shape[1]),
            gate=float(gate.detach().cpu()),
            active_token_rate=float((salience > 0).float().mean().detach().cpu()),
            relative_residual_l2=float(relative.detach().cpu()),
            importance_std=(
                float(importance.detach().float().std(unbiased=False).cpu())
                if importance is not None
                else None
            ),
            importance_maximum=(
                float(importance.detach().float().max().cpu())
                if importance is not None
                else None
            ),
        )
        return output, diagnostics
