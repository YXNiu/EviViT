"""Low-rank global/local communication through shared evidence relay tokens.

EviRelay keeps exactly the same global and fine token sets as EviViT-v4 and
TraceFovea.  A small fixed number of latent relays first read both streams,
then broadcast one bounded residual back to every token.  Relay tokens are
internal workspace only: they are not appended to the visual span and never
change the token budget.  Zero-initialized output projections make the module
an exact identity before training.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import torch
from torch import nn

from evivit_core.evivit_evidence_bridge import EvidenceBridgeDiagnostics


@dataclass(frozen=True)
class EviRelayDiagnostics:
    global_tokens: int
    fine_tokens: int
    relay_tokens: int
    gather_entropy_mean: float
    global_relative_residual_l2: float
    fine_relative_residual_l2: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_tokens": self.global_tokens,
            "fine_tokens": self.fine_tokens,
            "relay_tokens": self.relay_tokens,
            "gather_entropy_mean": self.gather_entropy_mean,
            "global_relative_residual_l2": self.global_relative_residual_l2,
            "fine_relative_residual_l2": self.fine_relative_residual_l2,
        }


class EviRelayBlock(nn.Module):
    """Two-stage gather/broadcast fusion with a fixed low-rank relay set."""

    def __init__(
        self,
        hidden_size: int,
        *,
        relay_dim: int = 256,
        relay_tokens: int = 8,
        max_relative_residual: float = 0.20,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or relay_dim <= 0 or relay_tokens <= 0:
            raise ValueError("hidden_size, relay_dim and relay_tokens must be positive")
        if not 0 < max_relative_residual <= 1:
            raise ValueError("max_relative_residual must be in (0, 1]")
        self.hidden_size = int(hidden_size)
        self.relay_dim = int(relay_dim)
        self.relay_tokens = int(relay_tokens)
        self.max_relative_residual = float(max_relative_residual)

        self.global_norm = nn.LayerNorm(hidden_size)
        self.fine_norm = nn.LayerNorm(hidden_size)
        self.global_key = nn.Linear(hidden_size, relay_dim, bias=False)
        self.global_value = nn.Linear(hidden_size, relay_dim, bias=False)
        self.fine_key = nn.Linear(hidden_size, relay_dim, bias=False)
        self.fine_value = nn.Linear(hidden_size, relay_dim, bias=False)
        self.global_query = nn.Linear(hidden_size, relay_dim, bias=False)
        self.fine_query = nn.Linear(hidden_size, relay_dim, bias=False)
        self.relay_query = nn.Parameter(
            torch.randn(relay_tokens, relay_dim) / sqrt(relay_dim)
        )
        self.relay_key = nn.Linear(relay_dim, relay_dim, bias=False)
        self.relay_value = nn.Sequential(
            nn.LayerNorm(relay_dim),
            nn.Linear(relay_dim, relay_dim),
            nn.SiLU(),
        )
        self.global_output = nn.Linear(relay_dim, hidden_size, bias=False)
        self.fine_output = nn.Linear(relay_dim, hidden_size, bias=False)
        nn.init.zeros_(self.global_output.weight)
        nn.init.zeros_(self.fine_output.weight)

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
        global_tokens: torch.Tensor,
        fine_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, EviRelayDiagnostics]:
        if global_tokens.ndim != 2 or global_tokens.shape[-1] != self.hidden_size:
            raise ValueError("global_tokens must have shape [global, hidden_size]")
        if fine_tokens.ndim != 2 or fine_tokens.shape[-1] != self.hidden_size:
            raise ValueError("fine_tokens must have shape [fine, hidden_size]")
        global_count = int(global_tokens.shape[0])
        fine_count = int(fine_tokens.shape[0])
        if global_count == 0:
            raise ValueError("global token stream must not be empty")
        if fine_count == 0:
            diagnostics = EviRelayDiagnostics(
                global_tokens=global_count,
                fine_tokens=0,
                relay_tokens=self.relay_tokens,
                gather_entropy_mean=0.0,
                global_relative_residual_l2=0.0,
                fine_relative_residual_l2=0.0,
            )
            return global_tokens, fine_tokens, diagnostics

        global_normalized = self.global_norm(global_tokens)
        fine_normalized = self.fine_norm(fine_tokens)
        keys = torch.cat(
            [self.global_key(global_normalized), self.fine_key(fine_normalized)],
            dim=0,
        )
        values = torch.cat(
            [
                self.global_value(global_normalized),
                self.fine_value(fine_normalized),
            ],
            dim=0,
        )
        gather_logits = (
            self.relay_query.to(keys.dtype) @ keys.transpose(0, 1)
        ) / sqrt(self.relay_dim)
        gather_probability = gather_logits.float().softmax(dim=-1).to(values.dtype)
        relays = self.relay_value(gather_probability @ values)
        relay_keys = self.relay_key(relays)

        global_logits = (
            self.global_query(global_normalized) @ relay_keys.transpose(0, 1)
        ) / sqrt(self.relay_dim)
        fine_logits = (
            self.fine_query(fine_normalized) @ relay_keys.transpose(0, 1)
        ) / sqrt(self.relay_dim)
        global_message = global_logits.float().softmax(dim=-1).to(relays.dtype) @ relays
        fine_message = fine_logits.float().softmax(dim=-1).to(relays.dtype) @ relays
        global_delta = self._bounded_delta(
            global_tokens, self.global_output(global_message)
        )
        fine_delta = self._bounded_delta(fine_tokens, self.fine_output(fine_message))
        updated_global = global_tokens + global_delta
        updated_fine = fine_tokens + fine_delta
        global_relative = (
            global_delta.float().pow(2).mean()
            / global_tokens.float().pow(2).mean().clamp_min(1e-6)
        )
        fine_relative = (
            fine_delta.float().pow(2).mean()
            / fine_tokens.float().pow(2).mean().clamp_min(1e-6)
        )
        entropy = -(
            gather_probability.float().clamp_min(1e-12)
            * gather_probability.float().clamp_min(1e-12).log()
        ).sum(dim=-1)
        diagnostics = EviRelayDiagnostics(
            global_tokens=global_count,
            fine_tokens=fine_count,
            relay_tokens=self.relay_tokens,
            gather_entropy_mean=float(entropy.mean().detach().cpu()),
            global_relative_residual_l2=float(global_relative.detach().cpu()),
            fine_relative_residual_l2=float(fine_relative.detach().cpu()),
        )
        return updated_global, updated_fine, diagnostics


class EviRelayBridgeAdapter(nn.Module):
    """Drop-in bridge contract for the existing segmented Qwen-ViT runtime."""

    def __init__(self, block: EviRelayBlock) -> None:
        super().__init__()
        self.block = block
        self.hidden_size = block.hidden_size
        self.last_relay_diagnostics: EviRelayDiagnostics | None = None

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
        # Geometry and PTEA are unchanged upstream.  The relay comparator is
        # intentionally content-only so fusion is the sole changed variable.
        del fine_centers_xy, fine_scales, global_grid_h, global_grid_w
        del evidence_weights
        global_out, fine_out, diagnostics = self.block(global_tokens, fine_tokens)
        self.last_relay_diagnostics = diagnostics
        return global_out, fine_out, EvidenceBridgeDiagnostics(
            global_tokens=diagnostics.global_tokens,
            fine_tokens=diagnostics.fine_tokens,
            covered_global_tokens=diagnostics.global_tokens,
            mean_valid_global_neighbors=float(diagnostics.relay_tokens),
            global_relative_residual_l2=diagnostics.global_relative_residual_l2,
            fine_relative_residual_l2=diagnostics.fine_relative_residual_l2,
        )


__all__ = [
    "EviRelayBlock",
    "EviRelayBridgeAdapter",
    "EviRelayDiagnostics",
]
