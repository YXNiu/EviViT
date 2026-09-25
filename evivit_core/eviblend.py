"""Safe residual blending between native and joint frozen-Qwen vision paths.

The native path is the unchanged EviViT-v4 execution: every global/fine stream
runs one frozen Qwen-ViT block independently.  The counterfactual joint path
runs the same frozen block with a shared original-image coordinate frame.  This
module does not replace the trusted native output.  It learns only a bounded,
zero-initialized token-wise coefficient for the difference

    joint_output - native_output.

Consequently an untrained EviBlend module is exactly the native v4 topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from evivit_core.tracefovea import normalized_trace_importance


@dataclass(frozen=True)
class EviBlendDiagnostics:
    tokens: int
    global_tokens: int
    fine_tokens: int
    gate_mean: float
    gate_std: float
    gate_absolute_mean: float
    gate_maximum_absolute: float
    joint_delta_relative_l2: float
    injected_relative_l2: float
    global_amplitude: float
    fine_amplitude: float
    global_importance_std: float
    fine_importance_std: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "global_tokens": self.global_tokens,
            "fine_tokens": self.fine_tokens,
            "gate_mean": self.gate_mean,
            "gate_std": self.gate_std,
            "gate_absolute_mean": self.gate_absolute_mean,
            "gate_maximum_absolute": self.gate_maximum_absolute,
            "joint_delta_relative_l2": self.joint_delta_relative_l2,
            "injected_relative_l2": self.injected_relative_l2,
            "global_amplitude": self.global_amplitude,
            "fine_amplitude": self.fine_amplitude,
            "global_importance_std": self.global_importance_std,
            "fine_importance_std": self.fine_importance_std,
        }


class SafeJointResidualMixer(nn.Module):
    """Inject only the useful part of a frozen joint-attention counterfactual.

    The gate sees the trusted native feature, the native-to-joint difference,
    a question-conditioned evidence prior, and a global/fine stream flag.
    Its final projection is initialized to zero, so the first forward is an
    exact identity even though gradients can immediately train token-specific
    signed gates.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        blend_dim: int = 256,
        maximum_gate: float = 0.50,
        minimum_importance: float = 0.25,
        maximum_importance: float = 3.0,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or blend_dim <= 0:
            raise ValueError("hidden_size and blend_dim must be positive")
        if not 0 < maximum_gate <= 1:
            raise ValueError("maximum_gate must be in (0, 1]")
        self.hidden_size = int(hidden_size)
        self.blend_dim = int(blend_dim)
        self.maximum_gate = float(maximum_gate)
        self.minimum_importance = float(minimum_importance)
        self.maximum_importance = float(maximum_importance)
        if not 0 < self.minimum_importance <= 1.0:
            raise ValueError("minimum_importance must be in (0, 1]")
        if self.maximum_importance < 1.0:
            raise ValueError("maximum_importance must be at least 1")

        self.native_norm = nn.LayerNorm(hidden_size)
        self.delta_norm = nn.LayerNorm(hidden_size)
        self.native_projection = nn.Linear(hidden_size, blend_dim, bias=False)
        self.delta_projection = nn.Linear(hidden_size, blend_dim, bias=False)
        self.condition_projection = nn.Sequential(
            nn.Linear(2, blend_dim),
            nn.SiLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(3 * blend_dim, blend_dim),
            nn.SiLU(),
            nn.Linear(blend_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.global_amplitude_logit = nn.Parameter(torch.zeros(()))
        self.fine_amplitude_logit = nn.Parameter(torch.zeros(()))
        self.last_gate_logits: torch.Tensor | None = None
        self.last_fine_gate_logits: torch.Tensor | None = None
        self.last_injected_relative_l2: torch.Tensor | None = None

    @staticmethod
    def _relative_l2(reference: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return (
            delta.float().pow(2).mean()
            / reference.float().pow(2).mean().clamp_min(1e-6)
        )

    def forward(
        self,
        native_tokens: torch.Tensor,
        joint_tokens: torch.Tensor,
        evidence_prior: torch.Tensor,
        level_ids: torch.Tensor,
        *,
        global_tokens: int,
    ) -> tuple[torch.Tensor, EviBlendDiagnostics]:
        if native_tokens.ndim != 2 or native_tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                "native_tokens must have shape [tokens, hidden_size]"
            )
        if joint_tokens.shape != native_tokens.shape:
            raise ValueError("joint_tokens must match native_tokens")
        tokens = int(native_tokens.shape[0])
        if evidence_prior.shape != (tokens,):
            raise ValueError("evidence_prior must have shape [tokens]")
        if level_ids.shape != (tokens,):
            raise ValueError("level_ids must have shape [tokens]")
        if not 0 < global_tokens < tokens:
            raise ValueError(
                "global_tokens must leave non-empty global and fine streams"
            )
        if not torch.isfinite(evidence_prior).all():
            raise ValueError("evidence_prior must be finite")

        delta = joint_tokens - native_tokens
        condition = torch.stack(
            [
                evidence_prior.to(native_tokens.dtype).clamp(0.0, 1.0),
                level_ids.to(native_tokens.dtype).clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        features = torch.cat(
            [
                self.native_projection(self.native_norm(native_tokens)),
                self.delta_projection(self.delta_norm(delta)),
                self.condition_projection(condition),
            ],
            dim=-1,
        )
        gate_logits = self.gate(features).squeeze(-1)
        global_logits = gate_logits[:global_tokens]
        fine_logits = gate_logits[global_tokens:]
        global_importance = normalized_trace_importance(
            global_logits,
            minimum=self.minimum_importance,
            maximum=self.maximum_importance,
        )
        fine_importance = normalized_trace_importance(
            fine_logits,
            minimum=self.minimum_importance,
            maximum=self.maximum_importance,
        )
        global_amplitude = (
            self.maximum_gate
            * torch.tanh(self.global_amplitude_logit.float())
        )
        fine_amplitude = (
            self.maximum_gate
            * torch.tanh(self.fine_amplitude_logit.float())
        )
        gate = torch.cat(
            [
                global_amplitude
                * global_importance.float()
                / self.maximum_importance,
                fine_amplitude
                * fine_importance.float()
                / self.maximum_importance,
            ],
            dim=0,
        )
        injected = gate.to(delta.dtype)[:, None] * delta
        output = native_tokens + injected
        self.last_gate_logits = gate_logits
        self.last_fine_gate_logits = gate_logits[global_tokens:]
        self.last_injected_relative_l2 = self._relative_l2(
            native_tokens, injected
        )

        detached_gate = gate.detach().float()
        diagnostics = EviBlendDiagnostics(
            tokens=tokens,
            global_tokens=int(global_tokens),
            fine_tokens=tokens - int(global_tokens),
            gate_mean=float(detached_gate.mean().cpu()),
            gate_std=float(detached_gate.std(unbiased=False).cpu()),
            gate_absolute_mean=float(detached_gate.abs().mean().cpu()),
            gate_maximum_absolute=float(detached_gate.abs().max().cpu()),
            joint_delta_relative_l2=float(
                self._relative_l2(native_tokens, delta).detach().cpu()
            ),
            injected_relative_l2=float(
                self.last_injected_relative_l2.detach().cpu()
            ),
            global_amplitude=float(global_amplitude.detach().cpu()),
            fine_amplitude=float(fine_amplitude.detach().cpu()),
            global_importance_std=float(
                global_importance.detach().float().std(unbiased=False).cpu()
            ),
            fine_importance_std=float(
                fine_importance.detach().float().std(unbiased=False).cpu()
            ),
        )
        return output, diagnostics


__all__ = ["EviBlendDiagnostics", "SafeJointResidualMixer"]
