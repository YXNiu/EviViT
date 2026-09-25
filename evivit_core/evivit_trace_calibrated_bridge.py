"""Human-trace calibration of an already trained EviViT-v4 sparse bridge.

This module deliberately does not invent another communication topology.  It
keeps the frozen v4 sparse-bridge residual as the trusted base and learns only
whether that residual should be strengthened or weakened at each token:

    v4 = native + bridge_delta
    calibrated = v4 + calibration * bridge_delta

The two stream-level calibration amplitudes are zero initialized, so an
untrained calibrator is exactly v4.  Token-wise importance and stream-level
amplitude are separated to prevent a shift-invariant trace loss from driving
all gates uniformly open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from evivit_core.tracefovea import normalized_trace_importance


@dataclass(frozen=True)
class TraceCalibratedBridgeDiagnostics:
    global_tokens: int
    fine_tokens: int
    global_amplitude: float
    fine_amplitude: float
    global_importance_std: float
    fine_importance_std: float
    calibration_absolute_mean: float
    calibration_maximum_absolute: float
    bridge_delta_relative_l2: float
    calibration_relative_l2: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_tokens": self.global_tokens,
            "fine_tokens": self.fine_tokens,
            "global_amplitude": self.global_amplitude,
            "fine_amplitude": self.fine_amplitude,
            "global_importance_std": self.global_importance_std,
            "fine_importance_std": self.fine_importance_std,
            "calibration_absolute_mean": self.calibration_absolute_mean,
            "calibration_maximum_absolute": self.calibration_maximum_absolute,
            "bridge_delta_relative_l2": self.bridge_delta_relative_l2,
            "calibration_relative_l2": self.calibration_relative_l2,
        }


class TraceCalibratedBridgeResidual(nn.Module):
    """Safely reweight a frozen v4 bridge without changing its topology."""

    def __init__(
        self,
        hidden_size: int,
        *,
        calibration_dim: int = 256,
        maximum_calibration: float = 0.50,
        minimum_importance: float = 0.25,
        maximum_importance: float = 3.0,
        amplitude_mode: str = "signed_zero",
        preserve_float32_importance: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or calibration_dim <= 0:
            raise ValueError("hidden_size and calibration_dim must be positive")
        if not 0 < maximum_calibration <= 1:
            raise ValueError("maximum_calibration must be in (0, 1]")
        if not 0 < minimum_importance <= 1:
            raise ValueError("minimum_importance must be in (0, 1]")
        if maximum_importance < 1:
            raise ValueError("maximum_importance must be at least 1")
        if amplitude_mode not in {"signed_zero", "positive_centered"}:
            raise ValueError(
                "amplitude_mode must be signed_zero or positive_centered"
            )
        self.hidden_size = int(hidden_size)
        self.calibration_dim = int(calibration_dim)
        self.maximum_calibration = float(maximum_calibration)
        self.minimum_importance = float(minimum_importance)
        self.maximum_importance = float(maximum_importance)
        self.amplitude_mode = str(amplitude_mode)
        self.preserve_float32_importance = bool(
            preserve_float32_importance
        )

        self.native_norm = nn.LayerNorm(hidden_size)
        self.delta_norm = nn.LayerNorm(hidden_size)
        self.native_projection = nn.Linear(
            hidden_size, calibration_dim, bias=False
        )
        self.delta_projection = nn.Linear(
            hidden_size, calibration_dim, bias=False
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(2, calibration_dim),
            nn.SiLU(),
        )
        self.importance_head = nn.Sequential(
            nn.Linear(3 * calibration_dim, calibration_dim),
            nn.SiLU(),
            nn.Linear(calibration_dim, 1),
        )
        nn.init.zeros_(self.importance_head[-1].weight)
        nn.init.zeros_(self.importance_head[-1].bias)
        self.global_amplitude_logit = nn.Parameter(torch.zeros(()))
        self.fine_amplitude_logit = nn.Parameter(torch.zeros(()))

        self.last_importance_logits: torch.Tensor | None = None
        self.last_global_importance_logits: torch.Tensor | None = None
        self.last_fine_importance_logits: torch.Tensor | None = None
        self.last_calibration_relative_l2: torch.Tensor | None = None

    @staticmethod
    def _relative_l2(
        reference: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        return (
            delta.float().pow(2).mean()
            / reference.float().pow(2).mean().clamp_min(1e-6)
        )

    def forward(
        self,
        native_tokens: torch.Tensor,
        bridged_tokens: torch.Tensor,
        evidence_prior: torch.Tensor,
        level_ids: torch.Tensor,
        *,
        global_tokens: int,
    ) -> tuple[torch.Tensor, TraceCalibratedBridgeDiagnostics]:
        if native_tokens.ndim != 2 or native_tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                "native_tokens must have shape [tokens, hidden_size]"
            )
        if bridged_tokens.shape != native_tokens.shape:
            raise ValueError("bridged_tokens must match native_tokens")
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

        bridge_delta = bridged_tokens - native_tokens
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
                self.delta_projection(self.delta_norm(bridge_delta)),
                self.condition_projection(condition),
            ],
            dim=-1,
        )
        logits = self.importance_head(features).squeeze(-1)
        global_logits = logits[:global_tokens]
        fine_logits = logits[global_tokens:]
        global_importance = normalized_trace_importance(
            global_logits,
            minimum=self.minimum_importance,
            maximum=self.maximum_importance,
            return_float32=self.preserve_float32_importance,
        )
        fine_importance = normalized_trace_importance(
            fine_logits,
            minimum=self.minimum_importance,
            maximum=self.maximum_importance,
            return_float32=self.preserve_float32_importance,
        )
        if self.amplitude_mode == "positive_centered":
            # A non-zero amplitude is still an exact v4 identity at
            # initialization because uniform importance is exactly one and
            # the correction below is centered as (importance - 1).  This
            # avoids the double-zero cold start of simultaneously learning
            # both a non-uniform distribution and a non-zero amplitude.
            global_amplitude = self.maximum_calibration * torch.sigmoid(
                self.global_amplitude_logit.float()
            )
            fine_amplitude = self.maximum_calibration * torch.sigmoid(
                self.fine_amplitude_logit.float()
            )
        else:
            global_amplitude = self.maximum_calibration * torch.tanh(
                self.global_amplitude_logit.float()
            )
            fine_amplitude = self.maximum_calibration * torch.tanh(
                self.fine_amplitude_logit.float()
            )
        calibration = torch.cat(
            [
                global_amplitude
                * (global_importance.float() - 1.0)
                / self.maximum_importance,
                fine_amplitude
                * (fine_importance.float() - 1.0)
                / self.maximum_importance,
            ],
            dim=0,
        )
        correction = calibration.to(bridge_delta.dtype)[:, None] * bridge_delta
        output = bridged_tokens + correction

        self.last_importance_logits = logits
        self.last_global_importance_logits = global_logits
        self.last_fine_importance_logits = fine_logits
        self.last_calibration_relative_l2 = self._relative_l2(
            bridged_tokens, correction
        )
        detached = calibration.detach().float()
        diagnostics = TraceCalibratedBridgeDiagnostics(
            global_tokens=int(global_tokens),
            fine_tokens=tokens - int(global_tokens),
            global_amplitude=float(global_amplitude.detach().cpu()),
            fine_amplitude=float(fine_amplitude.detach().cpu()),
            global_importance_std=float(
                global_importance.detach().float().std(unbiased=False).cpu()
            ),
            fine_importance_std=float(
                fine_importance.detach().float().std(unbiased=False).cpu()
            ),
            calibration_absolute_mean=float(detached.abs().mean().cpu()),
            calibration_maximum_absolute=float(detached.abs().max().cpu()),
            bridge_delta_relative_l2=float(
                self._relative_l2(native_tokens, bridge_delta).detach().cpu()
            ),
            calibration_relative_l2=float(
                self.last_calibration_relative_l2.detach().cpu()
            ),
        )
        return output, diagnostics


__all__ = [
    "TraceCalibratedBridgeDiagnostics",
    "TraceCalibratedBridgeResidual",
]
