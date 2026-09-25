"""Context-aware, group-relative box refinement for EviViT-v7.

The policy keeps the verified H-Safe boxes as its zero-action anchor.  It sees
both the existing problem-conditioned PTEA feature and a small 3x3 preview of
the *expanded high-resolution neighbourhood* around each anchor.  The latter
is the crucial difference from the failed v7 static selector: the policy can
inspect content immediately outside a box before deciding which way to move.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from evivit_core.evivit_adaptive_box import residual_target
from evivit_core.evivit_box_counterfactual import BoxPortfolio, counterfactual_portfolios


@dataclass(frozen=True)
class TraceRefineConfig:
    base_feature_dim: int
    preview_feature_dim: int
    hidden_dim: int = 192
    preview_dim: int = 128
    heads: int = 4
    maximum_center_factor: float = 0.50
    maximum_absolute_log_scale: float = 0.40
    minimum_log_std: float = -3.0
    maximum_log_std: float = -0.25

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class TraceRefinePolicy(nn.Module):
    """One shared bounded Gaussian policy for all evidence-region roles."""

    def __init__(self, config: TraceRefineConfig) -> None:
        super().__init__()
        if config.preview_dim % config.heads:
            raise ValueError("preview_dim must be divisible by heads")
        self.config = config
        self.base = nn.Sequential(
            nn.LayerNorm(config.base_feature_dim),
            nn.Linear(config.base_feature_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.preview = nn.Sequential(
            nn.LayerNorm(config.preview_feature_dim),
            nn.Linear(config.preview_feature_dim, config.preview_dim),
            nn.GELU(),
        )
        self.query = nn.Linear(config.hidden_dim, config.preview_dim)
        self.position = nn.Parameter(torch.zeros(9, config.preview_dim))
        self.cross_attention = nn.MultiheadAttention(
            config.preview_dim, config.heads, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.hidden_dim + config.preview_dim),
            nn.Linear(config.hidden_dim + config.preview_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 8),
        )
        # Zero mean exactly reproduces H-Safe before training.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def encode(
        self, base_features: torch.Tensor, preview_cells: torch.Tensor
    ) -> torch.Tensor:
        if base_features.shape[:-1] != preview_cells.shape[:-2]:
            raise ValueError("base and preview leading dimensions disagree")
        if preview_cells.shape[-2] != 9:
            raise ValueError("preview must contain a 3x3 grid (nine cells)")
        leading = base_features.shape[:-1]
        base = self.base(base_features.float())
        preview = self.preview(preview_cells.float()) + self.position
        query = self.query(base).reshape(-1, 1, self.config.preview_dim)
        key_value = preview.reshape(-1, 9, self.config.preview_dim)
        context, _ = self.cross_attention(
            query, key_value, key_value, need_weights=False
        )
        context = context.reshape(*leading, self.config.preview_dim)
        return torch.cat((base, context), dim=-1)

    def forward(
        self, base_features: torch.Tensor, preview_cells: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.encode(base_features, preview_cells)
        raw = self.output(fused)
        mean_raw, std_raw = raw.chunk(2, dim=-1)
        centre = self.config.maximum_center_factor * torch.tanh(mean_raw[..., :2])
        scale = self.config.maximum_absolute_log_scale * torch.tanh(mean_raw[..., 2:])
        mean = torch.cat((centre, scale), dim=-1)
        log_std = self.config.minimum_log_std + (
            self.config.maximum_log_std - self.config.minimum_log_std
        ) * torch.sigmoid(std_raw)
        return mean, log_std


@dataclass(frozen=True)
class ActStayGateConfig:
    """Small confidence gate applied after a TraceRefine proposal.

    The gate deliberately consumes only deployment-available diagnostics.  In
    particular it never receives final boxes, trace coverage, answer labels or
    candidate NLLs at inference time.
    """

    input_dim: int
    hidden_dim: int = 64

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class ActStayGate(nn.Module):
    def __init__(self, config: ActStayGateConfig) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features.float()).squeeze(-1)


def act_stay_gate_features(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    actions: torch.Tensor,
    logits: torch.Tensor,
    chosen_index: int,
    *,
    maximum_regions: int = 3,
) -> torch.Tensor:
    """Build stable, non-oracle diagnostics for the Act-or-Stay gate.

    ``mean``/``log_std`` are [R,4], ``actions`` is [C,R,4], and ``logits`` is
    [C].  Region actions are zero padded so the feature dimension is identical
    for two- and three-region samples.
    """

    if mean.ndim != 2 or mean.shape[-1] != 4:
        raise ValueError("mean must be [regions,4]")
    if actions.ndim != 3 or actions.shape[1:] != mean.shape:
        raise ValueError("actions must be [candidates,regions,4]")
    if len(mean) > maximum_regions:
        raise ValueError("region count exceeds gate padding")
    chosen = actions[chosen_index]
    padded = mean.new_zeros(maximum_regions, 4)
    padded[: len(chosen)] = chosen
    probabilities = torch.softmax(logits, dim=0)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum()
    entropy = entropy / max(1e-8, float(torch.log(mean.new_tensor(len(logits)))))
    identity_logit = logits[0]
    scalar = mean.new_tensor(
        [
            float(len(mean)) / maximum_regions,
            float(len(logits)) / 24.0,
            float(chosen_index) / max(1, len(logits) - 1),
        ]
    )
    return torch.cat(
        (
            mean.mean(dim=0),
            mean.abs().mean(dim=0),
            log_std.mean(dim=0),
            log_std.std(dim=0, unbiased=False),
            padded.flatten(),
            chosen.abs().mean().reshape(1),
            chosen.abs().max().reshape(1),
            (logits[chosen_index] - identity_logit).reshape(1),
            probabilities[0].reshape(1),
            probabilities[chosen_index].reshape(1),
            entropy.reshape(1),
            scalar,
        )
    )


def load_act_stay_gate(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[ActStayGate, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("version") != "evivit_v7_act_stay_gate_v1":
        raise ValueError(f"unsupported Act-or-Stay checkpoint: {checkpoint.get('version')}")
    gate = ActStayGate(ActStayGateConfig(**checkpoint["config"]))
    gate.load_state_dict(checkpoint["state_dict"])
    gate.to(device).eval().requires_grad_(False)
    return gate, checkpoint


def gaussian_action_logits(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Return unnormalised log probabilities for candidate joint actions.

    ``mean`` is [B,R,4], ``actions`` is [B,C,R,4], and output is [B,C].
    Constants independent of the candidate are omitted because the training
    loss normalises over the candidate group.
    """

    if actions.ndim != mean.ndim + 1:
        raise ValueError("actions must insert one candidate dimension")
    z = (actions.float() - mean[:, None]) / log_std.exp()[:, None]
    return (-0.5 * z.square() - log_std[:, None]).sum(dim=(-1, -2))


def load_trace_refine_policy(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[TraceRefinePolicy, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("version") != "evivit_v7_trace_refine_policy_v1":
        raise ValueError(
            f"unsupported TraceRefine checkpoint: {checkpoint.get('version')}"
        )
    policy = TraceRefinePolicy(TraceRefineConfig(**checkpoint["config"]))
    policy.load_state_dict(checkpoint["state_dict"])
    policy.to(device).eval().requires_grad_(False)
    return policy, checkpoint


def select_counterfactual_portfolio(
    policy: TraceRefinePolicy,
    base_features: torch.Tensor,
    preview_cells: torch.Tensor,
    anchor_boxes: Sequence[Sequence[int | float]],
) -> tuple[BoxPortfolio, dict[str, Any]]:
    """Select one audited local portfolio using the trained Gaussian policy.

    The OOF diagnostic was deliberately evaluated on the same deterministic
    24-action portfolio used to collect matched-prompt NLL rewards.  Test-time
    selection therefore projects the continuous Gaussian onto that portfolio
    instead of applying an unvalidated arbitrary mean action.
    """

    portfolios = counterfactual_portfolios(anchor_boxes, profile="full")
    device = next(policy.parameters()).device
    anchors = torch.tensor(anchor_boxes, dtype=torch.float32, device=device)
    if float(anchors.max()) > 1.5:
        anchors = anchors / 1000.0
    candidate_boxes = torch.tensor(
        [portfolio.boxes for portfolio in portfolios],
        dtype=torch.float32,
        device=device,
    ) / 1000.0
    actions = torch.stack(
        [residual_target(anchors, boxes) for boxes in candidate_boxes]
    )
    # The main Qwen path may be locked to deterministic Flash SDPA.  This tiny
    # policy deliberately runs in float32 and has a one-token query, for which
    # Flash/CuDNN do not provide a kernel.  Math SDPA is deterministic here, so
    # enable it only around the policy call and restore the backbone settings.
    use_cuda_math = torch.device(device).type == "cuda"
    previous_sdp = None
    if use_cuda_math:
        previous_sdp = (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    try:
        with torch.inference_mode():
            mean, log_std = policy(
                base_features.to(device)[None], preview_cells.to(device)[None]
            )
            logits = gaussian_action_logits(mean, log_std, actions[None])[0]
            chosen_index = int(logits.argmax())
    finally:
        if previous_sdp is not None:
            flash, memory_efficient, math = previous_sdp
            torch.backends.cuda.enable_flash_sdp(flash)
            torch.backends.cuda.enable_mem_efficient_sdp(memory_efficient)
            torch.backends.cuda.enable_math_sdp(math)
    chosen = portfolios[chosen_index]
    return chosen, {
        "candidate_count": len(portfolios),
        "chosen_index": chosen_index,
        "chosen_name": chosen.name,
        "chosen_operation": chosen.operation,
        "mean_action": mean[0].detach().float().cpu().tolist(),
        "log_std": log_std[0].detach().float().cpu().tolist(),
        "chosen_logit": float(logits[chosen_index].detach().cpu()),
        "identity_logit": float(logits[0].detach().cpu()),
    }


__all__ = [
    "ActStayGate",
    "ActStayGateConfig",
    "TraceRefineConfig",
    "TraceRefinePolicy",
    "act_stay_gate_features",
    "gaussian_action_logits",
    "load_act_stay_gate",
    "load_trace_refine_policy",
    "select_counterfactual_portfolio",
]
