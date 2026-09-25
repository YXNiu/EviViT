"""Topology-aligned evidence slots for high-resolution visual rereading.

EviSlot compresses one high-resolution evidence region at an intermediate
Qwen3-VL vision block.  Each emitted LLM-visible slot is represented by one
native Qwen merger group (``spatial_merge_size ** 2`` sub-slots), so the
compressed state can continue through the frozen upper ViT, DeepStack and
merger without replacing any pretrained Qwen component.

The compressor is deliberately local to one evidence region.  Slot queries
receive three kinds of information:

* Fine-token content and original-image coordinates;
* the region's PTEA evidence mass and bounding-box geometry;
* the identity/content of the nearest Global parent token.

The returned attention is also used to pool rotary positions and any
DeepStack features captured before the insertion block.  This keeps every
visual branch length-aligned after compression.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class EviSlotAuxiliaryLosses:
    """Differentiable engineering/structure losses for one Fine region."""

    trace_support: torch.Tensor
    slot_diversity: torch.Tensor
    parent_consistency: torch.Tensor
    residual_identity: torch.Tensor


@dataclass
class EviSlotRegionOutput:
    """One compressed Fine region, ready to resume through the upper ViT."""

    hidden_states: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    deepstack_features_by_block: dict[int, torch.Tensor]
    slot_centers_xy: torch.Tensor
    slot_scales: torch.Tensor
    parent_indices: torch.Tensor
    attention: torch.Tensor
    auxiliary_losses: EviSlotAuxiliaryLosses
    attention_entropy: torch.Tensor
    mean_slot_attention_overlap: torch.Tensor
    residual_gate: torch.Tensor


@dataclass
class EviSlotDiagnostics:
    """Compact runtime diagnostics aggregated over all evidence regions."""

    input_fine_tokens: int
    slot_encoder_tokens: int
    emitted_slots: int
    region_count: int
    slots_per_region: int
    parent_indices: list[int]
    mean_attention_entropy: float
    mean_slot_attention_overlap: float
    mean_residual_gate: float
    spatial_anchor_bias_enabled: bool
    spatial_anchor_strength: float
    spatial_anchor_sigma: float
    trace_support_loss: float
    slot_diversity_loss: float
    parent_consistency_loss: float
    residual_identity_loss: float

    @property
    def compression_ratio(self) -> float:
        if self.emitted_slots == 0:
            return 0.0
        return float(self.input_fine_tokens) / float(self.emitted_slots)

    def as_dict(self) -> dict[str, object]:
        return {
            "input_fine_tokens": self.input_fine_tokens,
            "slot_encoder_tokens": self.slot_encoder_tokens,
            "emitted_slots": self.emitted_slots,
            "region_count": self.region_count,
            "slots_per_region": self.slots_per_region,
            "compression_ratio": self.compression_ratio,
            "parent_indices": self.parent_indices,
            "mean_attention_entropy": self.mean_attention_entropy,
            "mean_slot_attention_overlap": self.mean_slot_attention_overlap,
            "mean_residual_gate": self.mean_residual_gate,
            "spatial_anchor_bias_enabled": self.spatial_anchor_bias_enabled,
            "spatial_anchor_strength": self.spatial_anchor_strength,
            "spatial_anchor_sigma": self.spatial_anchor_sigma,
            "trace_support_loss": self.trace_support_loss,
            "slot_diversity_loss": self.slot_diversity_loss,
            "parent_consistency_loss": self.parent_consistency_loss,
            "residual_identity_loss": self.residual_identity_loss,
        }


def _sample_probability_map(
    probability_map: torch.Tensor | None,
    centers_xy: torch.Tensor,
) -> torch.Tensor | None:
    if probability_map is None:
        return None
    if probability_map.ndim != 2:
        raise ValueError("evidence_probability_map must have shape [height, width]")
    probability = probability_map.detach().float().clamp_min(0)
    probability = probability / probability.sum().clamp_min(1e-12)
    sampling_grid = (
        2.0 * centers_xy.detach().float().clamp(0.0, 1.0) - 1.0
    ).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        probability.reshape(1, 1, *probability.shape),
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).reshape(-1)
    return sampled / sampled.sum().clamp_min(1e-12)


class EvidenceSlotCompressor(nn.Module):
    """Compress Fine tokens into a fixed number of topology-aware slots.

    ``slots_per_region`` is the number of tokens visible to the Qwen LLM.  At
    the intermediate ViT boundary each slot expands to ``merge_group_size``
    learned sub-slots so the unchanged Qwen merger later maps it back to one
    token.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        slot_dim: int = 256,
        heads: int = 4,
        slots_per_region: int = 4,
        spatial_merge_size: int = 2,
        maximum_residual_scale: float = 0.25,
        evidence_bias: float = 0.5,
        parent_temperature: float = 0.07,
        enable_spatial_anchor_bias: bool = False,
        spatial_anchor_strength: float = 2.0,
        spatial_anchor_sigma: float = 0.30,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or slot_dim <= 0:
            raise ValueError("hidden_size and slot_dim must be positive")
        if heads <= 0 or slot_dim % heads:
            raise ValueError("slot_dim must be divisible by heads")
        if slots_per_region <= 0:
            raise ValueError("slots_per_region must be positive")
        if spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        if maximum_residual_scale < 0:
            raise ValueError("maximum_residual_scale must be non-negative")
        if parent_temperature <= 0:
            raise ValueError("parent_temperature must be positive")
        if spatial_anchor_strength < 0:
            raise ValueError("spatial_anchor_strength must be non-negative")
        if spatial_anchor_sigma <= 0:
            raise ValueError("spatial_anchor_sigma must be positive")
        if enable_spatial_anchor_bias and slots_per_region != 4:
            raise ValueError(
                "fixed 2x2 spatial anchors require exactly four slots per region"
            )

        self.hidden_size = int(hidden_size)
        self.slot_dim = int(slot_dim)
        self.heads = int(heads)
        self.slots_per_region = int(slots_per_region)
        self.spatial_merge_size = int(spatial_merge_size)
        self.merge_group_size = self.spatial_merge_size**2
        self.maximum_residual_scale = float(maximum_residual_scale)
        self.parent_temperature = float(parent_temperature)
        self.enable_spatial_anchor_bias = bool(enable_spatial_anchor_bias)
        self.spatial_anchor_strength = float(spatial_anchor_strength)
        self.spatial_anchor_sigma = float(spatial_anchor_sigma)

        self.slot_queries = nn.Parameter(
            torch.empty(
                self.slots_per_region,
                self.merge_group_size,
                self.slot_dim,
            )
        )
        self.fine_norm = nn.LayerNorm(self.hidden_size)
        self.parent_norm = nn.LayerNorm(self.hidden_size)
        self.key_projection = nn.Linear(self.hidden_size, self.slot_dim, bias=False)
        self.value_projection = nn.Linear(
            self.hidden_size, self.slot_dim, bias=False
        )
        self.parent_query_projection = nn.Linear(
            self.hidden_size, self.slot_dim, bias=False
        )
        self.token_geometry = nn.Sequential(
            nn.Linear(7, self.slot_dim),
            nn.SiLU(),
            nn.Linear(self.slot_dim, self.slot_dim),
        )
        self.region_geometry = nn.Sequential(
            nn.Linear(7, self.slot_dim),
            nn.SiLU(),
            nn.Linear(self.slot_dim, self.slot_dim),
        )
        self.output_projection = nn.Linear(
            self.slot_dim, self.hidden_size, bias=False
        )
        self.slot_parent_projection = nn.Linear(
            self.hidden_size, self.slot_dim, bias=False
        )
        self.global_parent_projection = nn.Linear(
            self.hidden_size, self.slot_dim, bias=False
        )
        # A bounded residual retains a content-only attention pool as the
        # stable anchor while still allowing learned value refinement.
        self.residual_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.evidence_bias_logit = nn.Parameter(
            torch.tensor(math.log(math.expm1(max(evidence_bias, 1e-4))))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.slot_queries, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_inputs(
        self,
        fine_hidden: torch.Tensor,
        fine_centers_xy: torch.Tensor,
        fine_scales: torch.Tensor,
        fine_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        global_parent_hidden: torch.Tensor,
        global_parent_centers_xy: torch.Tensor,
        bbox_xyxy: Sequence[float],
        deepstack_features_by_block: Mapping[int, torch.Tensor],
    ) -> None:
        if fine_hidden.ndim != 2 or fine_hidden.shape[-1] != self.hidden_size:
            raise ValueError("fine_hidden must have shape [tokens, hidden_size]")
        token_count = int(fine_hidden.shape[0])
        if token_count == 0 or token_count % self.merge_group_size:
            raise ValueError("Fine token count must be non-zero and merger-aligned")
        if fine_centers_xy.shape != (token_count, 2):
            raise ValueError("fine_centers_xy must match Fine token count")
        if fine_scales.shape != (token_count,):
            raise ValueError("fine_scales must match Fine token count")
        cosine, sine = fine_position_embeddings
        if cosine.ndim != 2 or sine.shape != cosine.shape:
            raise ValueError("position embeddings must be matching rank-2 tensors")
        if int(cosine.shape[0]) != token_count:
            raise ValueError("position embeddings must match Fine token count")
        if (
            global_parent_hidden.ndim != 2
            or global_parent_hidden.shape[-1] != self.hidden_size
            or int(global_parent_hidden.shape[0]) == 0
        ):
            raise ValueError("global_parent_hidden must contain Global parents")
        if global_parent_centers_xy.shape != (
            int(global_parent_hidden.shape[0]),
            2,
        ):
            raise ValueError("global parent centers must match Global parent count")
        if len(bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must have four values")
        for block, feature in deepstack_features_by_block.items():
            if feature.ndim != 2:
                raise ValueError(f"DeepStack block {block} feature must be rank 2")
            if int(feature.shape[0]) != token_count // self.merge_group_size:
                raise ValueError(
                    f"DeepStack block {block} is not aligned to Fine merger groups"
                )

    def forward(
        self,
        fine_hidden: torch.Tensor,
        fine_centers_xy: torch.Tensor,
        fine_scales: torch.Tensor,
        fine_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        global_parent_hidden: torch.Tensor,
        global_parent_centers_xy: torch.Tensor,
        bbox_xyxy: Sequence[float],
        evidence_weight: float,
        deepstack_features_by_block: Mapping[int, torch.Tensor] | None = None,
        evidence_probability_map: torch.Tensor | None = None,
    ) -> EviSlotRegionOutput:
        deepstack = dict(deepstack_features_by_block or {})
        self._validate_inputs(
            fine_hidden,
            fine_centers_xy,
            fine_scales,
            fine_position_embeddings,
            global_parent_hidden,
            global_parent_centers_xy,
            bbox_xyxy,
            deepstack,
        )
        x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
        if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
            raise ValueError("bbox_xyxy must be a valid normalized box")
        width = x2 - x1
        height = y2 - y1
        center = fine_hidden.new_tensor([(x1 + x2) / 2, (y1 + y2) / 2])
        parent_index = int(
            (global_parent_centers_xy.float() - center.float())
            .square()
            .sum(dim=-1)
            .argmin()
            .item()
        )

        normalized_fine = self.fine_norm(fine_hidden)
        keys = self.key_projection(normalized_fine)
        values = self.value_projection(normalized_fine)
        local_x = (fine_centers_xy[:, 0] - center[0]) / max(width, 1e-6)
        local_y = (fine_centers_xy[:, 1] - center[1]) / max(height, 1e-6)
        token_geometry = torch.stack(
            [
                fine_centers_xy[:, 0],
                fine_centers_xy[:, 1],
                local_x,
                local_y,
                fine_scales,
                fine_scales.clamp_min(1e-8).log(),
                fine_scales.new_full(
                    fine_scales.shape, max(0.0, float(evidence_weight))
                ),
            ],
            dim=-1,
        )
        keys = keys + self.token_geometry(token_geometry.to(keys.dtype))

        region_geometry = torch.cat(
            [
                center.to(fine_hidden.dtype),
                fine_hidden.new_tensor(
                    [
                        width,
                        height,
                        math.sqrt(width * height),
                        math.log(max(width * height, 1e-8)),
                        max(0.0, float(evidence_weight)),
                    ]
                ),
            ]
        )
        parent_query = self.parent_query_projection(
            self.parent_norm(global_parent_hidden[parent_index])
        )
        query = (
            self.slot_queries
            + self.region_geometry(region_geometry.to(keys.dtype))[None, None, :]
            + parent_query[None, None, :]
        )
        head_dim = self.slot_dim // self.heads
        queries = query.reshape(
            self.slots_per_region, self.merge_group_size, self.heads, head_dim
        )
        head_keys = keys.reshape(-1, self.heads, head_dim)
        logits = torch.einsum("sghd,nhd->sghn", queries, head_keys)
        logits = logits.float() / math.sqrt(head_dim)

        if self.enable_spatial_anchor_bias:
            # Fixed top-left, top-right, bottom-left and bottom-right anchors
            # break the permutation-symmetric uniform-attention stationary
            # point without adding a predictor or trainable parameters.
            anchors = logits.new_tensor(
                [
                    [-0.25, -0.25],
                    [0.25, -0.25],
                    [-0.25, 0.25],
                    [0.25, 0.25],
                ]
            )
            local_coordinates = torch.stack([local_x, local_y], dim=-1).float()
            squared_distance = (
                local_coordinates[None, :, :] - anchors[:, None, :]
            ).square().sum(dim=-1)
            anchor_bias = -(
                self.spatial_anchor_strength
                * squared_distance
                / (2.0 * self.spatial_anchor_sigma**2)
            )
            # Only relative logits matter. Centering each slot keeps its best
            # quadrant at zero and avoids needlessly large negative offsets.
            anchor_bias = anchor_bias - anchor_bias.max(dim=-1, keepdim=True).values
            logits = logits + anchor_bias[:, None, None, :]

        evidence_target = _sample_probability_map(
            evidence_probability_map, fine_centers_xy
        )
        if evidence_target is not None:
            evidence_log_prior = (
                evidence_target.clamp_min(1e-8)
                * float(evidence_target.numel())
            ).log()
            logits = logits + F.softplus(self.evidence_bias_logit.float()) * (
                evidence_log_prior[None, None, None, :]
            )
        head_attention = logits.softmax(dim=-1).to(values.dtype)
        head_values = values.reshape(-1, self.heads, head_dim)
        context = torch.einsum(
            "sghn,nhd->sghd", head_attention, head_values
        ).reshape(self.slots_per_region, self.merge_group_size, self.slot_dim)
        attention = head_attention.mean(dim=2)
        anchor = torch.einsum("sgn,nh->sgh", attention, fine_hidden)
        residual = self.output_projection(context)
        residual_gate = (
            self.maximum_residual_scale
            * torch.sigmoid(self.residual_gate_logit.float())
        ).to(residual.dtype)
        slot_hidden = anchor + residual_gate * residual

        cosine, sine = fine_position_embeddings
        pooled_cosine = torch.einsum(
            "sgn,nd->sgd", attention.float(), cosine.float()
        )
        pooled_sine = torch.einsum(
            "sgn,nd->sgd", attention.float(), sine.float()
        )
        unit_norm = torch.sqrt(
            pooled_cosine.square() + pooled_sine.square()
        ).clamp_min(1e-6)
        pooled_positions = (
            (pooled_cosine / unit_norm).to(cosine.dtype).reshape(
                -1, cosine.shape[-1]
            ),
            (pooled_sine / unit_norm).to(sine.dtype).reshape(
                -1, sine.shape[-1]
            ),
        )

        slot_attention = attention.float().mean(dim=1)
        slot_centers = slot_attention @ fine_centers_xy.float()
        slot_scales = slot_attention @ fine_scales.float()
        grouped_attention = slot_attention.reshape(
            self.slots_per_region,
            -1,
            self.merge_group_size,
        ).sum(dim=-1)
        compressed_deepstack = {
            int(block): (
                grouped_attention.to(feature.dtype) @ feature
            )
            for block, feature in deepstack.items()
        }

        if self.slots_per_region > 1:
            normalized_attention = F.normalize(slot_attention, dim=-1)
            similarity = normalized_attention @ normalized_attention.transpose(0, 1)
            off_diagonal = ~torch.eye(
                self.slots_per_region,
                device=similarity.device,
                dtype=torch.bool,
            )
            diversity_loss = similarity[off_diagonal].square().mean()
            mean_slot_attention_overlap = similarity[off_diagonal].mean()
        else:
            diversity_loss = slot_hidden.new_zeros((), dtype=torch.float32)
            mean_slot_attention_overlap = slot_hidden.new_zeros(
                (), dtype=torch.float32
            )
        if evidence_target is not None:
            coverage = slot_attention.mean(dim=0).clamp_min(1e-8)
            trace_support_loss = (
                evidence_target
                * (
                    evidence_target.clamp_min(1e-8).log()
                    - coverage.log()
                )
            ).sum()
        else:
            trace_support_loss = slot_hidden.new_zeros((), dtype=torch.float32)

        slot_parent_features = slot_hidden.float().mean(dim=1)
        slot_parent_features = F.normalize(
            self.slot_parent_projection(
                slot_parent_features.to(self.slot_parent_projection.weight.dtype)
            ).float(),
            dim=-1,
        )
        global_parent_features = F.normalize(
            self.global_parent_projection(
                self.parent_norm(global_parent_hidden).to(
                    self.global_parent_projection.weight.dtype
                )
            ).float(),
            dim=-1,
        )
        if int(global_parent_features.shape[0]) > 1:
            parent_logits = (
                slot_parent_features @ global_parent_features.transpose(0, 1)
            ) / self.parent_temperature
            parent_targets = torch.full(
                (self.slots_per_region,),
                parent_index,
                device=parent_logits.device,
                dtype=torch.long,
            )
            parent_consistency_loss = F.cross_entropy(
                parent_logits, parent_targets
            )
        else:
            parent_consistency_loss = slot_hidden.new_zeros(
                (), dtype=torch.float32
            )
        residual_identity = (
            (slot_hidden.float() - anchor.float()).square().mean()
            / anchor.float().square().mean().clamp_min(1e-6)
        )
        attention_entropy = -(
            attention.float().clamp_min(1e-8)
            * attention.float().clamp_min(1e-8).log()
        ).sum(dim=-1).mean()

        return EviSlotRegionOutput(
            hidden_states=slot_hidden.reshape(-1, self.hidden_size),
            position_embeddings=pooled_positions,
            deepstack_features_by_block=compressed_deepstack,
            slot_centers_xy=slot_centers.to(fine_centers_xy.dtype),
            slot_scales=slot_scales.to(fine_scales.dtype),
            parent_indices=torch.full(
                (self.slots_per_region,),
                parent_index,
                device=fine_hidden.device,
                dtype=torch.long,
            ),
            attention=attention,
            auxiliary_losses=EviSlotAuxiliaryLosses(
                trace_support=trace_support_loss,
                slot_diversity=diversity_loss,
                parent_consistency=parent_consistency_loss,
                residual_identity=residual_identity,
            ),
            attention_entropy=attention_entropy,
            mean_slot_attention_overlap=mean_slot_attention_overlap,
            residual_gate=residual_gate.float(),
        )
