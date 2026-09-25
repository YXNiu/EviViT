"""Qwen3-VL mid-encoder integration for EviSlot.

The Global image follows the exact frozen Qwen path.  Each selected Fine view
is encoded only to the configured insertion block, compressed by EviSlot into
native merger groups, and then resumed through the unchanged upper ViT.  This
is a length-changing alternative to EviBind's fixed-length residual bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn

from evivit_core.evislot import (
    EvidenceSlotCompressor,
    EviSlotAuxiliaryLosses,
    EviSlotDiagnostics,
)
from evivit_core.evivit import multiresolution_fusion_indices
from evivit_core.qwen3vl_evivit import qwen_grid_centers_in_original
from evivit_core.qwen3vl_evivit_v3 import (
    SegmentedVisionState,
    advance_segmented_qwen3vl_vision,
    finalize_segmented_qwen3vl_vision,
    prepare_segmented_qwen3vl_vision,
)
from evivit_core.qwen3vl_evivit_v3_mid_encoder import (
    MidViTFineViewInput,
    PreparedGlobalMidViT,
    qwen_native_merge_group_grid,
)


@dataclass
class EviSlotVisionOutput:
    """Drop-in Qwen visual features plus EviSlot mechanism diagnostics."""

    image_embeds: torch.Tensor
    deepstack_features: list[torch.Tensor]
    slot_diagnostics: EviSlotDiagnostics
    slot_auxiliary_losses: EviSlotAuxiliaryLosses
    insertion_block: int
    global_image_tokens: int
    fine_image_tokens: int
    fine_encoder_tokens: int
    fine_view_token_counts: list[int]
    token_centers_xy: torch.Tensor
    token_scales: torch.Tensor
    token_levels: torch.Tensor
    token_region_ids: torch.Tensor
    token_parent_ids: torch.Tensor
    global_relative_residual_l2: torch.Tensor
    fine_relative_residual_l2: torch.Tensor
    relative_residual_l2: torch.Tensor
    token_serialization: str
    serialization_moved_tokens: int
    visual_output_mode: str
    probe_features_by_block: dict[int, torch.Tensor] = field(default_factory=dict)


class Qwen3VLEviSlotEncoder(nn.Module):
    """Frozen Qwen-ViT with one trainable topology-aligned slot compressor."""

    def __init__(
        self,
        vision_model: nn.Module,
        slot_compressor: EvidenceSlotCompressor,
        *,
        insertion_block: int,
        freeze_vision: bool = True,
        map_feature_blocks: Sequence[int] = (),
        token_serialization: str = "append",
    ) -> None:
        super().__init__()
        if not 0 < insertion_block < len(vision_model.blocks):
            raise ValueError("insertion_block must be inside the Qwen-ViT")
        hidden_size = int(vision_model.config.hidden_size)
        if slot_compressor.hidden_size != hidden_size:
            raise ValueError("slot compressor hidden size must match Qwen-ViT")
        if (
            slot_compressor.spatial_merge_size
            != int(vision_model.spatial_merge_size)
        ):
            raise ValueError("slot compressor merge size must match Qwen-ViT")
        if token_serialization not in {"append", "parent_interleave"}:
            raise ValueError(f"unsupported token serialization: {token_serialization}")
        self.vision_model = vision_model
        self.slot_compressor = slot_compressor
        self.insertion_block = int(insertion_block)
        self.freeze_vision = bool(freeze_vision)
        self.token_serialization = token_serialization
        requested_blocks = {int(value) for value in map_feature_blocks}
        requested_blocks.add(self.insertion_block)
        if any(
            block <= 0 or block > self.insertion_block
            for block in requested_blocks
        ):
            raise ValueError(
                "map feature blocks must be positive and no later than insertion"
            )
        self.map_feature_blocks = tuple(sorted(requested_blocks))
        if freeze_vision:
            self.vision_model.requires_grad_(False)

    @property
    def merge_size(self) -> int:
        return int(self.vision_model.spatial_merge_size)

    @staticmethod
    def _validate_single_still(
        grid_thw: torch.Tensor, label: str
    ) -> tuple[int, int]:
        if grid_thw.shape != (1, 3):
            raise ValueError(f"{label} must contain exactly one image grid")
        temporal, height, width = [int(value) for value in grid_thw[0].tolist()]
        if temporal != 1:
            raise ValueError(f"{label} currently supports still images only")
        return height, width

    def _prefix_context(self):
        return torch.no_grad() if self.freeze_vision else torch.enable_grad()

    def prepare_global(
        self,
        global_pixel_values: torch.Tensor,
        global_grid_thw: torch.Tensor,
    ) -> PreparedGlobalMidViT:
        """Run Global once to the slot insertion point and expose PTEA grids."""

        grid_h, grid_w = self._validate_single_still(
            global_grid_thw, "global_grid_thw"
        )
        with self._prefix_context():
            state = prepare_segmented_qwen3vl_vision(
                self.vision_model, global_pixel_values, global_grid_thw
            )
            feature_grids: dict[int, torch.Tensor] = {}
            for block in self.map_feature_blocks:
                state = advance_segmented_qwen3vl_vision(
                    self.vision_model,
                    state,
                    stop_after_blocks=block,
                )
                feature_grids[block] = qwen_native_merge_group_grid(
                    state.hidden_states,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    merge_size=self.merge_size,
                )
        return PreparedGlobalMidViT(
            state=state,
            insertion_block=self.insertion_block,
            grid_h=grid_h,
            grid_w=grid_w,
            map_feature_grid=feature_grids[self.insertion_block],
            map_feature_grids_by_block=feature_grids,
        )

    @staticmethod
    def _mean_losses(
        losses: list[EviSlotAuxiliaryLosses], reference: torch.Tensor
    ) -> EviSlotAuxiliaryLosses:
        def mean(name: str) -> torch.Tensor:
            if not losses:
                return reference.new_zeros((), dtype=torch.float32)
            return torch.stack(
                [getattr(loss, name).float() for loss in losses]
            ).mean()

        return EviSlotAuxiliaryLosses(
            trace_support=mean("trace_support"),
            slot_diversity=mean("slot_diversity"),
            parent_consistency=mean("parent_consistency"),
            residual_identity=mean("residual_identity"),
        )

    def complete_from_prepared(
        self,
        prepared: PreparedGlobalMidViT,
        fine_views: Sequence[MidViTFineViewInput],
        *,
        bridge_mode: str = "bidirectional",
        visual_output_mode: str = "append",
        global_evidence_map: torch.Tensor | None = None,
        capture_probe_blocks: Sequence[int] = (),
    ) -> EviSlotVisionOutput:
        """Compress Fine evidence and resume the frozen native Qwen path."""

        if prepared.insertion_block != self.insertion_block:
            raise ValueError("prepared state uses a different insertion block")
        if prepared.state.next_block_index != self.insertion_block:
            raise ValueError("prepared state is not paused at the insertion block")
        if bridge_mode not in {"bidirectional", "preserve_global", "identity"}:
            raise ValueError(
                "EviSlot bridge_mode must be bidirectional, preserve_global, or identity"
            )
        if visual_output_mode not in {"append", "global_only"}:
            raise ValueError(f"unknown visual_output_mode: {visual_output_mode}")
        if capture_probe_blocks:
            raise ValueError("EviSlot probe capture is not implemented in the MVP")

        global_parent_hidden = qwen_native_merge_group_grid(
            prepared.state.hidden_states,
            grid_h=prepared.grid_h,
            grid_w=prepared.grid_w,
            merge_size=self.merge_size,
        ).reshape(-1, prepared.state.hidden_states.shape[-1]).to(
            prepared.state.hidden_states.dtype
        )
        global_parent_centers, _ = qwen_grid_centers_in_original(
            (0.0, 0.0, 1.0, 1.0),
            grid_h=prepared.grid_h // self.merge_size,
            grid_w=prepared.grid_w // self.merge_size,
            spatial_merge_size=1,
            device=prepared.state.hidden_states.device,
            dtype=prepared.state.hidden_states.dtype,
        )

        compressed_states: list[SegmentedVisionState] = []
        slot_centers_by_region: list[torch.Tensor] = []
        slot_scales_by_region: list[torch.Tensor] = []
        parent_ids_by_region: list[torch.Tensor] = []
        auxiliary_losses: list[EviSlotAuxiliaryLosses] = []
        attention_entropies: list[torch.Tensor] = []
        attention_overlaps: list[torch.Tensor] = []
        residual_gates: list[torch.Tensor] = []
        input_fine_image_tokens = 0

        compress_fine = bridge_mode != "identity"
        if compress_fine:
            for index, view in enumerate(fine_views):
                grid_h, grid_w = self._validate_single_still(
                    view.grid_thw, f"fine_views[{index}].grid_thw"
                )
                with self._prefix_context():
                    state = prepare_segmented_qwen3vl_vision(
                        self.vision_model, view.pixel_values, view.grid_thw
                    )
                    state = advance_segmented_qwen3vl_vision(
                        self.vision_model,
                        state,
                        stop_after_blocks=self.insertion_block,
                    )
                fine_centers, fine_scales = qwen_grid_centers_in_original(
                    view.bbox_xyxy,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    spatial_merge_size=self.merge_size,
                    device=state.hidden_states.device,
                    dtype=state.hidden_states.dtype,
                )
                compressed = self.slot_compressor(
                    state.hidden_states,
                    fine_centers,
                    fine_scales,
                    state.position_embeddings,
                    global_parent_hidden=global_parent_hidden,
                    global_parent_centers_xy=global_parent_centers,
                    bbox_xyxy=view.bbox_xyxy,
                    evidence_weight=view.evidence_weight,
                    deepstack_features_by_block=(
                        state.deepstack_features_by_block
                    ),
                    evidence_probability_map=global_evidence_map,
                )
                compressed_count = int(compressed.hidden_states.shape[0])
                compressed_grid = state.grid_thw.new_tensor(
                    [
                        [
                            1,
                            self.merge_size,
                            self.slot_compressor.slots_per_region
                            * self.merge_size,
                        ]
                    ]
                )
                compressed_cu_seqlens = torch.tensor(
                    [0, compressed_count],
                    device=state.cu_seqlens.device,
                    dtype=state.cu_seqlens.dtype,
                )
                compressed_states.append(
                    SegmentedVisionState(
                        hidden_states=compressed.hidden_states,
                        grid_thw=compressed_grid,
                        position_embeddings=compressed.position_embeddings,
                        cu_seqlens=compressed_cu_seqlens,
                        next_block_index=self.insertion_block,
                        deepstack_features_by_block=(
                            compressed.deepstack_features_by_block
                        ),
                    )
                )
                slot_centers_by_region.append(compressed.slot_centers_xy)
                slot_scales_by_region.append(compressed.slot_scales)
                parent_ids_by_region.append(compressed.parent_indices)
                auxiliary_losses.append(compressed.auxiliary_losses)
                attention_entropies.append(compressed.attention_entropy)
                attention_overlaps.append(
                    compressed.mean_slot_attention_overlap
                )
                residual_gates.append(compressed.residual_gate)
                input_fine_image_tokens += (
                    int(state.hidden_states.shape[0])
                    // self.slot_compressor.merge_group_size
                )

        block_count = len(self.vision_model.blocks)
        # Global has no dependency on EviSlot parameters and remains a strict
        # frozen-Qwen identity path. Fine compressed states retain autograd so
        # QA gradients can pass through frozen upper blocks into the slots.
        with self._prefix_context():
            global_state = advance_segmented_qwen3vl_vision(
                self.vision_model,
                prepared.state,
                stop_after_blocks=block_count,
            )
            global_output = finalize_segmented_qwen3vl_vision(
                self.vision_model, global_state
            )
        compressed_states = [
            advance_segmented_qwen3vl_vision(
                self.vision_model, state, stop_after_blocks=block_count
            )
            for state in compressed_states
        ]
        fine_outputs = [
            finalize_segmented_qwen3vl_vision(self.vision_model, state)
            for state in compressed_states
        ]

        emit_slots = visual_output_mode == "append" and bridge_mode != "identity"
        emitted_outputs = fine_outputs if emit_slots else []
        image_embeds = torch.cat(
            [global_output.image_embeds]
            + [output.image_embeds for output in emitted_outputs],
            dim=0,
        )
        deepstack_features = [
            torch.cat(
                [global_feature]
                + [
                    output.deepstack_features[layer]
                    for output in emitted_outputs
                ],
                dim=0,
            )
            for layer, global_feature in enumerate(
                global_output.deepstack_features
            )
        ]
        global_centers, global_scales = qwen_grid_centers_in_original(
            (0.0, 0.0, 1.0, 1.0),
            grid_h=prepared.grid_h // self.merge_size,
            grid_w=prepared.grid_w // self.merge_size,
            spatial_merge_size=1,
            device=image_embeds.device,
            dtype=image_embeds.dtype,
        )
        emitted_centers = slot_centers_by_region if emit_slots else []
        emitted_scales = slot_scales_by_region if emit_slots else []
        emitted_parents = parent_ids_by_region if emit_slots else []
        token_centers = torch.cat([global_centers, *emitted_centers], dim=0)
        token_scales = torch.cat([global_scales, *emitted_scales], dim=0)
        token_levels = torch.cat(
            [
                torch.zeros(
                    global_centers.shape[0],
                    device=image_embeds.device,
                    dtype=torch.long,
                )
            ]
            + [
                torch.ones(
                    centers.shape[0],
                    device=image_embeds.device,
                    dtype=torch.long,
                )
                for centers in emitted_centers
            ],
            dim=0,
        )
        token_region_ids = torch.cat(
            [
                torch.zeros(
                    global_centers.shape[0],
                    device=image_embeds.device,
                    dtype=torch.long,
                )
            ]
            + [
                torch.full(
                    (centers.shape[0],),
                    region_index,
                    device=image_embeds.device,
                    dtype=torch.long,
                )
                for region_index, centers in enumerate(emitted_centers, 1)
            ],
            dim=0,
        )
        token_parent_ids = torch.cat(
            [
                torch.arange(
                    global_centers.shape[0],
                    device=image_embeds.device,
                    dtype=torch.long,
                )
            ]
            + emitted_parents,
            dim=0,
        )
        if int(token_centers.shape[0]) != int(image_embeds.shape[0]):
            raise RuntimeError("EviSlot metadata does not match visual token count")

        serialization_moved_tokens = 0
        if self.token_serialization == "parent_interleave" and emit_slots:
            metadata = {
                "centers_xy": token_centers.detach().float().cpu().numpy(),
                "levels": token_levels.detach().cpu().numpy(),
                "region_ids": token_region_ids.detach().cpu().numpy(),
            }
            indices = multiresolution_fusion_indices(
                metadata, (), mode="parent_interleave"
            )
            serialization_moved_tokens = int(
                (indices != torch.arange(len(indices)).numpy()).sum()
            )
            permutation = torch.from_numpy(indices).to(
                device=image_embeds.device, dtype=torch.long
            )
            image_embeds = image_embeds.index_select(0, permutation)
            deepstack_features = [
                feature.index_select(0, permutation)
                for feature in deepstack_features
            ]
            token_centers = token_centers.index_select(0, permutation)
            token_scales = token_scales.index_select(0, permutation)
            token_levels = token_levels.index_select(0, permutation)
            token_region_ids = token_region_ids.index_select(0, permutation)
            token_parent_ids = token_parent_ids.index_select(0, permutation)

        mean_losses = self._mean_losses(
            auxiliary_losses, global_output.image_embeds
        )
        zero = global_output.image_embeds.new_zeros((), dtype=torch.float32)
        residual_identity = mean_losses.residual_identity
        diagnostics = EviSlotDiagnostics(
            input_fine_tokens=input_fine_image_tokens,
            slot_encoder_tokens=sum(
                int(state.hidden_states.shape[0]) for state in compressed_states
            ),
            emitted_slots=(
                sum(int(output.image_embeds.shape[0]) for output in fine_outputs)
                if emit_slots
                else 0
            ),
            region_count=len(fine_outputs),
            slots_per_region=self.slot_compressor.slots_per_region,
            parent_indices=[
                int(value)
                for values in parent_ids_by_region
                for value in values.detach().cpu().tolist()
            ],
            mean_attention_entropy=(
                float(torch.stack(attention_entropies).mean().detach().cpu())
                if attention_entropies
                else 0.0
            ),
            mean_slot_attention_overlap=(
                float(torch.stack(attention_overlaps).mean().detach().cpu())
                if attention_overlaps
                else 0.0
            ),
            mean_residual_gate=(
                float(torch.stack(residual_gates).mean().detach().cpu())
                if residual_gates
                else 0.0
            ),
            spatial_anchor_bias_enabled=(
                self.slot_compressor.enable_spatial_anchor_bias
            ),
            spatial_anchor_strength=(
                self.slot_compressor.spatial_anchor_strength
            ),
            spatial_anchor_sigma=self.slot_compressor.spatial_anchor_sigma,
            trace_support_loss=float(
                mean_losses.trace_support.detach().cpu()
            ),
            slot_diversity_loss=float(
                mean_losses.slot_diversity.detach().cpu()
            ),
            parent_consistency_loss=float(
                mean_losses.parent_consistency.detach().cpu()
            ),
            residual_identity_loss=float(residual_identity.detach().cpu()),
        )
        return EviSlotVisionOutput(
            image_embeds=image_embeds,
            deepstack_features=deepstack_features,
            slot_diagnostics=diagnostics,
            slot_auxiliary_losses=mean_losses,
            insertion_block=self.insertion_block,
            global_image_tokens=int(global_output.image_embeds.shape[0]),
            fine_image_tokens=diagnostics.emitted_slots,
            fine_encoder_tokens=input_fine_image_tokens,
            fine_view_token_counts=[
                int(output.image_embeds.shape[0]) for output in fine_outputs
            ],
            token_centers_xy=token_centers,
            token_scales=token_scales,
            token_levels=token_levels,
            token_region_ids=token_region_ids,
            token_parent_ids=token_parent_ids,
            global_relative_residual_l2=zero,
            fine_relative_residual_l2=residual_identity,
            relative_residual_l2=residual_identity,
            token_serialization=self.token_serialization,
            serialization_moved_tokens=serialization_moved_tokens,
            visual_output_mode=visual_output_mode,
        )

    def forward(
        self,
        global_pixel_values: torch.Tensor,
        global_grid_thw: torch.Tensor,
        fine_views: Sequence[MidViTFineViewInput] = (),
        *,
        bridge_mode: str = "bidirectional",
        visual_output_mode: str = "append",
        global_evidence_map: torch.Tensor | None = None,
        capture_probe_blocks: Sequence[int] = (),
    ) -> EviSlotVisionOutput:
        prepared = self.prepare_global(global_pixel_values, global_grid_thw)
        return self.complete_from_prepared(
            prepared,
            fine_views,
            bridge_mode=bridge_mode,
            visual_output_mode=visual_output_mode,
            global_evidence_map=global_evidence_map,
            capture_probe_blocks=capture_probe_blocks,
        )
