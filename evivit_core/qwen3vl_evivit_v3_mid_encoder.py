"""Question-aware mid-ViT evidence path for EviViT-v3.

This module connects the already verified segmented Qwen3-VL vision forward to
the sparse global--local evidence bridge.  It intentionally separates the
online path into two explicit phases:

1. encode the low-cost global image up to a chosen ViT block and expose the
   native Qwen 2x2-pooled feature grid to Mid-PTEA;
2. after Mid-PTEA allocates high-resolution regions, encode those regions to
   the same block, exchange sparse messages, retain both token streams, and
   resume the unchanged frozen Qwen-ViT.

The split prevents an online evaluator from encoding the global image twice.
With the bridge's zero-initialized output projections, the completed output is
bitwise identical to independently encoding and concatenating the same global
and fine views.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from evivit_core.evivit import multiresolution_fusion_indices
from evivit_core.evivit_evidence_bridge import (
    EvidenceBridgeDiagnostics,
    SparseGlobalLocalEvidenceBridge,
)
from evivit_core.evivit_evidence_salience_adapter import (
    EvidenceSalienceDiagnostics,
    EvidenceSalienceGlobalAdapter,
)
from evivit_core.eviblend import EviBlendDiagnostics, SafeJointResidualMixer
from evivit_core.evivit_global_anchor_relay import (
    GlobalAnchorRelay,
    GlobalAnchorRelayDiagnostics,
)
from evivit_core.evivit_global_frame_joint import (
    GlobalFrameJointDiagnostics,
    run_global_frame_joint_block,
)
from evivit_core.evivit_trace_calibrated_bridge import (
    TraceCalibratedBridgeDiagnostics,
    TraceCalibratedBridgeResidual,
)
from evivit_core.qwen3vl_evivit import qwen_grid_centers_in_original
from evivit_core.qwen3vl_evivit_v3 import (
    SegmentedVisionState,
    advance_segmented_qwen3vl_vision,
    finalize_segmented_qwen3vl_vision,
    prepare_segmented_qwen3vl_vision,
)


@dataclass(frozen=True)
class MidViTFineViewInput:
    """One high-resolution region sampled from the original source image."""

    pixel_values: torch.Tensor
    grid_thw: torch.Tensor
    bbox_xyxy: tuple[float, float, float, float]
    evidence_weight: float = 1.0
    token_evidence_weights: torch.Tensor | None = None
    readout_evidence_weights: torch.Tensor | None = None


@dataclass
class PreparedGlobalMidViT:
    """Reusable global state paused exactly at the bridge insertion block."""

    state: SegmentedVisionState
    insertion_block: int
    grid_h: int
    grid_w: int
    map_feature_grid: torch.Tensor
    map_feature_grids_by_block: dict[int, torch.Tensor] = field(
        default_factory=dict
    )

    def selector_visual_input(
        self, selector: nn.Module
    ) -> torch.Tensor | dict[int, torch.Tensor]:
        """Return the single- or multi-scale feature contract of a selector."""

        blocks = tuple(
            int(value)
            for value in getattr(selector, "required_visual_blocks", ())
        )
        if not blocks:
            return self.map_feature_grid
        missing = set(blocks).difference(self.map_feature_grids_by_block)
        if missing:
            raise ValueError(
                f"prepared global state lacks selector blocks: {sorted(missing)}"
            )
        return {
            block: self.map_feature_grids_by_block[block] for block in blocks
        }


@dataclass
class EviViTV3VisionOutput:
    """Unified global+fine visual tokens ready for the unchanged Qwen LLM."""

    image_embeds: torch.Tensor
    deepstack_features: list[torch.Tensor]
    bridge_diagnostics: EvidenceBridgeDiagnostics
    salience_diagnostics: EvidenceSalienceDiagnostics | None
    relay_diagnostics: GlobalAnchorRelayDiagnostics | None
    insertion_block: int
    global_image_tokens: int
    fine_image_tokens: int
    fine_encoder_tokens: int
    fine_view_token_counts: list[int]
    token_centers_xy: torch.Tensor
    token_scales: torch.Tensor
    token_levels: torch.Tensor
    token_region_ids: torch.Tensor
    salience_relative_residual_l2: torch.Tensor
    global_relative_residual_l2: torch.Tensor
    fine_relative_residual_l2: torch.Tensor
    relative_residual_l2: torch.Tensor
    relay_relative_residual_l2: torch.Tensor
    token_serialization: str
    serialization_moved_tokens: int
    visual_output_mode: str
    fine_readout_mode: str = "native"
    # Exact post-merger native Global raster.  These dimensions are needed by
    # Global-only identity routes because normalized source centers can lose
    # distinct columns after BF16 geometry bookkeeping.
    global_grid_h: int = 0
    global_grid_w: int = 0
    # Optional channel-preserving, native-merge-group pooled features captured
    # after selected ViT blocks. Empty in all normal inference/training calls.
    probe_features_by_block: dict[int, torch.Tensor] = field(default_factory=dict)
    # Optional repeated global/local fusion diagnostics. The insertion block is
    # included when progressive fusion is enabled.
    progressive_bridge_diagnostics: dict[int, EvidenceBridgeDiagnostics] = field(
        default_factory=dict
    )
    global_frame_joint_diagnostics: dict[
        int, GlobalFrameJointDiagnostics
    ] = field(default_factory=dict)
    safe_joint_residual_diagnostics: dict[
        int, EviBlendDiagnostics
    ] = field(default_factory=dict)
    trace_calibrated_bridge_diagnostics: (
        TraceCalibratedBridgeDiagnostics | None
    ) = None


def qwen_native_merge_group_grid(
    hidden_states: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    merge_size: int,
) -> torch.Tensor:
    """Return the channel-preserving H/m x W/m grid used by Mid-PTEA.

    Qwen orders pre-merger tokens by merged cell and then by the intra-cell
    row/column.  Reshaping in that native order and averaging only the m x m
    group preserves all feature channels and exactly matches offline feature
    extraction.
    """

    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden_size]")
    if grid_h <= 0 or grid_w <= 0 or grid_h % merge_size or grid_w % merge_size:
        raise ValueError("grid dimensions must be positive and divisible by merge_size")
    expected = grid_h * grid_w
    if int(hidden_states.shape[0]) != expected:
        raise ValueError(
            f"hidden-state length {hidden_states.shape[0]} != grid area {expected}"
        )
    grouped = hidden_states.reshape(
        grid_h // merge_size,
        grid_w // merge_size,
        merge_size,
        merge_size,
        hidden_states.shape[-1],
    )
    return grouped.float().mean(dim=(2, 3))


def max_evidence_confidence_by_global_parent(
    token_centers: torch.Tensor,
    evidence_weights: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Pool Fine-token confidence onto the native Global grid.

    Geometry is computed in float32 before discretization.  With BF16,
    values close to one (for example the center of a token near the right or
    bottom image boundary) may round to exactly ``1.0``.  Multiplying such a
    value by the grid width/height used to produce an out-of-range parent
    index.  The final integer clamp is an additional invariant guard.
    """

    if token_centers.ndim != 2 or token_centers.shape[-1] != 2:
        raise ValueError("token_centers must have shape [tokens, 2]")
    if evidence_weights.shape != (int(token_centers.shape[0]),):
        raise ValueError("evidence_weights must match the token count")
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("global grid dimensions must be positive")

    centers_float = token_centers.float()
    parent_columns = torch.floor(centers_float[:, 0] * grid_w).long()
    parent_rows = torch.floor(centers_float[:, 1] * grid_h).long()
    parent_columns.clamp_(0, grid_w - 1)
    parent_rows.clamp_(0, grid_h - 1)
    parent_indexes = parent_rows * grid_w + parent_columns
    parent_confidence = evidence_weights.new_zeros((grid_h * grid_w,))
    parent_confidence.scatter_reduce_(
        0,
        parent_indexes,
        evidence_weights.clamp(0.0, 1.0),
        reduce="amax",
        include_self=True,
    )
    return parent_confidence


class Qwen3VLEviViTV3MidEncoder(nn.Module):
    """Frozen Qwen-ViT with one trainable sparse evidence bridge."""

    def __init__(
        self,
        vision_model: nn.Module,
        evidence_bridge: SparseGlobalLocalEvidenceBridge,
        *,
        insertion_block: int,
        freeze_vision: bool = True,
        global_anchor_relay: GlobalAnchorRelay | None = None,
        evidence_salience_adapter: EvidenceSalienceGlobalAdapter | None = None,
        map_feature_blocks: Sequence[int] = (),
        token_serialization: str = "append",
        progressive_bridge_blocks: Sequence[int] = (),
        global_frame_joint_blocks: Sequence[int] = (),
        global_frame_coordinate_bins: int = 0,
        global_frame_joint_topology: str = "all_to_all",
        safe_joint_residual_mixer: SafeJointResidualMixer | None = None,
        safe_joint_residual_blocks: Sequence[int] = (),
        trace_calibrated_bridge_residual: (
            TraceCalibratedBridgeResidual | None
        ) = None,
    ) -> None:
        super().__init__()
        if not 0 < insertion_block < len(vision_model.blocks):
            raise ValueError("insertion_block must be inside the Qwen-ViT")
        hidden_size = int(vision_model.config.hidden_size)
        if evidence_bridge.hidden_size != hidden_size:
            raise ValueError("bridge hidden size must match Qwen-ViT hidden size")
        self.vision_model = vision_model
        self.evidence_bridge = evidence_bridge
        self.insertion_block = insertion_block
        self.freeze_vision = freeze_vision
        self.global_anchor_relay = global_anchor_relay
        self.evidence_salience_adapter = evidence_salience_adapter
        requested_map_blocks = {
            int(value) for value in map_feature_blocks
        }
        requested_map_blocks.add(int(insertion_block))
        if any(
            value <= 0 or value > insertion_block
            for value in requested_map_blocks
        ):
            raise ValueError(
                "map feature blocks must be positive and no later than "
                "the insertion block"
            )
        self.map_feature_blocks = tuple(sorted(requested_map_blocks))
        self.progressive_bridge_blocks = tuple(
            sorted({int(value) for value in progressive_bridge_blocks})
        )
        if any(
            value <= insertion_block or value > len(vision_model.blocks)
            for value in self.progressive_bridge_blocks
        ):
            raise ValueError(
                "progressive bridge blocks must be after insertion_block and "
                "at or before the final Qwen-ViT block"
            )
        self.global_frame_joint_blocks = tuple(
            sorted({int(value) for value in global_frame_joint_blocks})
        )
        if any(
            value < insertion_block or value >= len(vision_model.blocks)
            for value in self.global_frame_joint_blocks
        ):
            raise ValueError(
                "global-frame joint blocks must be at or after insertion_block "
                "and before the final Qwen-ViT boundary"
            )
        deepstack_indexes = set(
            int(value)
            for value in getattr(vision_model, "deepstack_visual_indexes", ())
        )
        if any(
            value in deepstack_indexes
            for value in self.global_frame_joint_blocks
        ):
            raise ValueError(
                "global-frame joint blocks must not be DeepStack capture blocks"
            )
        if global_frame_coordinate_bins == 1 or global_frame_coordinate_bins < 0:
            raise ValueError(
                "global_frame_coordinate_bins must be 0 or at least 2"
            )
        self.global_frame_coordinate_bins = int(global_frame_coordinate_bins)
        if global_frame_joint_topology not in {
            "all_to_all",
            "global_hub",
        }:
            raise ValueError(
                "global_frame_joint_topology must be all_to_all or global_hub"
            )
        self.global_frame_joint_topology = global_frame_joint_topology
        self.safe_joint_residual_mixer = safe_joint_residual_mixer
        self.trace_calibrated_bridge_residual = (
            trace_calibrated_bridge_residual
        )
        self.safe_joint_residual_blocks = tuple(
            sorted({int(value) for value in safe_joint_residual_blocks})
        )
        if bool(self.safe_joint_residual_blocks) != bool(
            self.safe_joint_residual_mixer is not None
        ):
            raise ValueError(
                "safe_joint_residual_mixer and safe_joint_residual_blocks "
                "must be configured together"
            )
        if any(
            value < insertion_block or value >= len(vision_model.blocks)
            for value in self.safe_joint_residual_blocks
        ):
            raise ValueError(
                "safe-joint residual blocks must be at or after "
                "insertion_block and before the final Qwen-ViT boundary"
            )
        if set(self.safe_joint_residual_blocks) & set(
            self.global_frame_joint_blocks
        ):
            raise ValueError(
                "a block cannot be both a replacing joint block and a "
                "safe-joint residual block"
            )
        if any(
            value in deepstack_indexes
            for value in self.safe_joint_residual_blocks
        ):
            raise ValueError(
                "safe-joint residual blocks must not be DeepStack capture blocks"
            )
        if (
            self.safe_joint_residual_mixer is not None
            and self.safe_joint_residual_mixer.hidden_size != hidden_size
        ):
            raise ValueError("safe-joint mixer hidden size must match Qwen-ViT")
        if (
            self.trace_calibrated_bridge_residual is not None
            and self.trace_calibrated_bridge_residual.hidden_size != hidden_size
        ):
            raise ValueError(
                "trace-calibrated bridge hidden size must match Qwen-ViT"
            )
        if (
            self.trace_calibrated_bridge_residual is not None
            and (
                self.safe_joint_residual_mixer is not None
                or self.global_anchor_relay is not None
                or bool(self.progressive_bridge_blocks)
            )
        ):
            raise ValueError(
                "trace-calibrated bridge is a standalone v4 calibration, "
                "not a module-stacking option"
            )
        self.last_safe_joint_gate_logits_by_block: dict[int, torch.Tensor] = {}
        self.last_safe_joint_global_gate_logits_by_block: dict[
            int, torch.Tensor
        ] = {}
        self.last_safe_joint_fine_centers_xy: torch.Tensor | None = None
        self.last_safe_joint_global_centers_xy: torch.Tensor | None = None
        self.last_trace_calibrated_global_importance_logits: (
            torch.Tensor | None
        ) = None
        self.last_trace_calibrated_fine_importance_logits: (
            torch.Tensor | None
        ) = None
        self.last_trace_calibrated_global_centers_xy: torch.Tensor | None = None
        self.last_trace_calibrated_fine_centers_xy: torch.Tensor | None = None
        self.last_bridge_gate_logits_by_block: dict[int, torch.Tensor] = {}
        if token_serialization not in {"append", "parent_interleave"}:
            raise ValueError(f"unsupported token serialization: {token_serialization}")
        self.token_serialization = token_serialization
        if global_anchor_relay is not None and global_anchor_relay.hidden_size != hidden_size:
            raise ValueError("relay hidden size must match Qwen-ViT hidden size")
        if (
            evidence_salience_adapter is not None
            and evidence_salience_adapter.hidden_size != hidden_size
        ):
            raise ValueError("salience adapter hidden size must match Qwen-ViT")
        if freeze_vision:
            self.vision_model.requires_grad_(False)

    @property
    def merge_size(self) -> int:
        return int(self.vision_model.spatial_merge_size)

    @staticmethod
    def _validate_single_still(grid_thw: torch.Tensor, label: str) -> tuple[int, int]:
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
        """Run the global image once and expose the online Mid-PTEA grid."""

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
        map_grid = feature_grids[self.insertion_block]
        return PreparedGlobalMidViT(
            state=state,
            insertion_block=self.insertion_block,
            grid_h=grid_h,
            grid_w=grid_w,
            map_feature_grid=map_grid,
            map_feature_grids_by_block=feature_grids,
        )

    def complete_from_prepared(
        self,
        prepared: PreparedGlobalMidViT,
        fine_views: Sequence[MidViTFineViewInput],
        *,
        bridge_mode: str = "bidirectional",
        visual_output_mode: str = "append",
        fine_readout_mode: str = "native",
        global_evidence_map: torch.Tensor | None = None,
        capture_probe_blocks: Sequence[int] = (),
    ) -> EviViTV3VisionOutput:
        """Bridge selected fine views and resume all streams through Qwen-ViT."""

        if prepared.insertion_block != self.insertion_block:
            raise ValueError("prepared state uses a different insertion block")
        if prepared.state.next_block_index != self.insertion_block:
            raise ValueError("prepared state is not paused at the insertion block")
        if visual_output_mode not in {"append", "global_only"}:
            raise ValueError(f"unknown visual_output_mode: {visual_output_mode}")
        if fine_readout_mode not in {"native", "evidence_residual"}:
            raise ValueError(f"unknown fine_readout_mode: {fine_readout_mode}")

        fine_states: list[SegmentedVisionState] = []
        fine_centers: list[torch.Tensor] = []
        fine_scales: list[torch.Tensor] = []
        fine_weights: list[torch.Tensor] = []
        fine_merged_evidence_weights: list[torch.Tensor] = []
        fine_lengths: list[int] = []
        fine_region_ids: list[torch.Tensor] = []
        fine_merged_centers: list[torch.Tensor] = []
        fine_merged_scales: list[torch.Tensor] = []
        fine_grids: list[tuple[int, int]] = []
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
            centers, scales = qwen_grid_centers_in_original(
                view.bbox_xyxy,
                grid_h=grid_h,
                grid_w=grid_w,
                spatial_merge_size=self.merge_size,
                device=state.hidden_states.device,
                dtype=state.hidden_states.dtype,
            )
            length = int(state.hidden_states.shape[0])
            fine_states.append(state)
            fine_centers.append(centers)
            fine_scales.append(scales)
            if view.token_evidence_weights is None:
                token_weights = torch.ones(
                    (length,),
                    device=state.hidden_states.device,
                    dtype=state.hidden_states.dtype,
                )
            else:
                token_weights = view.token_evidence_weights.to(
                    device=state.hidden_states.device,
                    dtype=state.hidden_states.dtype,
                )
                if token_weights.shape != (length,):
                    raise ValueError(
                        "token_evidence_weights must match the pre-merger "
                        f"Fine token count ({length},), got {tuple(token_weights.shape)}"
                    )
                if torch.any(token_weights < 0):
                    raise ValueError("token_evidence_weights must be non-negative")
            fine_weights.append(
                token_weights * max(0.0, float(view.evidence_weight))
            )
            readout_weights = (
                view.readout_evidence_weights.to(
                    device=state.hidden_states.device,
                    dtype=state.hidden_states.dtype,
                )
                if view.readout_evidence_weights is not None
                else token_weights
            )
            if readout_weights.shape != (length,):
                raise ValueError(
                    "readout_evidence_weights must match the pre-merger "
                    f"Fine token count ({length},), got {tuple(readout_weights.shape)}"
                )
            if torch.any(readout_weights < 0):
                raise ValueError("readout_evidence_weights must be non-negative")
            merge_area = self.merge_size * self.merge_size
            if int(readout_weights.numel()) % merge_area:
                raise ValueError(
                    "Fine evidence weights do not align with native merge groups"
                )
            fine_merged_evidence_weights.append(
                readout_weights.reshape(-1, merge_area).amax(dim=1)
            )
            fine_lengths.append(length)
            fine_grids.append((grid_h, grid_w))
            fine_region_ids.append(
                torch.full(
                    (length,),
                    index,
                    device=state.hidden_states.device,
                    dtype=torch.long,
                )
            )
            merged_centers, merged_scales = qwen_grid_centers_in_original(
                view.bbox_xyxy,
                grid_h=grid_h // self.merge_size,
                grid_w=grid_w // self.merge_size,
                spatial_merge_size=1,
                device=state.hidden_states.device,
                dtype=state.hidden_states.dtype,
            )
            fine_merged_centers.append(merged_centers)
            fine_merged_scales.append(merged_scales)

        native_global_tokens = prepared.state.hidden_states
        global_tokens = native_global_tokens
        salience_diagnostics: EvidenceSalienceDiagnostics | None = None
        if self.evidence_salience_adapter is not None:
            if global_evidence_map is None:
                raise ValueError(
                    "global_evidence_map is required when salience adapter is enabled"
                )
            global_tokens, salience_diagnostics = self.evidence_salience_adapter(
                global_tokens,
                global_evidence_map,
                grid_h=prepared.grid_h,
                grid_w=prepared.grid_w,
            )
        elif (
            global_evidence_map is not None
            and self.safe_joint_residual_mixer is None
            and self.trace_calibrated_bridge_residual is None
        ):
            raise ValueError(
                "global_evidence_map requires an evidence salience adapter"
            )
        salience_relative_residual_l2 = (
            (global_tokens.float() - native_global_tokens.float()).pow(2).mean()
            / native_global_tokens.float().pow(2).mean().clamp_min(1e-6)
        )
        if fine_states:
            fine_tokens = torch.cat(
                [state.hidden_states for state in fine_states], dim=0
            )
            centers = torch.cat(fine_centers, dim=0)
            scales = torch.cat(fine_scales, dim=0)
            weights = torch.cat(fine_weights, dim=0)
            region_ids = torch.cat(fine_region_ids, dim=0)
        else:
            fine_tokens = global_tokens.new_empty((0, global_tokens.shape[-1]))
            centers = global_tokens.new_empty((0, 2))
            scales = global_tokens.new_empty((0,))
            weights = global_tokens.new_empty((0,))
            region_ids = torch.empty((0,), device=global_tokens.device, dtype=torch.long)

        progressive_diagnostics: dict[int, EvidenceBridgeDiagnostics] = {}
        self.last_bridge_gate_logits_by_block = {}
        self.last_safe_joint_gate_logits_by_block = {}
        self.last_safe_joint_global_gate_logits_by_block = {}
        self.last_safe_joint_fine_centers_xy = None
        self.last_safe_joint_global_centers_xy = None
        self.last_trace_calibrated_global_importance_logits = None
        self.last_trace_calibrated_fine_importance_logits = None
        self.last_trace_calibrated_global_centers_xy = None
        self.last_trace_calibrated_fine_centers_xy = None

        def evidence_prior_at(
            token_centers: torch.Tensor,
        ) -> torch.Tensor:
            if global_evidence_map is None:
                return token_centers.new_zeros((int(token_centers.shape[0]),))
            probability = global_evidence_map.float()
            if probability.ndim != 2:
                raise ValueError(
                    "global_evidence_map must have shape [height, width]"
                )
            probability = probability.clamp_min(0)
            probability = probability / probability.sum().clamp_min(1e-12)
            sampling_grid = (
                2.0 * token_centers.float().clamp(0.0, 1.0) - 1.0
            ).reshape(1, -1, 1, 2)
            values = F.grid_sample(
                probability.reshape(1, 1, *probability.shape),
                sampling_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).reshape(-1)
            return (
                values / values.max().clamp_min(1e-12)
            ).to(token_centers.dtype)

        def bridge_streams(
            block_index: int,
            current_global: torch.Tensor,
            current_fine: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, EvidenceBridgeDiagnostics]:
            updated_global, updated_fine, current_diagnostics = (
                self.evidence_bridge(
                    current_global,
                    current_fine,
                    centers,
                    scales,
                    global_grid_h=prepared.grid_h,
                    global_grid_w=prepared.grid_w,
                    evidence_weights=weights,
                )
            )
            gate_logits = getattr(
                self.evidence_bridge, "last_child_gate_logits", None
            )
            if gate_logits is not None:
                self.last_bridge_gate_logits_by_block[block_index] = gate_logits
            if bridge_mode == "preserve_global":
                updated_global = current_global
                current_diagnostics = replace(
                    current_diagnostics,
                    global_relative_residual_l2=0.0,
                )
            elif bridge_mode == "evidence_weighted_writeback":
                # The bridge normalizes weighted child messages by their
                # accumulated weight.  That is desirable for feature
                # averaging, but it also removes absolute evidence confidence:
                # a weakly supported parent can receive a residual as strong as
                # a decisive one.  Restore that confidence after attention by
                # scaling each parent residual with the maximum evidence weight
                # of its children.  This is parameter-free, threshold-free and
                # can only reduce the already bounded bidirectional residual.
                parent_confidence = max_evidence_confidence_by_global_parent(
                    centers,
                    weights,
                    grid_h=prepared.grid_h,
                    grid_w=prepared.grid_w,
                )
                global_delta = updated_global - current_global
                weighted_delta = global_delta * parent_confidence[:, None]
                updated_global = current_global + weighted_delta
                global_relative = float(
                    (
                        weighted_delta.float().pow(2).mean()
                        / current_global.float().pow(2).mean().clamp_min(1e-6)
                    )
                    .detach()
                    .cpu()
                )
                current_diagnostics = replace(
                    current_diagnostics,
                    global_relative_residual_l2=global_relative,
                )
            elif bridge_mode == "preserve_fine":
                updated_fine = current_fine
                current_diagnostics = replace(
                    current_diagnostics,
                    fine_relative_residual_l2=0.0,
                )
            elif bridge_mode == "identity":
                updated_global = current_global
                updated_fine = current_fine
                current_diagnostics = replace(
                    current_diagnostics,
                    global_relative_residual_l2=0.0,
                    fine_relative_residual_l2=0.0,
                )
            elif bridge_mode != "bidirectional":
                raise ValueError(f"unknown bridge_mode: {bridge_mode}")
            if self.progressive_bridge_blocks:
                progressive_diagnostics[block_index] = current_diagnostics
            return updated_global, updated_fine, current_diagnostics

        bridged_global, bridged_fine, diagnostics = bridge_streams(
            self.insertion_block, global_tokens, fine_tokens
        )
        trace_calibrated_bridge_diagnostics = None
        if self.trace_calibrated_bridge_residual is not None:
            if not int(fine_tokens.shape[0]):
                raise RuntimeError(
                    "trace-calibrated bridge requires fine evidence tokens"
                )
            global_centers_premerger, _ = qwen_grid_centers_in_original(
                (0.0, 0.0, 1.0, 1.0),
                grid_h=prepared.grid_h,
                grid_w=prepared.grid_w,
                spatial_merge_size=self.merge_size,
                device=global_tokens.device,
                dtype=global_tokens.dtype,
            )
            global_prior = evidence_prior_at(global_centers_premerger)
            fine_prior = evidence_prior_at(centers)
            native_hidden = torch.cat([global_tokens, fine_tokens], dim=0)
            bridged_hidden = torch.cat(
                [bridged_global, bridged_fine], dim=0
            )
            level_ids = torch.cat(
                [
                    torch.zeros_like(global_prior),
                    torch.ones_like(fine_prior),
                ],
                dim=0,
            )
            calibrated_hidden, trace_calibrated_bridge_diagnostics = (
                self.trace_calibrated_bridge_residual(
                    native_hidden,
                    bridged_hidden,
                    torch.cat([global_prior, fine_prior], dim=0),
                    level_ids,
                    global_tokens=int(global_tokens.shape[0]),
                )
            )
            bridged_global, bridged_fine = calibrated_hidden.split(
                [int(global_tokens.shape[0]), int(fine_tokens.shape[0])],
                dim=0,
            )
            global_logits = (
                self.trace_calibrated_bridge_residual
                .last_global_importance_logits
            )
            fine_logits = (
                self.trace_calibrated_bridge_residual
                .last_fine_importance_logits
            )
            if global_logits is None or fine_logits is None:
                raise RuntimeError(
                    "trace-calibrated importance logits are missing"
                )
            self.last_trace_calibrated_global_importance_logits = (
                global_logits
            )
            self.last_trace_calibrated_fine_importance_logits = fine_logits
            self.last_trace_calibrated_global_centers_xy = (
                global_centers_premerger
            )
            self.last_trace_calibrated_fine_centers_xy = centers
        relay_diagnostics: GlobalAnchorRelayDiagnostics | None = None
        relay_reference_fine = bridged_fine
        if self.global_anchor_relay is not None:
            bridged_global, bridged_fine, relay_diagnostics = self.global_anchor_relay(
                bridged_global,
                bridged_fine,
                centers,
                scales,
                region_ids,
                global_grid_h=prepared.grid_h,
                global_grid_w=prepared.grid_w,
                evidence_weights=weights,
            )
            if int(bridged_fine.shape[0]):
                relay_relative_residual_l2 = (
                    (bridged_fine.float() - relay_reference_fine.float()).pow(2).mean()
                    / relay_reference_fine.float().pow(2).mean().clamp_min(1e-6)
                )
            else:
                relay_relative_residual_l2 = global_tokens.new_zeros(())
        else:
            relay_relative_residual_l2 = global_tokens.new_zeros(())
        global_relative_residual_l2 = (
            (bridged_global.float() - global_tokens.float()).pow(2).mean()
            / global_tokens.float().pow(2).mean().clamp_min(1e-6)
        )
        if int(fine_tokens.shape[0]):
            fine_relative_residual_l2 = (
                (bridged_fine.float() - fine_tokens.float()).pow(2).mean()
                / fine_tokens.float().pow(2).mean().clamp_min(1e-6)
            )
            relative_residual_l2 = 0.5 * (
                global_relative_residual_l2 + fine_relative_residual_l2
            )
        else:
            fine_relative_residual_l2 = global_relative_residual_l2.new_zeros(())
            relative_residual_l2 = global_relative_residual_l2
        global_state = replace(prepared.state, hidden_states=bridged_global)
        if fine_states:
            chunks = list(bridged_fine.split(fine_lengths, dim=0))
            fine_states = [
                replace(state, hidden_states=chunk)
                for state, chunk in zip(fine_states, chunks)
            ]

        block_count = len(self.vision_model.blocks)
        capture_points = tuple(sorted({int(value) for value in capture_probe_blocks}))
        if any(
            point < self.insertion_block or point > block_count
            for point in capture_points
        ):
            raise ValueError(
                "capture_probe_blocks must lie between insertion_block and the final block"
            )

        def pooled_probe_features() -> torch.Tensor:
            global_grid = qwen_native_merge_group_grid(
                global_state.hidden_states,
                grid_h=prepared.grid_h,
                grid_w=prepared.grid_w,
                merge_size=self.merge_size,
            ).reshape(-1, global_state.hidden_states.shape[-1])
            fine_grids_pooled = [
                qwen_native_merge_group_grid(
                    state.hidden_states,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    merge_size=self.merge_size,
                ).reshape(-1, state.hidden_states.shape[-1])
                for state, (grid_h, grid_w) in zip(fine_states, fine_grids)
            ]
            return torch.cat([global_grid, *fine_grids_pooled], dim=0)

        probe_features_by_block: dict[int, torch.Tensor] = {}
        global_frame_joint_diagnostics: dict[
            int, GlobalFrameJointDiagnostics
        ] = {}
        safe_joint_residual_diagnostics: dict[
            int, EviBlendDiagnostics
        ] = {}
        topology_blocks = {
            *self.global_frame_joint_blocks,
            *self.safe_joint_residual_blocks,
        }
        if set(capture_points) & topology_blocks:
            raise ValueError(
                "capture points cannot coincide with topology-interaction blocks"
            )
        if self.progressive_bridge_blocks or topology_blocks:
            boundaries = tuple(
                sorted(
                    {
                        *self.progressive_bridge_blocks,
                        *self.global_frame_joint_blocks,
                        *self.safe_joint_residual_blocks,
                        *capture_points,
                        block_count,
                    }
                )
            )
            for point in boundaries:
                global_state = advance_segmented_qwen3vl_vision(
                    self.vision_model, global_state, stop_after_blocks=point
                )
                fine_states = [
                    advance_segmented_qwen3vl_vision(
                        self.vision_model, state, stop_after_blocks=point
                    )
                    for state in fine_states
                ]
                if point in self.progressive_bridge_blocks:
                    stage_fine = (
                        torch.cat(
                            [state.hidden_states for state in fine_states], dim=0
                        )
                        if fine_states
                        else global_state.hidden_states.new_empty(
                            (0, global_state.hidden_states.shape[-1])
                        )
                    )
                    stage_global, stage_fine, _ = bridge_streams(
                        point, global_state.hidden_states, stage_fine
                    )
                    global_state = replace(
                        global_state, hidden_states=stage_global
                    )
                    if fine_states:
                        stage_chunks = list(stage_fine.split(fine_lengths, dim=0))
                        fine_states = [
                            replace(state, hidden_states=chunk)
                            for state, chunk in zip(fine_states, stage_chunks)
                        ]
                if point in self.global_frame_joint_blocks:
                    global_centers_premerger, _ = (
                        qwen_grid_centers_in_original(
                            (0.0, 0.0, 1.0, 1.0),
                            grid_h=prepared.grid_h,
                            grid_w=prepared.grid_w,
                            spatial_merge_size=self.merge_size,
                            device=global_state.hidden_states.device,
                            dtype=global_state.hidden_states.dtype,
                        )
                    )
                    joint_states, joint_diagnostics = (
                        run_global_frame_joint_block(
                            self.vision_model,
                            [global_state, *fine_states],
                            [global_centers_premerger, *fine_centers],
                            block_index=point,
                            topology=self.global_frame_joint_topology,
                            **(
                                {
                                    "coordinate_bins": (
                                        self.global_frame_coordinate_bins
                                    )
                                }
                                if self.global_frame_coordinate_bins
                                else {
                                    "reference_grid_h": prepared.grid_h,
                                    "reference_grid_w": prepared.grid_w,
                                }
                            ),
                        )
                    )
                    global_state = joint_states[0]
                    fine_states = joint_states[1:]
                    global_frame_joint_diagnostics[point] = (
                        joint_diagnostics
                    )
                if point in self.safe_joint_residual_blocks:
                    if self.safe_joint_residual_mixer is None:
                        raise RuntimeError("safe-joint mixer is missing")
                    global_centers_premerger, _ = (
                        qwen_grid_centers_in_original(
                            (0.0, 0.0, 1.0, 1.0),
                            grid_h=prepared.grid_h,
                            grid_w=prepared.grid_w,
                            spatial_merge_size=self.merge_size,
                            device=global_state.hidden_states.device,
                            dtype=global_state.hidden_states.dtype,
                        )
                    )
                    pre_states = [global_state, *fine_states]
                    independent_states = [
                        advance_segmented_qwen3vl_vision(
                            self.vision_model,
                            state,
                            stop_after_blocks=point + 1,
                        )
                        for state in pre_states
                    ]
                    joint_states, joint_diagnostics = (
                        run_global_frame_joint_block(
                            self.vision_model,
                            pre_states,
                            [global_centers_premerger, *fine_centers],
                            block_index=point,
                            topology=self.global_frame_joint_topology,
                            **(
                                {
                                    "coordinate_bins": (
                                        self.global_frame_coordinate_bins
                                    )
                                }
                                if self.global_frame_coordinate_bins
                                else {
                                    "reference_grid_h": prepared.grid_h,
                                    "reference_grid_w": prepared.grid_w,
                                }
                            ),
                        )
                    )
                    independent_hidden = torch.cat(
                        [state.hidden_states for state in independent_states],
                        dim=0,
                    )
                    joint_hidden = torch.cat(
                        [state.hidden_states for state in joint_states],
                        dim=0,
                    )
                    global_prior = evidence_prior_at(
                        global_centers_premerger
                    )
                    fine_prior = evidence_prior_at(
                        torch.cat(fine_centers, dim=0)
                    )
                    evidence_prior = torch.cat(
                        [global_prior, fine_prior], dim=0
                    )
                    level_ids = torch.cat(
                        [
                            torch.zeros_like(global_prior),
                            torch.ones_like(weights),
                        ],
                        dim=0,
                    )
                    blended_hidden, blend_diagnostics = (
                        self.safe_joint_residual_mixer(
                            independent_hidden,
                            joint_hidden,
                            evidence_prior,
                            level_ids,
                            global_tokens=int(
                                independent_states[0].hidden_states.shape[0]
                            ),
                        )
                    )
                    lengths = [
                        int(state.hidden_states.shape[0])
                        for state in independent_states
                    ]
                    blended_chunks = list(blended_hidden.split(lengths, dim=0))
                    blended_states = [
                        replace(state, hidden_states=chunk)
                        for state, chunk in zip(
                            independent_states, blended_chunks
                        )
                    ]
                    global_state = blended_states[0]
                    fine_states = blended_states[1:]
                    global_frame_joint_diagnostics[point] = (
                        joint_diagnostics
                    )
                    safe_joint_residual_diagnostics[point] = (
                        blend_diagnostics
                    )
                    gate_logits = (
                        self.safe_joint_residual_mixer.last_fine_gate_logits
                    )
                    if gate_logits is None:
                        raise RuntimeError("safe-joint gate logits are missing")
                    self.last_safe_joint_gate_logits_by_block[point] = (
                        gate_logits
                    )
                    all_gate_logits = (
                        self.safe_joint_residual_mixer.last_gate_logits
                    )
                    if all_gate_logits is None:
                        raise RuntimeError(
                            "safe-joint full gate logits are missing"
                        )
                    self.last_safe_joint_global_gate_logits_by_block[point] = (
                        all_gate_logits[
                            : int(
                                independent_states[0].hidden_states.shape[0]
                            )
                        ]
                    )
                    self.last_safe_joint_fine_centers_xy = (
                        torch.cat(fine_centers, dim=0)
                        if fine_centers
                        else gate_logits.new_empty((0, 2))
                    )
                    self.last_safe_joint_global_centers_xy = (
                        global_centers_premerger
                    )
                if point in capture_points:
                    probe_features_by_block[point] = pooled_probe_features()
        elif capture_points:
            for point in capture_points:
                global_state = advance_segmented_qwen3vl_vision(
                    self.vision_model, global_state, stop_after_blocks=point
                )
                fine_states = [
                    advance_segmented_qwen3vl_vision(
                        self.vision_model, state, stop_after_blocks=point
                    )
                    for state in fine_states
                ]
                probe_features_by_block[point] = pooled_probe_features()
            if capture_points[-1] < block_count:
                global_state = advance_segmented_qwen3vl_vision(
                    self.vision_model, global_state, stop_after_blocks=block_count
                )
                fine_states = [
                    advance_segmented_qwen3vl_vision(
                        self.vision_model, state, stop_after_blocks=block_count
                    )
                    for state in fine_states
                ]
        else:
            # Preserve the exact production execution path when probing is off.
            global_state = advance_segmented_qwen3vl_vision(
                self.vision_model, global_state, stop_after_blocks=block_count
            )
            fine_states = [
                advance_segmented_qwen3vl_vision(
                    self.vision_model, state, stop_after_blocks=block_count
                )
                for state in fine_states
            ]
        global_output = finalize_segmented_qwen3vl_vision(
            self.vision_model, global_state
        )
        fine_outputs = [
            finalize_segmented_qwen3vl_vision(self.vision_model, state)
            for state in fine_states
        ]

        if fine_readout_mode == "evidence_residual":
            global_grid_h = prepared.grid_h // self.merge_size
            global_grid_w = prepared.grid_w // self.merge_size
            residual_outputs = []
            for output, view_centers, view_weights in zip(
                fine_outputs,
                fine_merged_centers,
                fine_merged_evidence_weights,
            ):
                if int(output.image_embeds.shape[0]) != int(view_weights.shape[0]):
                    raise RuntimeError(
                        "Fine readout weights do not match merged Fine tokens"
                    )
                centers_float = view_centers.float()
                parent_columns = torch.floor(
                    centers_float[:, 0] * global_grid_w
                ).long().clamp(0, global_grid_w - 1)
                parent_rows = torch.floor(
                    centers_float[:, 1] * global_grid_h
                ).long().clamp(0, global_grid_h - 1)
                parent_indexes = parent_rows * global_grid_w + parent_columns
                gates = view_weights.to(
                    device=output.image_embeds.device,
                    dtype=output.image_embeds.dtype,
                ).clamp(0.0, 1.0)[:, None]
                parent_image = global_output.image_embeds.index_select(
                    0, parent_indexes
                )
                residual_image = parent_image + gates * (
                    output.image_embeds - parent_image
                )
                residual_deepstack = []
                for layer, fine_feature in enumerate(output.deepstack_features):
                    parent_feature = global_output.deepstack_features[
                        layer
                    ].index_select(0, parent_indexes)
                    residual_deepstack.append(
                        parent_feature + gates * (fine_feature - parent_feature)
                    )
                residual_outputs.append(
                    replace(
                        output,
                        image_embeds=residual_image,
                        deepstack_features=residual_deepstack,
                    )
                )
            fine_outputs = residual_outputs

        emitted_fine_outputs = fine_outputs if visual_output_mode == "append" else []
        image_chunks = [global_output.image_embeds] + [
            output.image_embeds for output in emitted_fine_outputs
        ]
        image_embeds = torch.cat(image_chunks, dim=0)
        deepstack_features = [
            torch.cat(
                [global_feature]
                + [
                    output.deepstack_features[layer]
                    for output in emitted_fine_outputs
                ],
                dim=0,
            )
            for layer, global_feature in enumerate(global_output.deepstack_features)
        ]
        fine_view_token_counts = [
            int(output.image_embeds.shape[0]) for output in fine_outputs
        ]
        global_centers, global_scales = qwen_grid_centers_in_original(
            (0.0, 0.0, 1.0, 1.0),
            grid_h=prepared.grid_h // self.merge_size,
            grid_w=prepared.grid_w // self.merge_size,
            spatial_merge_size=1,
            device=image_embeds.device,
            dtype=image_embeds.dtype,
        )
        emitted_fine_centers = (
            fine_merged_centers if visual_output_mode == "append" else []
        )
        emitted_fine_scales = (
            fine_merged_scales if visual_output_mode == "append" else []
        )
        centers = torch.cat([global_centers, *emitted_fine_centers], dim=0)
        token_scales = torch.cat([global_scales, *emitted_fine_scales], dim=0)
        token_levels = torch.cat(
            [
                torch.zeros(
                    global_centers.shape[0], device=image_embeds.device, dtype=torch.long
                )
            ]
            + [
                torch.ones(center.shape[0], device=image_embeds.device, dtype=torch.long)
                for center in emitted_fine_centers
            ],
            dim=0,
        )
        token_region_ids = torch.cat(
            [
                torch.zeros(
                    global_centers.shape[0], device=image_embeds.device, dtype=torch.long
                )
            ]
            + [
                torch.full(
                    (center.shape[0],),
                    index,
                    device=image_embeds.device,
                    dtype=torch.long,
                )
                for index, center in enumerate(emitted_fine_centers, 1)
            ],
            dim=0,
        )
        if int(centers.shape[0]) != int(image_embeds.shape[0]):
            raise RuntimeError("visual token metadata does not match merged token count")

        serialization_moved_tokens = 0
        if self.token_serialization == "parent_interleave":
            metadata = {
                "centers_xy": centers.detach().float().cpu().numpy(),
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
                feature.index_select(0, permutation) for feature in deepstack_features
            ]
            centers = centers.index_select(0, permutation)
            token_scales = token_scales.index_select(0, permutation)
            token_levels = token_levels.index_select(0, permutation)
            token_region_ids = token_region_ids.index_select(0, permutation)
            probe_features_by_block = {
                point: features.index_select(0, permutation)
                for point, features in probe_features_by_block.items()
            }
        return EviViTV3VisionOutput(
            image_embeds=image_embeds,
            deepstack_features=deepstack_features,
            bridge_diagnostics=diagnostics,
            salience_diagnostics=salience_diagnostics,
            relay_diagnostics=relay_diagnostics,
            insertion_block=self.insertion_block,
            global_image_tokens=int(global_output.image_embeds.shape[0]),
            fine_image_tokens=(
                sum(fine_view_token_counts)
                if visual_output_mode == "append"
                else 0
            ),
            fine_encoder_tokens=sum(fine_view_token_counts),
            fine_view_token_counts=fine_view_token_counts,
            token_centers_xy=centers,
            token_scales=token_scales,
            token_levels=token_levels,
            token_region_ids=token_region_ids,
            salience_relative_residual_l2=salience_relative_residual_l2,
            global_relative_residual_l2=global_relative_residual_l2,
            fine_relative_residual_l2=fine_relative_residual_l2,
            relative_residual_l2=relative_residual_l2,
            relay_relative_residual_l2=relay_relative_residual_l2,
            token_serialization=self.token_serialization,
            serialization_moved_tokens=serialization_moved_tokens,
            visual_output_mode=visual_output_mode,
            fine_readout_mode=fine_readout_mode,
            global_grid_h=int(prepared.grid_h // self.merge_size),
            global_grid_w=int(prepared.grid_w // self.merge_size),
            probe_features_by_block=probe_features_by_block,
            progressive_bridge_diagnostics=progressive_diagnostics,
            global_frame_joint_diagnostics=global_frame_joint_diagnostics,
            safe_joint_residual_diagnostics=safe_joint_residual_diagnostics,
            trace_calibrated_bridge_diagnostics=(
                trace_calibrated_bridge_diagnostics
            ),
        )

    def forward(
        self,
        global_pixel_values: torch.Tensor,
        global_grid_thw: torch.Tensor,
        fine_views: Sequence[MidViTFineViewInput] = (),
        *,
        bridge_mode: str = "bidirectional",
        visual_output_mode: str = "append",
        fine_readout_mode: str = "native",
        global_evidence_map: torch.Tensor | None = None,
        capture_probe_blocks: Sequence[int] = (),
    ) -> EviViTV3VisionOutput:
        """Convenience path for already allocated regions."""

        prepared = self.prepare_global(global_pixel_values, global_grid_thw)
        return self.complete_from_prepared(
            prepared,
            fine_views,
            bridge_mode=bridge_mode,
            visual_output_mode=visual_output_mode,
            fine_readout_mode=fine_readout_mode,
            global_evidence_map=global_evidence_map,
            capture_probe_blocks=capture_probe_blocks,
        )
