"""One native Qwen-ViT block over a shared original-image coordinate frame.

Global and high-resolution evidence streams are encoded independently through
the lower frozen Qwen-ViT blocks.  At one selected upper block, this primitive
temporarily concatenates the streams, assigns every patch an original-image
RoPE coordinate, runs the unchanged Qwen block once with one attention
sequence, and splits the streams again.  It adds no trainable parameters and
does not change the number or ordering of visual tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from evivit_core.qwen3vl_evivit_v3 import SegmentedVisionState


@dataclass(frozen=True)
class GlobalFrameJointDiagnostics:
    block_index: int
    streams: int
    total_tokens: int
    coordinate_bins: int
    minimum_stream_tokens: int
    maximum_stream_tokens: int
    coordinate_mode: str = "fixed_bins"
    reference_grid_h: int | None = None
    reference_grid_w: int | None = None
    native_global_positions_preserved: bool = False
    topology: str = "all_to_all"
    processed_tokens: int | None = None

    def as_dict(self) -> dict[str, int | str | bool | None]:
        return {
            "block_index": self.block_index,
            "streams": self.streams,
            "total_tokens": self.total_tokens,
            "coordinate_bins": self.coordinate_bins,
            "minimum_stream_tokens": self.minimum_stream_tokens,
            "maximum_stream_tokens": self.maximum_stream_tokens,
            "coordinate_mode": self.coordinate_mode,
            "reference_grid_h": self.reference_grid_h,
            "reference_grid_w": self.reference_grid_w,
            "native_global_positions_preserved": (
                self.native_global_positions_preserved
            ),
            "topology": self.topology,
            "processed_tokens": self.processed_tokens,
        }


def original_frame_position_embeddings(
    vision_model: nn.Module,
    centers_xy: torch.Tensor,
    *,
    coordinate_bins: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Qwen-compatible 2D RoPE from normalized original-image centers."""

    if centers_xy.ndim != 2 or centers_xy.shape[-1] != 2:
        raise ValueError("centers_xy must have shape [tokens, 2]")
    if coordinate_bins < 2:
        raise ValueError("coordinate_bins must be at least 2")
    if not torch.isfinite(centers_xy).all():
        raise ValueError("centers_xy must be finite")
    if int(centers_xy.shape[0]) == 0:
        raise ValueError("centers_xy must not be empty")
    centers = centers_xy.float().clamp(0.0, 1.0)
    # Match Qwen's native (row, column) lookup order.
    rows = torch.floor(centers[:, 1] * coordinate_bins).long()
    columns = torch.floor(centers[:, 0] * coordinate_bins).long()
    rows.clamp_(max=coordinate_bins - 1)
    columns.clamp_(max=coordinate_bins - 1)
    position_ids = torch.stack([rows, columns], dim=-1)
    frequency_table = vision_model.rotary_pos_emb(coordinate_bins)
    rotary_pos_emb = frequency_table[position_ids].flatten(1)
    rotary = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    return rotary.cos(), rotary.sin()


def continuous_positions_in_native_global_frame(
    vision_model: nn.Module,
    centers_xy: torch.Tensor,
    *,
    reference_grid_h: int,
    reference_grid_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map fine centers into the native global grid without quantizing subcells."""

    if centers_xy.ndim != 2 or centers_xy.shape[-1] != 2:
        raise ValueError("centers_xy must have shape [tokens, 2]")
    if reference_grid_h <= 0 or reference_grid_w <= 0:
        raise ValueError("reference grid dimensions must be positive")
    if not torch.isfinite(centers_xy).all():
        raise ValueError("centers_xy must be finite")
    centers = centers_xy.float().clamp(0.0, 1.0)
    # A native global patch at center (c + 0.5) / W must map exactly to
    # Qwen's integer rotary position c. Fine patches retain fractional offsets.
    rows = (centers[:, 1] * reference_grid_h - 0.5).clamp(
        0.0, reference_grid_h - 1
    )
    columns = (centers[:, 0] * reference_grid_w - 0.5).clamp(
        0.0, reference_grid_w - 1
    )
    inv_freq = vision_model.rotary_pos_emb.inv_freq.float()
    row_frequency = torch.outer(rows.to(inv_freq.device), inv_freq)
    column_frequency = torch.outer(columns.to(inv_freq.device), inv_freq)
    rotary_pos_emb = torch.stack(
        [row_frequency, column_frequency], dim=1
    ).flatten(1)
    rotary = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    return rotary.cos(), rotary.sin()


def run_global_frame_joint_block(
    vision_model: nn.Module,
    states: Sequence[SegmentedVisionState],
    centers_xy: Sequence[torch.Tensor],
    *,
    block_index: int,
    coordinate_bins: int | None = None,
    reference_grid_h: int | None = None,
    reference_grid_w: int | None = None,
    topology: str = "all_to_all",
) -> tuple[list[SegmentedVisionState], GlobalFrameJointDiagnostics]:
    """Run one unchanged Qwen block jointly and return the original streams."""

    if not states:
        raise ValueError("states must contain at least one stream")
    if topology not in {"all_to_all", "global_hub"}:
        raise ValueError("topology must be all_to_all or global_hub")
    if len(states) != len(centers_xy):
        raise ValueError("states and centers_xy must have the same length")
    block_count = len(vision_model.blocks)
    if not 0 <= block_index < block_count:
        raise ValueError("block_index is outside the Qwen-ViT tower")
    if block_index in set(vision_model.deepstack_visual_indexes):
        raise ValueError("joint block must not be a DeepStack capture block")
    if any(state.next_block_index != block_index for state in states):
        raise ValueError("every stream must be paused immediately before block_index")

    lengths = [int(state.hidden_states.shape[0]) for state in states]
    if any(length <= 0 for length in lengths):
        raise ValueError("joint streams must not be empty")
    for length, centers in zip(lengths, centers_xy):
        if centers.shape != (length, 2):
            raise ValueError("center count must match each stream token count")
    hidden_states = torch.cat([state.hidden_states for state in states], dim=0)
    native_global_mode = reference_grid_h is not None or reference_grid_w is not None
    per_stream_position_embeddings: list[
        tuple[torch.Tensor, torch.Tensor]
    ] = []
    if native_global_mode:
        if reference_grid_h is None or reference_grid_w is None:
            raise ValueError(
                "reference_grid_h and reference_grid_w must be provided together"
            )
        global_cosine, global_sine = states[0].position_embeddings
        if int(global_cosine.shape[0]) != lengths[0]:
            raise ValueError("native global position length does not match stream")
        per_stream_position_embeddings.append((global_cosine, global_sine))
        if len(states) > 1:
            for fine_center in centers_xy[1:]:
                fine_cosine, fine_sine = (
                    continuous_positions_in_native_global_frame(
                        vision_model,
                        fine_center.to(
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        ),
                        reference_grid_h=reference_grid_h,
                        reference_grid_w=reference_grid_w,
                    )
                )
                if fine_cosine.shape[1:] != global_cosine.shape[1:]:
                    raise ValueError(
                        "continuous fine RoPE must match the native global RoPE "
                        f"shape, got {tuple(fine_cosine.shape[1:])} versus "
                        f"{tuple(global_cosine.shape[1:])}"
                    )
                per_stream_position_embeddings.append(
                    (
                        fine_cosine.to(global_cosine.dtype),
                        fine_sine.to(global_sine.dtype),
                    )
                )
            position_embeddings = (
                torch.cat(
                    [value[0] for value in per_stream_position_embeddings],
                    dim=0,
                ),
                torch.cat(
                    [value[1] for value in per_stream_position_embeddings],
                    dim=0,
                ),
            )
        else:
            position_embeddings = (global_cosine, global_sine)
        coordinate_mode = "native_global_continuous_fine"
        diagnostic_bins = 0
    else:
        if coordinate_bins is None:
            raise ValueError(
                "coordinate_bins is required when no reference grid is provided"
            )
        all_centers = torch.cat(
            [
                centers.to(
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                for centers in centers_xy
            ],
            dim=0,
        )
        position_embeddings = original_frame_position_embeddings(
            vision_model,
            all_centers,
            coordinate_bins=coordinate_bins,
        )
        coordinate_mode = "fixed_bins"
        diagnostic_bins = int(coordinate_bins)
    if topology == "global_hub" and not native_global_mode:
        raise ValueError("global_hub topology requires native global coordinates")

    def run_one_sequence(
        values: torch.Tensor,
        positions: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        sequence_boundaries = F.pad(
            torch.tensor(
                [int(values.shape[0])],
                device=values.device,
                dtype=torch.int32,
            ),
            (1, 0),
            value=0,
        )
        return vision_model.blocks[block_index](
            values,
            cu_seqlens=sequence_boundaries,
            position_embeddings=positions,
        )

    if topology == "global_hub" and len(states) > 1:
        global_input = states[0].hidden_states
        global_candidates = []
        updated_fines = []
        processed_tokens = 0
        for fine_state, fine_positions in zip(
            states[1:], per_stream_position_embeddings[1:]
        ):
            branch_hidden = torch.cat(
                [global_input, fine_state.hidden_states], dim=0
            )
            branch_positions = (
                torch.cat(
                    [per_stream_position_embeddings[0][0], fine_positions[0]],
                    dim=0,
                ),
                torch.cat(
                    [per_stream_position_embeddings[0][1], fine_positions[1]],
                    dim=0,
                ),
            )
            branch_updated = run_one_sequence(
                branch_hidden, branch_positions
            )
            global_candidate, fine_updated = branch_updated.split(
                [lengths[0], int(fine_state.hidden_states.shape[0])],
                dim=0,
            )
            global_candidates.append(global_candidate)
            updated_fines.append(fine_updated)
            processed_tokens += int(branch_hidden.shape[0])
        updated_global = torch.stack(global_candidates, dim=0).mean(dim=0)
        chunks = [updated_global, *updated_fines]
    else:
        updated = run_one_sequence(hidden_states, position_embeddings)
        chunks = list(updated.split(lengths, dim=0))
        processed_tokens = int(hidden_states.shape[0])
    outputs = [
        replace(
            state,
            hidden_states=chunk,
            next_block_index=block_index + 1,
        )
        for state, chunk in zip(states, chunks)
    ]
    diagnostics = GlobalFrameJointDiagnostics(
        block_index=block_index,
        streams=len(states),
        total_tokens=sum(lengths),
        coordinate_bins=diagnostic_bins,
        minimum_stream_tokens=min(lengths),
        maximum_stream_tokens=max(lengths),
        coordinate_mode=coordinate_mode,
        reference_grid_h=reference_grid_h,
        reference_grid_w=reference_grid_w,
        native_global_positions_preserved=native_global_mode,
        topology=topology,
        processed_tokens=processed_tokens,
    )
    return outputs, diagnostics


__all__ = [
    "GlobalFrameJointDiagnostics",
    "continuous_positions_in_native_global_frame",
    "original_frame_position_embeddings",
    "run_global_frame_joint_block",
]
