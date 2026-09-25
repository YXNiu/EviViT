"""Segmented Qwen3-VL vision execution for EviViT-v3.

EviViT-v3 must inspect a global image after an intermediate Qwen-ViT block,
predict a question-conditioned Evidence Map, encode selected high-resolution
patches to the same depth, and then resume the frozen vision tower.  This module
implements only the pause/resume primitive.  It does not select regions, change
attention, or claim a v3 result.

The identity contract is strict: advancing one prepared state through all
blocks in one call or through any monotonic sequence of stop points must produce
the same pre-merger hidden states and DeepStack features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _deepstack_indexes(vision_model: nn.Module) -> list[int]:
    """Return optional Qwen-VL DeepStack capture points.

    Qwen3-VL exposes ``deepstack_visual_indexes`` and one merger per capture
    point.  Qwen3.5 keeps the same 24-block visual tower contract but has no
    DeepStack branch.  Treating the branch as optional lets the segmented
    pause/resume primitive support both towers without changing either model's
    native forward computation.
    """

    return [int(value) for value in getattr(vision_model, "deepstack_visual_indexes", ())]


@dataclass
class SegmentedVisionState:
    """A resumable Qwen3-VL vision-tower state.

    ``next_block_index`` is an exclusive progress counter.  A value of 12 means
    that blocks with zero-based indexes 0 through 11 have already run and that
    the state corresponds to the user-facing "after Block 12" insertion point.
    """

    hidden_states: torch.Tensor
    grid_thw: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    cu_seqlens: torch.Tensor
    next_block_index: int
    deepstack_features_by_block: dict[int, torch.Tensor]


@dataclass
class SegmentedVisionOutput:
    """Completed frozen-Qwen visual outputs."""

    premerger_hidden_states: torch.Tensor
    image_embeds: torch.Tensor
    deepstack_features: list[torch.Tensor]


def prepare_segmented_qwen3vl_vision(
    vision_model: nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
) -> SegmentedVisionState:
    """Patch-embed one image batch and prepare the unchanged Qwen positions."""

    hidden_states = vision_model.patch_embed(pixel_values)
    hidden_states = hidden_states + vision_model.fast_pos_embed_interpolate(grid_thw)
    rotary_pos_emb = vision_model.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    rotary = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (rotary.cos(), rotary.sin())
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    return SegmentedVisionState(
        hidden_states=hidden_states,
        grid_thw=grid_thw,
        position_embeddings=position_embeddings,
        cu_seqlens=cu_seqlens,
        next_block_index=0,
        deepstack_features_by_block={},
    )


def advance_segmented_qwen3vl_vision(
    vision_model: nn.Module,
    state: SegmentedVisionState,
    *,
    stop_after_blocks: int,
    **block_kwargs: Any,
) -> SegmentedVisionState:
    """Advance ``state`` to an exclusive block count without changing Qwen.

    For a 24-block tower, valid stop points are 0 through 24.  Calls must be
    monotonic; rewinding would require the caller to retain an earlier state.
    The input state is not mutated, which makes identity comparisons and branch
    experiments auditable.
    """

    block_count = len(vision_model.blocks)
    if not 0 <= stop_after_blocks <= block_count:
        raise ValueError(
            f"stop_after_blocks must be in [0, {block_count}], got "
            f"{stop_after_blocks}"
        )
    if stop_after_blocks < state.next_block_index:
        raise ValueError(
            "segmented Qwen-ViT execution cannot rewind: "
            f"state is at {state.next_block_index}, requested {stop_after_blocks}"
        )

    hidden_states = state.hidden_states
    deepstack = dict(state.deepstack_features_by_block)
    deepstack_indexes = _deepstack_indexes(vision_model)
    for layer_index in range(state.next_block_index, stop_after_blocks):
        hidden_states = vision_model.blocks[layer_index](
            hidden_states,
            cu_seqlens=state.cu_seqlens,
            position_embeddings=state.position_embeddings,
            **block_kwargs,
        )
        if layer_index in deepstack_indexes:
            merger_index = deepstack_indexes.index(layer_index)
            deepstack[layer_index] = vision_model.deepstack_merger_list[
                merger_index
            ](hidden_states)

    return SegmentedVisionState(
        hidden_states=hidden_states,
        grid_thw=state.grid_thw,
        position_embeddings=state.position_embeddings,
        cu_seqlens=state.cu_seqlens,
        next_block_index=stop_after_blocks,
        deepstack_features_by_block=deepstack,
    )


def finalize_segmented_qwen3vl_vision(
    vision_model: nn.Module,
    state: SegmentedVisionState,
) -> SegmentedVisionOutput:
    """Apply the unchanged Qwen merger after all vision blocks have run."""

    block_count = len(vision_model.blocks)
    if state.next_block_index != block_count:
        raise ValueError(
            "cannot finalize an incomplete vision state: "
            f"ran {state.next_block_index}/{block_count} blocks"
        )
    expected_indexes = _deepstack_indexes(vision_model)
    missing = [index for index in expected_indexes if index not in state.deepstack_features_by_block]
    if missing:
        raise RuntimeError(f"missing DeepStack features for blocks: {missing}")
    deepstack_features = [
        state.deepstack_features_by_block[index] for index in expected_indexes
    ]
    return SegmentedVisionOutput(
        premerger_hidden_states=state.hidden_states,
        image_embeds=vision_model.merger(state.hidden_states),
        deepstack_features=deepstack_features,
    )


def encode_qwen3vl_vision_segmented(
    vision_model: nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    stop_points: tuple[int, ...] = (),
    **block_kwargs: Any,
) -> SegmentedVisionOutput:
    """Reference segmented execution used by identity tests and smoke runs."""

    block_count = len(vision_model.blocks)
    points = (*stop_points, block_count)
    if tuple(sorted(set(points))) != points:
        raise ValueError("stop_points must be strictly increasing and exclude duplicates")
    state = prepare_segmented_qwen3vl_vision(
        vision_model, pixel_values, grid_thw
    )
    for point in points:
        state = advance_segmented_qwen3vl_vision(
            vision_model,
            state,
            stop_after_blocks=point,
            **block_kwargs,
        )
    return finalize_segmented_qwen3vl_vision(vision_model, state)
