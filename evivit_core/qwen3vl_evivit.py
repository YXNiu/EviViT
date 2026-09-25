"""Qwen3-VL integration primitives for the first true EviViT-v2 encoder.

This file mirrors the public Transformers Qwen3-VL vision forward only up to
the final spatial merger.  Exposing that boundary lets EviViT write selected
high-resolution evidence into the global pre-merger grid while leaving Qwen's
patch embedder, 24 vision blocks, DeepStack branches, merger, and language
model unchanged.

The implementation targets Transformers 4.57.x.  A strict identity check
against the installed model must pass before any adapter training is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from evivit_core.evivit_fusion import CoordinateAwareEvidenceFusion, FusionDiagnostics


@dataclass
class VisionStemOutput:
    """Unmerged final ViT features and the unchanged Qwen DeepStack outputs."""

    hidden_states: torch.Tensor
    deepstack_features: list[torch.Tensor]


@dataclass(frozen=True)
class FineViewInput:
    """One internally sampled high-resolution region from the source image."""

    pixel_values: torch.Tensor
    grid_thw: torch.Tensor
    bbox_xyxy: tuple[float, float, float, float]
    evidence_weight: float


@dataclass
class EviViTVisionOutput:
    """Drop-in image features plus fusion diagnostics."""

    image_embeds: torch.Tensor
    deepstack_features: list[torch.Tensor]
    diagnostics: FusionDiagnostics
    deepstack_diagnostics: list[FusionDiagnostics]
    residual_l2: torch.Tensor
    relative_residual_l2: torch.Tensor
    final_residual_l2: torch.Tensor
    deepstack_residual_l2: torch.Tensor


def qwen_grid_centers_in_original(
    bbox_xyxy: Sequence[float],
    *,
    grid_h: int,
    grid_w: int,
    spatial_merge_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Qwen-ordered token centers and normalized patch scales.

    Qwen permutes spatial patches so the cells inside each merger block are
    contiguous.  Coordinates must use the same ordering as the hidden states.
    The box uses normalized source-image coordinates in ``[0, 1]``.
    """

    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain four coordinates")
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid dimensions must be positive")
    if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
        raise ValueError("fine grid must be divisible by Qwen spatial merge size")
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("bbox_xyxy must be a valid normalized source-image box")

    merged_h = grid_h // spatial_merge_size
    merged_w = grid_w // spatial_merge_size
    block_rows = torch.arange(merged_h, device=device)
    block_columns = torch.arange(merged_w, device=device)
    intra_rows = torch.arange(spatial_merge_size, device=device)
    intra_columns = torch.arange(spatial_merge_size, device=device)
    rows = (
        block_rows[:, None, None, None] * spatial_merge_size
        + intra_rows[None, None, :, None]
    )
    columns = (
        block_columns[None, :, None, None] * spatial_merge_size
        + intra_columns[None, None, None, :]
    )
    rows = rows.expand(
        merged_h, merged_w, spatial_merge_size, spatial_merge_size
    ).reshape(-1)
    columns = columns.expand(
        merged_h, merged_w, spatial_merge_size, spatial_merge_size
    ).reshape(-1)
    x = x1 + (columns.to(dtype) + 0.5) / grid_w * (x2 - x1)
    y = y1 + (rows.to(dtype) + 0.5) / grid_h * (y2 - y1)
    centers = torch.stack([x, y], dim=-1)
    patch_area = ((x2 - x1) / grid_w) * ((y2 - y1) / grid_h)
    scales = torch.full(
        (grid_h * grid_w,),
        patch_area**0.5,
        device=device,
        dtype=dtype,
    )
    return centers, scales


def encode_qwen3vl_premerger(
    vision_model: nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs: Any,
) -> VisionStemOutput:
    """Run the installed Qwen3-VL vision tower up to its final merger."""

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

    deepstack_features: list[torch.Tensor] = []
    for layer_num, block in enumerate(vision_model.blocks):
        hidden_states = block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if layer_num in vision_model.deepstack_visual_indexes:
            merger_index = vision_model.deepstack_visual_indexes.index(layer_num)
            deepstack_features.append(
                vision_model.deepstack_merger_list[merger_index](hidden_states)
            )
    return VisionStemOutput(hidden_states, deepstack_features)


class Qwen3VLEviViTEncoder(nn.Module):
    """Shared-ViT global/foveal encoder with fixed-length language input."""

    def __init__(
        self,
        vision_model: nn.Module,
        fusion_adapter: CoordinateAwareEvidenceFusion,
        deepstack_fusion_adapter: CoordinateAwareEvidenceFusion | None = None,
    ) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.fusion_adapter = fusion_adapter
        self.deepstack_fusion_adapter = deepstack_fusion_adapter
        config = vision_model.config
        if fusion_adapter.hidden_size != int(config.hidden_size):
            raise ValueError("fusion adapter hidden size must match Qwen-ViT")
        if (
            deepstack_fusion_adapter is not None
            and deepstack_fusion_adapter.hidden_size != int(config.out_hidden_size)
        ):
            raise ValueError(
                "DeepStack fusion hidden size must match Qwen vision output size"
            )

    def forward(
        self,
        global_pixel_values: torch.Tensor,
        global_grid_thw: torch.Tensor,
        fine_views: Sequence[FineViewInput] = (),
    ) -> EviViTVisionOutput:
        """Encode one source image with a global path and internal fine paths."""

        if global_grid_thw.shape != (1, 3):
            raise ValueError("v2.0 currently expects exactly one global image")
        if int(global_grid_thw[0, 0]) != 1:
            raise ValueError("v2.0 currently supports still images only")
        global_grid_h = int(global_grid_thw[0, 1].item())
        global_grid_w = int(global_grid_thw[0, 2].item())
        merge_size = int(self.vision_model.spatial_merge_size)
        global_stem = encode_qwen3vl_premerger(
            self.vision_model, global_pixel_values, global_grid_thw
        )

        fine_tokens: list[torch.Tensor] = []
        fine_centers: list[torch.Tensor] = []
        fine_scales: list[torch.Tensor] = []
        fine_weights: list[torch.Tensor] = []
        fine_deepstack_tokens: list[list[torch.Tensor]] = [
            [] for _ in global_stem.deepstack_features
        ]
        fine_merged_centers: list[torch.Tensor] = []
        fine_merged_scales: list[torch.Tensor] = []
        fine_merged_weights: list[torch.Tensor] = []
        for view in fine_views:
            if view.grid_thw.shape != (1, 3) or int(view.grid_thw[0, 0]) != 1:
                raise ValueError("each fine view must describe one still image")
            stem = encode_qwen3vl_premerger(
                self.vision_model, view.pixel_values, view.grid_thw
            )
            grid_h = int(view.grid_thw[0, 1].item())
            grid_w = int(view.grid_thw[0, 2].item())
            centers, scales = qwen_grid_centers_in_original(
                view.bbox_xyxy,
                grid_h=grid_h,
                grid_w=grid_w,
                spatial_merge_size=merge_size,
                device=stem.hidden_states.device,
                dtype=stem.hidden_states.dtype,
            )
            fine_tokens.append(stem.hidden_states)
            fine_centers.append(centers)
            fine_scales.append(scales)
            region_weight = max(0.0, float(view.evidence_weight))
            fine_weights.append(
                torch.full(
                    (stem.hidden_states.shape[0],),
                    region_weight,
                    device=stem.hidden_states.device,
                    dtype=stem.hidden_states.dtype,
                )
            )
            if len(stem.deepstack_features) != len(fine_deepstack_tokens):
                raise RuntimeError("DeepStack layer count changed across internal views")
            merged_centers, merged_scales = qwen_grid_centers_in_original(
                view.bbox_xyxy,
                grid_h=grid_h // merge_size,
                grid_w=grid_w // merge_size,
                spatial_merge_size=1,
                device=stem.hidden_states.device,
                dtype=stem.hidden_states.dtype,
            )
            fine_merged_centers.append(merged_centers)
            fine_merged_scales.append(merged_scales)
            fine_merged_weights.append(
                torch.full(
                    (merged_centers.shape[0],),
                    region_weight,
                    device=stem.hidden_states.device,
                    dtype=stem.hidden_states.dtype,
                )
            )
            for layer_index, feature in enumerate(stem.deepstack_features):
                fine_deepstack_tokens[layer_index].append(feature)

        if fine_tokens:
            concatenated_tokens = torch.cat(fine_tokens, dim=0)
            concatenated_centers = torch.cat(fine_centers, dim=0)
            concatenated_scales = torch.cat(fine_scales, dim=0)
            concatenated_weights = torch.cat(fine_weights, dim=0)
        else:
            concatenated_tokens = global_stem.hidden_states.new_empty(
                (0, global_stem.hidden_states.shape[-1])
            )
            concatenated_centers = global_stem.hidden_states.new_empty((0, 2))
            concatenated_scales = global_stem.hidden_states.new_empty((0,))
            concatenated_weights = global_stem.hidden_states.new_empty((0,))

        fused_global, diagnostics = self.fusion_adapter(
            global_stem.hidden_states,
            concatenated_tokens,
            concatenated_centers,
            concatenated_scales,
            global_grid_h=global_grid_h,
            global_grid_w=global_grid_w,
            evidence_weights=concatenated_weights,
            token_order="qwen_merge",
            spatial_merge_size=merge_size,
        )
        final_residual_l2 = (
            fused_global.float() - global_stem.hidden_states.float()
        ).pow(2).mean()

        fused_deepstack = list(global_stem.deepstack_features)
        deepstack_diagnostics: list[FusionDiagnostics] = []
        deepstack_residuals: list[torch.Tensor] = []
        relative_components = [
            final_residual_l2
            / global_stem.hidden_states.float().pow(2).mean().clamp_min(1e-6)
        ]
        if self.deepstack_fusion_adapter is not None and fine_tokens:
            merged_centers = torch.cat(fine_merged_centers, dim=0)
            merged_scales = torch.cat(fine_merged_scales, dim=0)
            merged_weights = torch.cat(fine_merged_weights, dim=0)
            fused_deepstack = []
            for global_feature, layer_fine_chunks in zip(
                global_stem.deepstack_features, fine_deepstack_tokens
            ):
                layer_fine = torch.cat(layer_fine_chunks, dim=0)
                fused_feature, layer_diagnostics = self.deepstack_fusion_adapter(
                    global_feature,
                    layer_fine,
                    merged_centers,
                    merged_scales,
                    global_grid_h=global_grid_h // merge_size,
                    global_grid_w=global_grid_w // merge_size,
                    evidence_weights=merged_weights,
                    token_order="row_major",
                    spatial_merge_size=1,
                )
                residual = (
                    fused_feature.float() - global_feature.float()
                ).pow(2).mean()
                fused_deepstack.append(fused_feature)
                deepstack_diagnostics.append(layer_diagnostics)
                deepstack_residuals.append(residual)
                relative_components.append(
                    residual
                    / global_feature.float().pow(2).mean().clamp_min(1e-6)
                )
        if deepstack_residuals:
            deepstack_residual_l2 = torch.stack(deepstack_residuals).mean()
        else:
            deepstack_residual_l2 = final_residual_l2.new_zeros(())
        residual_l2 = torch.stack(
            [final_residual_l2, deepstack_residual_l2]
            if deepstack_residuals
            else [final_residual_l2]
        ).mean()
        return EviViTVisionOutput(
            image_embeds=self.vision_model.merger(fused_global),
            deepstack_features=fused_deepstack,
            diagnostics=diagnostics,
            deepstack_diagnostics=deepstack_diagnostics,
            residual_l2=residual_l2,
            relative_residual_l2=torch.stack(relative_components).mean(),
            final_residual_l2=final_residual_l2,
            deepstack_residual_l2=deepstack_residual_l2,
        )
