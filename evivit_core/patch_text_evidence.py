"""Patch--text alignment heads for dense question-conditioned evidence maps."""

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from evivit_core.dense_models import coordinate_grid


def controlled_question_tokens(
    rows: list[dict[str, Any]],
    tokens: dict[str, dict[str, Any]],
    *,
    mode: str,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Build correct, shuffled, or zero question-token controls.

    The second return value records which source question was assigned to every
    target sample so control predictions remain auditable.
    """
    row_ids = [str(row["id"]) for row in rows]
    selected = dict(tokens)
    source_ids = {row_id: row_id for row_id in row_ids}
    if mode == "correct":
        return selected, source_ids
    if mode == "zero":
        dimension = int(next(iter(tokens.values()))["question_tokens"].shape[-1])
        for row_id in row_ids:
            selected[row_id] = {
                "id": row_id,
                "question": "",
                "token_ids": [0],
                "token_strings": ["<zero>"],
                "question_tokens": torch.zeros(1, dimension, dtype=torch.float16),
            }
            source_ids[row_id] = "<zero>"
        return selected, source_ids
    if mode == "shuffled":
        if len(row_ids) < 2:
            raise ValueError("shuffled question control requires at least two rows")
        shuffled = list(row_ids)
        random.Random(seed).shuffle(shuffled)
        # Deterministically rotate until there are no fixed points.
        for _ in range(len(shuffled)):
            if all(target != source for target, source in zip(row_ids, shuffled)):
                break
            shuffled = shuffled[1:] + shuffled[:1]
        if any(target == source for target, source in zip(row_ids, shuffled)):
            raise RuntimeError("failed to construct shuffled-question derangement")
        for target, source in zip(row_ids, shuffled):
            selected[target] = tokens[source]
            source_ids[target] = source
        return selected, source_ids
    raise ValueError(f"unknown question mode: {mode}")


class PatchTextEvidenceHead(nn.Module):
    """Predict one evidence logit per frozen Qwen visual token.

    Unlike ``FiLMDenseTraceHead``, which conditions every visual position on one
    mean-pooled question vector, this head retains a contextualized question
    sequence.  A text self-attention layer first models token relations; each
    visual token then attends to the whole contextualized sequence, and the
    resulting matched representation is refined on the original visual grid.
    """

    def __init__(
        self,
        input_dim: int | None = 512,
        visual_input_dim: int | None = None,
        text_input_dim: int | None = None,
        hidden_dim: int = 192,
        text_layers: int = 1,
        text_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if visual_input_dim is None:
            visual_input_dim = input_dim
        if text_input_dim is None:
            text_input_dim = input_dim
        if visual_input_dim is None or text_input_dim is None:
            raise ValueError(
                "input_dim or both visual_input_dim/text_input_dim must be provided"
            )
        if visual_input_dim <= 0 or text_input_dim <= 0:
            raise ValueError("visual/text input dimensions must be positive")
        if hidden_dim % text_heads:
            raise ValueError("hidden_dim must be divisible by text_heads")
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.hidden_dim = hidden_dim
        self.visual_input_dim = int(visual_input_dim)
        self.text_input_dim = int(text_input_dim)
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(self.visual_input_dim),
            nn.Linear(self.visual_input_dim, hidden_dim),
        )
        self.text_projection = nn.Sequential(
            nn.LayerNorm(self.text_input_dim),
            nn.Linear(self.text_input_dim, hidden_dim),
        )
        if text_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=text_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.text_context = nn.TransformerEncoder(encoder_layer, num_layers=text_layers)
        else:
            self.text_context = nn.Identity()
        self.log_temperature = nn.Parameter(torch.tensor(math.log(10.0)))
        # [visual, attended text, product, absolute difference, max similarity,
        #  normalized token-attention entropy]
        self.match_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4 + 2),
            nn.Linear(hidden_dim * 4 + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.coordinate_projection = nn.Conv2d(2, hidden_dim, kernel_size=1)
        self.spatial = nn.Sequential(
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Dropout2d(dropout),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.evidence_head = nn.Sequential(
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def aligned_hidden(
        self,
        visual: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the shared problem-conditioned spatial representation."""

        if visual.ndim == 3:
            visual = visual.unsqueeze(0)
        if text_tokens.ndim == 2:
            text_tokens = text_tokens.unsqueeze(0)
        if text_mask.ndim == 1:
            text_mask = text_mask.unsqueeze(0)
        if visual.ndim != 4:
            raise ValueError(f"visual must be HxWxD or BxHxWxD, got {tuple(visual.shape)}")
        if text_tokens.ndim != 3 or text_mask.ndim != 2:
            raise ValueError("text_tokens/text_mask must be BxLxD and BxL")

        batch, height, width, _ = visual.shape
        visual_tokens = self.visual_projection(visual.float().reshape(batch, height * width, -1))
        text = self.text_projection(text_tokens.float())
        padding_mask = ~text_mask.bool()
        if isinstance(self.text_context, nn.TransformerEncoder):
            text = self.text_context(text, src_key_padding_mask=padding_mask)
        else:
            text = self.text_context(text)

        visual_unit = F.normalize(visual_tokens, dim=-1)
        text_unit = F.normalize(text, dim=-1)
        temperature = self.log_temperature.exp().clamp(1.0, 100.0)
        similarity = torch.einsum("bnd,bld->bnl", visual_unit, text_unit) * temperature
        similarity = similarity.masked_fill(padding_mask[:, None, :], -torch.inf)
        alignment = torch.softmax(similarity, dim=-1)
        attended_text = torch.einsum("bnl,bld->bnd", alignment, text)

        valid_similarity = similarity.masked_fill(padding_mask[:, None, :], -1e4)
        max_similarity = valid_similarity.amax(dim=-1, keepdim=True) / temperature
        entropy = -(alignment.clamp_min(1e-8) * alignment.clamp_min(1e-8).log()).sum(
            dim=-1, keepdim=True
        )
        valid_lengths = text_mask.sum(dim=-1).clamp_min(2).float()
        entropy = entropy / valid_lengths.log()[:, None, None]
        match = torch.cat(
            [
                visual_tokens,
                attended_text,
                visual_tokens * attended_text,
                (visual_tokens - attended_text).abs(),
                max_similarity,
                entropy,
            ],
            dim=-1,
        )
        hidden = self.match_projection(match)
        hidden = hidden.reshape(batch, height, width, self.hidden_dim).permute(0, 3, 1, 2)
        coords = coordinate_grid(height, width, device=hidden.device)
        if batch > 1:
            coords = coords.expand(batch, -1, -1, -1)
        hidden = hidden + self.coordinate_projection(coords)
        return hidden, alignment.reshape(batch, height, width, -1)

    def forward(
        self,
        visual: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        return_alignment: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden, alignment = self.aligned_hidden(visual, text_tokens, text_mask)
        hidden = hidden + self.spatial(hidden)
        logits = self.evidence_head(hidden).squeeze(1)
        if return_alignment:
            return logits, alignment
        return logits


class DualPatchTextEvidenceHead(nn.Module):
    """Share problem alignment while predicting decisive and context maps.

    The original PTEA remains the decisive branch.  Only the final spatial
    residual block and 1x1 evidence head are duplicated for the broader context
    target, so visual/text projections and patch--text alignment stay shared.
    """

    def __init__(
        self,
        input_dim: int | None = 512,
        visual_input_dim: int | None = None,
        text_input_dim: int | None = None,
        hidden_dim: int = 192,
        text_layers: int = 1,
        text_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.shared = PatchTextEvidenceHead(
            input_dim=input_dim,
            visual_input_dim=visual_input_dim,
            text_input_dim=text_input_dim,
            hidden_dim=hidden_dim,
            text_layers=text_layers,
            text_heads=text_heads,
            dropout=dropout,
        )
        self.context_spatial = deepcopy(self.shared.spatial)
        self.context_evidence_head = deepcopy(self.shared.evidence_head)

    @property
    def visual_input_dim(self) -> int:
        return self.shared.visual_input_dim

    @property
    def text_input_dim(self) -> int:
        return self.shared.text_input_dim

    @property
    def hidden_dim(self) -> int:
        return self.shared.hidden_dim

    def initialize_from_single(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Warm-start both branches from one leakage-safe PTEA checkpoint."""

        self.shared.load_state_dict(state_dict, strict=True)
        self.context_spatial.load_state_dict(self.shared.spatial.state_dict(), strict=True)
        self.context_evidence_head.load_state_dict(
            self.shared.evidence_head.state_dict(), strict=True
        )

    def forward(
        self,
        visual: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        return_alignment: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        hidden, alignment = self.shared.aligned_hidden(visual, text_tokens, text_mask)
        decisive_hidden = hidden + self.shared.spatial(hidden)
        context_hidden = hidden + self.context_spatial(hidden)
        decisive_logits = self.shared.evidence_head(decisive_hidden).squeeze(1)
        context_logits = self.context_evidence_head(context_hidden).squeeze(1)
        if return_alignment:
            return decisive_logits, context_logits, alignment
        return decisive_logits, context_logits


class MultiScalePatchTextEvidenceHead(nn.Module):
    """Safely add early-detail evidence to a frozen semantic PTEA.

    Block16 remains the semantic anchor used by EviViT-v4.  A second branch
    reads an earlier Qwen-ViT grid (Block8 by default), aligns it to the same
    contextualized question tokens, and proposes a bounded hidden-state
    residual.  The residual projection is zero initialized, so a warm-started
    head is exactly the frozen single-scale PTEA before training.

    This is deliberately not a feature concatenation followed by a new map
    predictor: the verified semantic path is preserved and the detail branch
    can only contribute a spatially gated, norm-bounded correction.
    """

    def __init__(
        self,
        input_dim: int | None = 512,
        visual_input_dim: int | None = None,
        text_input_dim: int | None = None,
        hidden_dim: int = 192,
        text_layers: int = 1,
        text_heads: int = 4,
        dropout: float = 0.1,
        detail_block: int = 8,
        semantic_block: int = 16,
        max_relative_residual: float = 0.20,
    ) -> None:
        super().__init__()
        if not 0 < detail_block < semantic_block:
            raise ValueError("detail_block must be positive and before semantic_block")
        if not 0 < max_relative_residual <= 1:
            raise ValueError("max_relative_residual must be in (0, 1]")
        self.detail_block = int(detail_block)
        self.semantic_block = int(semantic_block)
        self.max_relative_residual = float(max_relative_residual)
        self.semantic = PatchTextEvidenceHead(
            input_dim=input_dim,
            visual_input_dim=visual_input_dim,
            text_input_dim=text_input_dim,
            hidden_dim=hidden_dim,
            text_layers=text_layers,
            text_heads=text_heads,
            dropout=dropout,
        )
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.detail_visual_projection = deepcopy(self.semantic.visual_projection)
        self.detail_match_projection = deepcopy(self.semantic.match_projection)
        self.detail_coordinate_projection = deepcopy(
            self.semantic.coordinate_projection
        )
        self.detail_spatial = deepcopy(self.semantic.spatial)
        fusion_dim = hidden_dim * 4
        self.detail_residual = nn.Sequential(
            nn.Conv2d(fusion_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.detail_gate = nn.Sequential(
            nn.Conv2d(fusion_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        # Exact v4 identity at initialization.  The first optimizer update can
        # move this projection; the gate then receives gradients as soon as a
        # non-zero detail residual exists.
        nn.init.zeros_(self.detail_residual[-1].weight)
        nn.init.zeros_(self.detail_residual[-1].bias)

    @property
    def visual_input_dim(self) -> int:
        return self.semantic.visual_input_dim

    @property
    def text_input_dim(self) -> int:
        return self.semantic.text_input_dim

    @property
    def hidden_dim(self) -> int:
        return self.semantic.hidden_dim

    @property
    def required_visual_blocks(self) -> tuple[int, int]:
        return self.detail_block, self.semantic_block

    def initialize_from_single(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        freeze_semantic: bool = True,
    ) -> None:
        """Warm-start from the frozen Block16 PTEA checkpoint."""

        self.semantic.load_state_dict(state_dict, strict=True)
        self.detail_visual_projection.load_state_dict(
            self.semantic.visual_projection.state_dict(), strict=True
        )
        self.detail_match_projection.load_state_dict(
            self.semantic.match_projection.state_dict(), strict=True
        )
        self.detail_coordinate_projection.load_state_dict(
            self.semantic.coordinate_projection.state_dict(), strict=True
        )
        self.detail_spatial.load_state_dict(
            self.semantic.spatial.state_dict(), strict=True
        )
        if freeze_semantic:
            self.semantic.requires_grad_(False)

    def _detail_aligned_hidden(
        self,
        visual: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        if visual.ndim == 3:
            visual = visual.unsqueeze(0)
        if text_tokens.ndim == 2:
            text_tokens = text_tokens.unsqueeze(0)
        if text_mask.ndim == 1:
            text_mask = text_mask.unsqueeze(0)
        if visual.ndim != 4:
            raise ValueError("detail visual must have shape HxWxD or BxHxWxD")
        batch, height, width, _ = visual.shape
        visual_tokens = self.detail_visual_projection(
            visual.float().reshape(batch, height * width, -1)
        )
        # Text semantics are shared with the frozen Block16 anchor so the two
        # scales differ only in visual depth, not in question interpretation.
        text = self.semantic.text_projection(text_tokens.float())
        padding_mask = ~text_mask.bool()
        if isinstance(self.semantic.text_context, nn.TransformerEncoder):
            text = self.semantic.text_context(
                text, src_key_padding_mask=padding_mask
            )
        else:
            text = self.semantic.text_context(text)
        visual_unit = F.normalize(visual_tokens, dim=-1)
        text_unit = F.normalize(text, dim=-1)
        temperature = self.semantic.log_temperature.exp().clamp(1.0, 100.0)
        similarity = (
            torch.einsum("bnd,bld->bnl", visual_unit, text_unit) * temperature
        )
        similarity = similarity.masked_fill(padding_mask[:, None, :], -torch.inf)
        alignment = torch.softmax(similarity, dim=-1)
        attended_text = torch.einsum("bnl,bld->bnd", alignment, text)
        max_similarity = (
            similarity.masked_fill(padding_mask[:, None, :], -1e4)
            .amax(dim=-1, keepdim=True)
            / temperature
        )
        entropy = -(
            alignment.clamp_min(1e-8) * alignment.clamp_min(1e-8).log()
        ).sum(dim=-1, keepdim=True)
        valid_lengths = text_mask.sum(dim=-1).clamp_min(2).float()
        entropy = entropy / valid_lengths.log()[:, None, None]
        match = torch.cat(
            [
                visual_tokens,
                attended_text,
                visual_tokens * attended_text,
                (visual_tokens - attended_text).abs(),
                max_similarity,
                entropy,
            ],
            dim=-1,
        )
        hidden = self.detail_match_projection(match)
        hidden = hidden.reshape(
            batch, height, width, self.hidden_dim
        ).permute(0, 3, 1, 2)
        coords = coordinate_grid(height, width, device=hidden.device)
        if batch > 1:
            coords = coords.expand(batch, -1, -1, -1)
        return hidden + self.detail_coordinate_projection(coords)

    def forward(
        self,
        visual_by_block: dict[int, torch.Tensor],
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        missing = set(self.required_visual_blocks).difference(visual_by_block)
        if missing:
            raise ValueError(f"missing visual blocks: {sorted(missing)}")
        detail_visual = visual_by_block[self.detail_block]
        semantic_visual = visual_by_block[self.semantic_block]
        if detail_visual.shape[:-1] != semantic_visual.shape[:-1]:
            raise ValueError("detail and semantic visual grids must be aligned")

        semantic_hidden, _ = self.semantic.aligned_hidden(
            semantic_visual, text_tokens, text_mask
        )
        semantic_hidden = semantic_hidden + self.semantic.spatial(semantic_hidden)
        detail_hidden = self._detail_aligned_hidden(
            detail_visual, text_tokens, text_mask
        )
        detail_hidden = detail_hidden + self.detail_spatial(detail_hidden)
        fusion = torch.cat(
            [
                semantic_hidden,
                detail_hidden,
                (semantic_hidden - detail_hidden).abs(),
                semantic_hidden * detail_hidden,
            ],
            dim=1,
        )
        raw_residual = self.detail_residual(fusion)
        semantic_scale = (
            semantic_hidden.float().square().mean(dim=1, keepdim=True)
            .sqrt()
            .clamp_min(1e-6)
            .detach()
        )
        bounded_residual = (
            torch.tanh(raw_residual.float() / semantic_scale)
            * semantic_scale
            * self.max_relative_residual
        ).to(semantic_hidden.dtype)
        gate = torch.sigmoid(self.detail_gate(fusion))
        applied = gate * bounded_residual
        fused = semantic_hidden + applied
        logits = self.semantic.evidence_head(fused).squeeze(1)
        if not return_diagnostics:
            return logits
        relative = (
            applied.float().square().mean().sqrt()
            / semantic_hidden.float().square().mean().sqrt().clamp_min(1e-8)
        )
        return logits, {
            "detail_gate_mean": gate.float().mean(),
            "detail_gate_std": gate.float().std(unbiased=False),
            "detail_relative_residual_l2": relative,
        }
