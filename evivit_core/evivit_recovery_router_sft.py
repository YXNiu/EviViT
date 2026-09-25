"""Fail-closed building blocks for router-supervised recovery SFT.

This module deliberately contains no Qwen forward and launches no training.
It establishes the data, gradient, geometry, checkpoint, and phase contracts
needed before ``E-P*`` lanes are connected to the existing GPU trainer:

* ``train.qa.jsonl`` and ``router_aux.jsonl`` must match by raw-file hash,
  row count, ID, order, and duplicated QA metadata;
* PTEA probabilities used by map losses remain differentiable, while the
  existing top-k/connected-component decoder receives a detached copy;
* Visual-CoT boxes supervise only the decisive map.  The context map receives
  only original-checkpoint distillation because Visual-CoT has no context
  trace label;
* H-Safe consumes detached PTEA features and is trained by geometry only.
  Answer CE is intentionally absent from this contract after the previous
  answer-only H-Safe adaptation proved unstable;
* H-Safe geometry targets are projected through its bounded action space and
  use a robust log-area loss; a hash-bound sidecar may exclude only audited
  unit-ambiguous labels without changing the PTEA manifest;
* router checkpoints are atomic and bind optimizer/scheduler/RNG state to a
  canonical protocol fingerprint.

Registering a lane here does not enable it in ``train_evivit_v3_bridge.py``.
That parser continues to reject router-supervised lanes until the 32/128-row
gradient and geometry gates have passed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from evivit_core.evivit_adaptive_box import (
    AdaptiveBoxHead,
    apply_residual,
    box_area,
    generalized_iou,
    residual_target,
    target_coverage,
)
from evivit_core.evivit_recovery_sft import RecoveryLane


ROUTER_AUX_FORMAT_VERSION = "visual_cot_recovery_router_aux_v1"
ROUTER_PROTOCOL_VERSION = "evivit_recovery_router_sft_contract_v1"
ROUTER_STAGE_A_END_EPOCH = 0.5
H_SAFE_GEOMETRY_LOSS_VERSION = "h_safe_geometry_v2_realizable_huber_log_area"
H_SAFE_GEOMETRY_MASK_FORMAT_VERSION = (
    "visual_cot_recovery_h_safe_geometry_mask_v1"
)
H_SAFE_MAXIMUM_SUPERVISED_DECISIVE_ANCHORS = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash one protocol without depending on dictionary insertion order."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL file: {path}")
    return rows


def _normalized_boxes(value: Any, *, row_id: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"router aux {row_id} has no bbox_xyxy_normalized")
    boxes: list[tuple[float, ...]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise ValueError(f"router aux {row_id} bbox {index} is not xyxy")
        box = tuple(float(item) for item in raw)
        if not all(math.isfinite(item) for item in box):
            raise ValueError(f"router aux {row_id} bbox {index} is non-finite")
        x1, y1, x2, y2 = box
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError(f"router aux {row_id} bbox {index} is out of range")
        boxes.append(box)
    return tuple(boxes)


@dataclass(frozen=True)
class AlignedRouterRow:
    id: str
    qa: dict[str, Any]
    router_aux: dict[str, Any]
    boxes: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class RouterAlignment:
    rows: tuple[AlignedRouterRow, ...]
    train_manifest_sha256: str
    router_aux_sha256: str
    order_sha256: str

    def protocol_fields(self) -> dict[str, Any]:
        return {
            "router_aux_format": ROUTER_AUX_FORMAT_VERSION,
            "rows": len(self.rows),
            "train_manifest_sha256": self.train_manifest_sha256,
            "router_aux_sha256": self.router_aux_sha256,
            "order_sha256": self.order_sha256,
        }


@dataclass(frozen=True)
class HSafeGeometryMask:
    ids: frozenset[str]
    sha256: str
    train_manifest_sha256: str
    router_aux_sha256: str

    def protocol_fields(self) -> dict[str, Any]:
        return {
            "format_version": H_SAFE_GEOMETRY_MASK_FORMAT_VERSION,
            "scope": "h_safe_geometry_loss_only",
            "action": "exclude",
            "sha256": self.sha256,
            "train_manifest_sha256": self.train_manifest_sha256,
            "router_aux_sha256": self.router_aux_sha256,
            "rows": len(self.ids),
            "ids": sorted(self.ids),
            "ptea_router_aux": "unchanged",
        }


def load_h_safe_geometry_mask(
    path: Path,
    *,
    expected_sha256: str,
    expected_train_manifest_sha256: str,
    expected_router_aux_sha256: str,
) -> HSafeGeometryMask:
    """Load a hash-bound mask that is forbidden from changing PTEA labels."""

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"H-Safe geometry mask hash mismatch: {actual_sha256} != "
            f"{expected_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("H-Safe geometry mask must be a JSON object")
    if payload.get("format_version") != H_SAFE_GEOMETRY_MASK_FORMAT_VERSION:
        raise ValueError("unknown H-Safe geometry mask format")
    if (
        payload.get("scope") != "h_safe_geometry_loss_only"
        or payload.get("action") != "exclude"
        or payload.get("ptea_router_aux") != "unchanged"
    ):
        raise ValueError("H-Safe geometry mask has an unsafe scope")
    if payload.get("router_aux_sha256") != expected_router_aux_sha256:
        raise ValueError("H-Safe geometry mask is bound to a different router aux")
    if payload.get("train_manifest_sha256") != expected_train_manifest_sha256:
        raise ValueError("H-Safe geometry mask is bound to a different train manifest")
    raw_ids = payload.get("ids")
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or not all(isinstance(value, str) and value for value in raw_ids)
        or len(raw_ids) != len(set(raw_ids))
        or payload.get("rows") != len(raw_ids)
    ):
        raise ValueError("H-Safe geometry mask IDs are missing or duplicated")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != len(raw_ids):
        raise ValueError("H-Safe geometry mask audit entries do not match its IDs")
    entry_ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("H-Safe geometry mask contains a malformed audit entry")
        entry_id = entry.get("id")
        issue = entry.get("issue")
        raw_box = entry.get("raw_release_bbox")
        if (
            not isinstance(entry_id, str)
            or not entry_id
            or not isinstance(issue, str)
            or not issue.strip()
            or not isinstance(raw_box, list)
            or len(raw_box) != 4
        ):
            raise ValueError("H-Safe geometry mask contains a malformed audit entry")
        numeric_box = [float(value) for value in raw_box]
        if not all(math.isfinite(value) for value in numeric_box):
            raise ValueError("H-Safe geometry mask contains a non-finite raw bbox")
        x1, y1, x2, y2 = numeric_box
        if not (x1 < x2 and y1 < y2):
            raise ValueError("H-Safe geometry mask contains an invalid raw bbox")
        entry_ids.append(entry_id)
    if len(entry_ids) != len(set(entry_ids)) or set(entry_ids) != set(raw_ids):
        raise ValueError("H-Safe geometry mask audit entries do not match its IDs")
    return HSafeGeometryMask(
        ids=frozenset(raw_ids),
        sha256=actual_sha256,
        train_manifest_sha256=expected_train_manifest_sha256,
        router_aux_sha256=expected_router_aux_sha256,
    )


def load_aligned_router_rows(
    train_manifest: Path,
    router_aux: Path,
    *,
    expected_train_sha256: str | None = None,
    expected_router_sha256: str | None = None,
    expected_rows: int | None = None,
) -> RouterAlignment:
    """Load bbox supervision only after exact manifest alignment succeeds."""

    train_hash = sha256_file(train_manifest)
    router_hash = sha256_file(router_aux)
    if expected_train_sha256 is not None and train_hash != expected_train_sha256:
        raise ValueError(
            f"train manifest hash mismatch: {train_hash} != {expected_train_sha256}"
        )
    if expected_router_sha256 is not None and router_hash != expected_router_sha256:
        raise ValueError(
            f"router aux hash mismatch: {router_hash} != {expected_router_sha256}"
        )
    qa_rows = _read_jsonl(train_manifest)
    aux_rows = _read_jsonl(router_aux)
    if len(qa_rows) != len(aux_rows):
        raise ValueError(
            f"train/router row-count mismatch: {len(qa_rows)} != {len(aux_rows)}"
        )
    if expected_rows is not None and len(qa_rows) != expected_rows:
        raise ValueError(
            f"unexpected aligned row count: {len(qa_rows)} != {expected_rows}"
        )

    aligned: list[AlignedRouterRow] = []
    seen: set[str] = set()
    duplicated_fields = ("image", "question", "answer", "source_dataset")
    for index, (qa, aux) in enumerate(zip(qa_rows, aux_rows)):
        qa_id = str(qa.get("id", ""))
        aux_id = str(aux.get("id", ""))
        if not qa_id or qa_id != aux_id:
            raise ValueError(
                f"train/router ID or order mismatch at row {index}: "
                f"{qa_id!r} != {aux_id!r}"
            )
        if qa_id in seen:
            raise ValueError(f"duplicate aligned sample ID: {qa_id}")
        seen.add(qa_id)
        for key in duplicated_fields:
            if qa.get(key) != aux.get(key):
                raise ValueError(
                    f"train/router {key} mismatch for {qa_id}: "
                    f"{qa.get(key)!r} != {aux.get(key)!r}"
                )
        for key in ("image_width", "image_height"):
            if not isinstance(aux.get(key), int) or int(aux[key]) <= 0:
                raise ValueError(f"router aux {qa_id} has invalid {key}")
        aligned.append(
            AlignedRouterRow(
                id=qa_id,
                qa=qa,
                router_aux=aux,
                boxes=_normalized_boxes(
                    aux.get("bbox_xyxy_normalized"), row_id=qa_id
                ),
            )
        )
    order_hash = hashlib.sha256(
        b"\0".join(row.id.encode("utf-8") for row in aligned)
    ).hexdigest()
    return RouterAlignment(
        rows=tuple(aligned),
        train_manifest_sha256=train_hash,
        router_aux_sha256=router_hash,
        order_sha256=order_hash,
    )


def bbox_soft_map(
    boxes: Sequence[Sequence[float]],
    height: int,
    width: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Rasterize normalized boxes by fractional cell overlap.

    Fractional overlap avoids the aliasing of a hard integer mask for thin OCR
    boxes.  Each target box contributes one normalized distribution and the
    mean is normalized again, so large boxes do not dominate small ones.
    """

    if height <= 0 or width <= 0:
        raise ValueError("soft-map dimensions must be positive")
    normalized = _normalized_boxes([list(box) for box in boxes], row_id="<tensor>")
    x0 = torch.arange(width, device=device, dtype=torch.float32) / width
    x1 = torch.arange(1, width + 1, device=device, dtype=torch.float32) / width
    y0 = torch.arange(height, device=device, dtype=torch.float32) / height
    y1 = torch.arange(1, height + 1, device=device, dtype=torch.float32) / height
    maps = []
    for left, top, right, bottom in normalized:
        x_overlap = (
            torch.minimum(x1, torch.tensor(right, device=device))
            - torch.maximum(x0, torch.tensor(left, device=device))
        ).clamp_min(0)
        y_overlap = (
            torch.minimum(y1, torch.tensor(bottom, device=device))
            - torch.maximum(y0, torch.tensor(top, device=device))
        ).clamp_min(0)
        mass = y_overlap[:, None] * x_overlap[None, :]
        if float(mass.sum()) <= 0:
            raise ValueError("bbox has no support on requested soft-map grid")
        maps.append(mass / mass.sum())
    result = torch.stack(maps).mean(dim=0)
    return result / result.sum().clamp_min(1e-12)


def _spatial_probability(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim not in {2, 3}:
        raise ValueError("PTEA logits must be HxW or BxHxW")
    if logits.ndim == 2:
        return torch.softmax(logits.flatten(), dim=0).reshape_as(logits)
    return torch.softmax(logits.flatten(1), dim=1).reshape_as(logits)


@dataclass(frozen=True)
class PTEATrainingPaths:
    decisive_probability: torch.Tensor
    context_probability: torch.Tensor
    decoder_decisive_probability: torch.Tensor
    decoder_context_probability: torch.Tensor


def split_ptea_training_paths(
    decisive_logits: torch.Tensor,
    context_logits: torch.Tensor,
) -> PTEATrainingPaths:
    """Separate differentiable map losses from the discrete region decoder."""

    if decisive_logits.shape != context_logits.shape:
        raise ValueError("decisive/context PTEA logits must have equal shapes")
    decisive = _spatial_probability(decisive_logits)
    context = _spatial_probability(context_logits)
    return PTEATrainingPaths(
        decisive_probability=decisive,
        context_probability=context,
        decoder_decisive_probability=decisive.detach(),
        decoder_context_probability=context.detach(),
    )


def _as_batched_map(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError("map must be HxW or BxHxW")
    return value


def _distribution_ce_cosine(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probability = _as_batched_map(probability).float()
    target = _as_batched_map(target).to(probability).float()
    if probability.shape != target.shape:
        raise ValueError("predicted and target evidence maps must have equal shapes")
    probability_flat = probability.flatten(1).clamp_min(1e-12)
    probability_flat = probability_flat / probability_flat.sum(dim=1, keepdim=True)
    target_flat = target.flatten(1).clamp_min(0)
    target_flat = target_flat / target_flat.sum(dim=1, keepdim=True).clamp_min(1e-12)
    cross_entropy = (-(target_flat * probability_flat.log()).sum(dim=1)).mean()
    cosine = (
        1.0 - F.cosine_similarity(probability_flat, target_flat, dim=1)
    ).mean()
    return cross_entropy + cosine_weight * cosine, cross_entropy, cosine


def _forward_kl(current: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    current = _as_batched_map(current).float().flatten(1).clamp_min(1e-12)
    teacher = (
        _as_batched_map(teacher).detach().to(current).float().flatten(1).clamp_min(1e-12)
    )
    current = current / current.sum(dim=1, keepdim=True)
    teacher = teacher / teacher.sum(dim=1, keepdim=True)
    return (current * (current.log() - teacher.log())).sum(dim=1).mean()


@dataclass(frozen=True)
class PTEARecoveryLoss:
    total: torch.Tensor
    decisive_map: torch.Tensor
    decisive_cross_entropy: torch.Tensor
    decisive_cosine: torch.Tensor
    decisive_teacher_kl: torch.Tensor
    context_teacher_kl: torch.Tensor


def ptea_recovery_loss(
    paths: PTEATrainingPaths,
    decisive_bbox_target: torch.Tensor,
    teacher_decisive_probability: torch.Tensor,
    teacher_context_probability: torch.Tensor,
    *,
    cosine_weight: float = 0.30,
    decisive_distillation_weight: float = 0.10,
    context_distillation_weight: float = 0.25,
) -> PTEARecoveryLoss:
    """Train decisive evidence; distill, but never fabricate, context labels."""

    map_loss, cross_entropy, cosine = _distribution_ce_cosine(
        paths.decisive_probability,
        decisive_bbox_target,
        cosine_weight=cosine_weight,
    )
    decisive_kl = _forward_kl(
        paths.decisive_probability, teacher_decisive_probability
    )
    # Visual-CoT bbox supervision is intentionally absent here.  It is not a
    # context trace and must not be reused as one.
    context_kl = _forward_kl(
        paths.context_probability, teacher_context_probability
    )
    total = (
        map_loss
        + decisive_distillation_weight * decisive_kl
        + context_distillation_weight * context_kl
    )
    return PTEARecoveryLoss(
        total=total,
        decisive_map=map_loss,
        decisive_cross_entropy=cross_entropy,
        decisive_cosine=cosine,
        decisive_teacher_kl=decisive_kl,
        context_teacher_kl=context_kl,
    )


def _hungarian_minimize(cost: torch.Tensor) -> list[tuple[int, int]]:
    """Return a minimum-cost one-to-one assignment for a rectangular matrix."""

    if cost.ndim != 2 or not cost.numel():
        raise ValueError("matching cost must be a non-empty matrix")
    original_rows, original_columns = cost.shape
    transposed = original_rows > original_columns
    matrix = cost.detach().float().cpu()
    if transposed:
        matrix = matrix.t()
    rows, columns = matrix.shape
    # O(n*m*m) Hungarian potentials implementation, with rows <= columns.
    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for row in range(1, rows + 1):
        p[0] = row
        column0 = 0
        minimum = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = float(matrix[row0 - 1, column - 1]) - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    pairs = [(p[column] - 1, column - 1) for column in range(1, columns + 1) if p[column]]
    if transposed:
        pairs = [(column, row) for row, column in pairs]
    return sorted(pairs)


@dataclass(frozen=True)
class BoxMatching:
    matched_targets: torch.Tensor
    matched_mask: torch.Tensor
    target_indices: torch.Tensor


def match_anchors_to_targets(
    anchors: torch.Tensor,
    targets: torch.Tensor,
) -> BoxMatching:
    """Hungarian-match multiple targets; unmatched anchors keep identity."""

    if anchors.ndim != 2 or anchors.shape[-1] != 4:
        raise ValueError("anchors must have shape Ax4")
    if targets.ndim != 2 or targets.shape[-1] != 4 or not len(targets):
        raise ValueError("targets must have non-empty shape Tx4")
    if not torch.isfinite(anchors).all() or not torch.isfinite(targets).all():
        raise ValueError("boxes must be finite")
    cost = 1.0 - generalized_iou(anchors[:, None, :], targets[None, :, :])
    pairs = _hungarian_minimize(cost)
    matched_targets = anchors.detach().clone()
    matched_mask = torch.zeros(len(anchors), dtype=torch.bool, device=anchors.device)
    target_indices = torch.full(
        (len(anchors),), -1, dtype=torch.long, device=anchors.device
    )
    for anchor_index, target_index in pairs:
        matched_targets[anchor_index] = targets[target_index]
        matched_mask[anchor_index] = True
        target_indices[anchor_index] = target_index
    return BoxMatching(matched_targets, matched_mask, target_indices)


def h_safe_forward_detached(
    head: AdaptiveBoxHead,
    ptea_features: torch.Tensor,
) -> torch.Tensor:
    """Train H-Safe without allowing its losses to update PTEA."""

    return head(ptea_features.detach())


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return value[mask].mean()
    return value.new_zeros(())


def _masked_min(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return value[mask].min()
    return value.new_zeros(())


def _masked_max(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool(mask.any()):
        return value[mask].max()
    return value.new_zeros(())


@dataclass(frozen=True)
class HSafeGeometryLoss:
    total: torch.Tensor
    regression: torch.Tensor
    giou: torch.Tensor
    coverage: torch.Tensor
    matched_coverage: torch.Tensor
    union_coverage: torch.Tensor
    excess_area: torch.Tensor
    identity: torch.Tensor
    drift: torch.Tensor
    refined_boxes: torch.Tensor
    realizable_targets: torch.Tensor
    realizable_coverage_targets: torch.Tensor
    raw_target_min_area: torch.Tensor
    realizable_target_min_area: torch.Tensor
    maximum_refined_to_reference_area_ratio: torch.Tensor
    unrealizable_target_fraction: torch.Tensor
    supervised_decisive_anchors: int
    matching: BoxMatching


def h_safe_geometry_loss(
    head: AdaptiveBoxHead,
    ptea_features: torch.Tensor,
    anchors: torch.Tensor,
    targets: torch.Tensor,
    *,
    frozen_teacher_residuals: torch.Tensor | None = None,
    giou_weight: float = 2.0,
    coverage_weight: float = 0.5,
    excess_area_weight: float = 0.05,
    identity_weight: float = 0.10,
    drift_weight: float = 0.10,
    maximum_area_ratio: float = 1.8,
    supervised_decisive_anchors: int = H_SAFE_MAXIMUM_SUPERVISED_DECISIVE_ANCHORS,
) -> HSafeGeometryLoss:
    """Geometry-only H-Safe objective with realizable target supervision.

    Human evidence boxes can be much thinner than H-Safe's bounded action
    space (``minimum_side`` and bounded log-scale residuals).  Comparing a
    reachable region directly with such an unreachable target makes the old
    squared linear area ratio arbitrarily large and lets a few OCR boxes
    dominate the globally clipped gradient.  We therefore project each
    matched target through the *same* bounded residual parameterization used
    at inference.  GIoU and the area regularizer use that bounded-action
    projection.  This is a deterministic feasible surrogate, not a claim of
    globally closest projection under an unspecified box metric.  Visual-CoT
    boxes supervise every available decisive anchor (one or two); the final
    context anchor has no context-box label and stays under identity/teacher
    regularization.  Coverage projects every raw target, including targets
    beyond the available decisive slots, through its nearest/matched decisive
    anchor before computing overlap.  This preserves all evidence labels
    without retaining an inverse-tiny-area gradient path.

    Excess area is measured with Smooth-L1 in log-ratio space.  This retains a
    multiplicative size penalty without a near-zero-area singularity, while
    also preventing the largest realizable scale mismatch from dominating the
    joint update.
    """

    if maximum_area_ratio < 1.0 or not math.isfinite(maximum_area_ratio):
        raise ValueError("maximum_area_ratio must be finite and at least one")
    if supervised_decisive_anchors not in {1, 2}:
        raise ValueError("H-Safe recovery requires one or two decisive anchors")

    anchors = anchors.to(ptea_features).float()
    targets = targets.to(ptea_features).float()
    expected_regions = supervised_decisive_anchors + 1
    if len(anchors) != expected_regions or len(ptea_features) != expected_regions:
        raise ValueError(
            "H-Safe recovery requires one or two decisive anchors followed by "
            "exactly one context anchor"
        )
    decisive_anchors = anchors[:supervised_decisive_anchors]
    matching = match_anchors_to_targets(decisive_anchors, targets)
    residuals = h_safe_forward_detached(head, ptea_features)
    if residuals.shape != anchors.shape:
        raise ValueError("H-Safe residual and anchor shapes differ")
    refined = apply_residual(
        anchors,
        residuals,
        minimum_side=head.config.minimum_side,
        maximum_side=head.config.maximum_side,
    )
    decisive_residuals = residuals[:supervised_decisive_anchors]
    decisive_refined = refined[:supervised_decisive_anchors]
    mask = matching.matched_mask
    encoded_target = residual_target(
        decisive_anchors,
        matching.matched_targets,
        maximum_center_factor=head.config.maximum_center_factor,
        maximum_absolute_log_scale=head.config.maximum_absolute_log_scale,
    )
    realizable_targets = apply_residual(
        decisive_anchors,
        encoded_target,
        minimum_side=head.config.minimum_side,
        maximum_side=head.config.maximum_side,
    )
    # Hungarian matching supervises at most one target per anchor.  Coverage
    # must nevertheless retain every raw target, including T>A annotations.
    # Bind matched targets to their Hungarian anchor and assign any remaining
    # target to its nearest-GIoU anchor, then project all of them through the
    # same bounded action space.  This removes the 1/raw_tiny_area gradient
    # singularity without silently dropping unmatched evidence.
    anchor_target_cost = 1.0 - generalized_iou(
        decisive_anchors[:, None, :], targets[None, :, :]
    )
    coverage_anchor_indices = anchor_target_cost.argmin(dim=0)
    matched_anchor_indices = torch.arange(
        len(decisive_anchors), device=anchors.device
    )[mask]
    matched_target_indices = matching.target_indices[mask]
    coverage_anchor_indices = coverage_anchor_indices.scatter(
        0, matched_target_indices, matched_anchor_indices
    )
    coverage_anchors = decisive_anchors[coverage_anchor_indices]
    coverage_encoded_targets = residual_target(
        coverage_anchors,
        targets,
        maximum_center_factor=head.config.maximum_center_factor,
        maximum_absolute_log_scale=head.config.maximum_absolute_log_scale,
    )
    realizable_coverage_targets = apply_residual(
        coverage_anchors,
        coverage_encoded_targets,
        minimum_side=head.config.minimum_side,
        maximum_side=head.config.maximum_side,
    )
    regression_per_anchor = F.smooth_l1_loss(
        decisive_residuals,
        encoded_target,
        reduction="none",
        beta=0.15,
    ).mean(dim=-1)
    regression = _masked_mean(regression_per_anchor, mask)
    giou = _masked_mean(
        1.0 - generalized_iou(decisive_refined, realizable_targets), mask
    )
    matched_coverage = _masked_mean(
        1.0 - target_coverage(decisive_refined, realizable_targets), mask
    )
    # Every target must be covered by at least one refined region.  This keeps
    # multi-object labels separate rather than replacing them by one huge box.
    all_coverage = target_coverage(
        decisive_refined[:, None, :], realizable_coverage_targets[None, :, :]
    )
    union_coverage = (1.0 - all_coverage.max(dim=0).values).mean()
    coverage = matched_coverage + union_coverage
    raw_target_area = box_area(matching.matched_targets).clamp_min(1e-12)
    realizable_target_area = box_area(realizable_targets).clamp_min(1e-12)
    # A raw target can be either smaller or larger than the bounded H-Safe
    # action space.  The larger reference prevents penalizing geometry that is
    # already the bounded-action projection of that annotation.
    reference_area = torch.maximum(raw_target_area, realizable_target_area)
    refined_to_reference_area_ratio = (
        box_area(decisive_refined).clamp_min(1e-12) / reference_area
    )
    log_area_excess = F.relu(
        refined_to_reference_area_ratio.log() - math.log(maximum_area_ratio)
    )
    excess_per_anchor = F.smooth_l1_loss(
        log_area_excess,
        torch.zeros_like(log_area_excess),
        reduction="none",
        beta=1.0,
    )
    excess = _masked_mean(excess_per_anchor, mask)
    raw_all_target_area = box_area(targets).clamp_min(1e-12)
    realizable_all_target_area = box_area(realizable_coverage_targets).clamp_min(
        1e-12
    )
    target_projection_delta = (
        realizable_coverage_targets - targets
    ).abs().amax(dim=-1)
    unrealizable = (target_projection_delta > 1e-6).to(refined.dtype)
    geometry_supervised_mask = torch.zeros(
        len(residuals), dtype=torch.bool, device=residuals.device
    )
    geometry_supervised_mask[:supervised_decisive_anchors] = mask
    # Visual-CoT supplies decisive evidence boxes, not a context-box target.
    # The final context region receives identity/teacher regularization only;
    # it is never relabelled as decisive by Hungarian matching.
    identity = _masked_mean(
        residuals.square().mean(dim=-1), ~geometry_supervised_mask
    )
    teacher = (
        torch.zeros_like(residuals)
        if frozen_teacher_residuals is None
        else frozen_teacher_residuals.detach().to(residuals)
    )
    if teacher.shape != residuals.shape:
        raise ValueError("frozen H-Safe teacher residual shape mismatch")
    drift = (residuals - teacher).square().mean()
    total = (
        regression
        + giou_weight * giou
        + coverage_weight * coverage
        + excess_area_weight * excess
        + identity_weight * identity
        + drift_weight * drift
    )
    return HSafeGeometryLoss(
        total=total,
        regression=regression,
        giou=giou,
        coverage=coverage,
        matched_coverage=matched_coverage,
        union_coverage=union_coverage,
        excess_area=excess,
        identity=identity,
        drift=drift,
        refined_boxes=refined,
        realizable_targets=realizable_targets,
        realizable_coverage_targets=realizable_coverage_targets,
        raw_target_min_area=raw_all_target_area.min(),
        realizable_target_min_area=realizable_all_target_area.min(),
        maximum_refined_to_reference_area_ratio=_masked_max(
            refined_to_reference_area_ratio, mask
        ),
        unrealizable_target_fraction=unrealizable.mean(),
        supervised_decisive_anchors=supervised_decisive_anchors,
        matching=matching,
    )


def assert_router_phase_contract(
    lane: RecoveryLane,
    *,
    stage: str,
    start_epoch: float,
    parent_protocol_fingerprint: str | None,
) -> None:
    """Enforce the PTEA warm-up then matched H-Safe fork schedule."""

    if not lane.requires_router_aux:
        raise ValueError(f"{lane.name} is not a router-supervised lane")
    if stage == "A":
        if lane.train_h_safe:
            raise ValueError("stage A freezes H-Safe")
        if start_epoch != 0.0:
            raise ValueError("stage A must start at epoch 0")
        if parent_protocol_fingerprint is not None:
            raise ValueError("stage A cannot have a parent checkpoint")
        return
    if stage == "B":
        if lane.name not in {"E-PLB-C", "E-PHLB"}:
            raise ValueError("stage B is the matched E-PLB-C/E-PHLB fork")
        if start_epoch != ROUTER_STAGE_A_END_EPOCH:
            raise ValueError("stage B must fork exactly at epoch 0.5")
        if not parent_protocol_fingerprint:
            raise ValueError("stage B requires the shared stage-A checkpoint fingerprint")
        return
    raise ValueError(f"unknown router recovery stage: {stage}")


def router_protocol(
    *,
    lane: RecoveryLane,
    alignment: RouterAlignment,
    stage: str,
    start_epoch: float,
    parent_protocol_fingerprint: str | None,
    selector_sha256: str,
    h_safe_sha256: str,
    optimizer: Mapping[str, Any],
    losses: Mapping[str, Any],
) -> dict[str, Any]:
    assert_router_phase_contract(
        lane,
        stage=stage,
        start_epoch=start_epoch,
        parent_protocol_fingerprint=parent_protocol_fingerprint,
    )
    payload: dict[str, Any] = {
        "format_version": ROUTER_PROTOCOL_VERSION,
        "lane": lane.name,
        "stage": stage,
        "start_epoch": start_epoch,
        "stage_a_end_epoch": ROUTER_STAGE_A_END_EPOCH,
        "parent_protocol_fingerprint": parent_protocol_fingerprint,
        "alignment": alignment.protocol_fields(),
        "selector_sha256": selector_sha256,
        "h_safe_sha256": h_safe_sha256,
        "trainable": {
            "language_lora": lane.train_language_lora,
            "bridge": lane.train_bridge,
            "ptea": lane.train_ptea,
            "h_safe": lane.train_h_safe,
        },
        "frozen": ["host_vit", "host_merger", "context_need", "llm_base"],
        "gradient_contract": {
            "answer_ce": [
                name
                for enabled, name in (
                    (lane.train_language_lora, "language_lora"),
                    (lane.train_bridge, "bridge"),
                )
                if enabled
            ],
            "ptea": ["decisive_bbox_soft_map", "decisive_teacher", "context_teacher"],
            "h_safe": ["geometry", "identity", "teacher_drift"] if lane.train_h_safe else [],
            "ptea_to_discrete_decoder": "detached",
            "ptea_to_h_safe": "detached",
            "answer_ce_to_h_safe": "disabled",
        },
        "optimizer": dict(optimizer),
        "losses": dict(losses),
    }
    payload["protocol_fingerprint"] = canonical_fingerprint(payload)
    return payload


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def save_router_recovery_checkpoint(
    destination: Path,
    *,
    lane: RecoveryLane,
    selector: torch.nn.Module,
    h_safe: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    micro_step: int,
    optimizer_step: int,
    protocol: Mapping[str, Any],
) -> None:
    """Atomically save every trainable router state plus exact resume state."""

    expected_fingerprint = canonical_fingerprint(
        {key: value for key, value in protocol.items() if key != "protocol_fingerprint"}
    )
    if protocol.get("protocol_fingerprint") != expected_fingerprint:
        raise ValueError("router protocol fingerprint is stale or malformed")
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        payload = {
            "format_version": ROUTER_PROTOCOL_VERSION,
            "lane": lane.name,
            "protocol_fingerprint": expected_fingerprint,
            "micro_step": int(micro_step),
            "optimizer_step": int(optimizer_step),
            "selector_state_dict": selector.state_dict() if lane.train_ptea else None,
            "h_safe_state_dict": h_safe.state_dict() if lane.train_h_safe else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            **_rng_state(),
        }
        torch.save(payload, temporary / "training_state.pt")
        (temporary / "run_protocol.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_router_recovery_checkpoint(
    checkpoint: Path,
    *,
    lane: RecoveryLane,
    selector: torch.nn.Module,
    h_safe: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    expected_protocol_fingerprint: str,
) -> tuple[int, int]:
    """Restore a router lane only when lane and protocol are identical."""

    state = torch.load(
        checkpoint / "training_state.pt", map_location="cpu", weights_only=False
    )
    if state.get("format_version") != ROUTER_PROTOCOL_VERSION:
        raise ValueError("unsupported router recovery checkpoint")
    if state.get("lane") != lane.name:
        raise ValueError("router recovery checkpoint lane mismatch")
    if state.get("protocol_fingerprint") != expected_protocol_fingerprint:
        raise ValueError("router recovery checkpoint protocol mismatch")
    if lane.train_ptea:
        if state.get("selector_state_dict") is None:
            raise ValueError("router checkpoint lacks trainable PTEA state")
        selector.load_state_dict(state["selector_state_dict"], strict=True)
    elif state.get("selector_state_dict") is not None:
        raise ValueError("frozen-PTEA lane unexpectedly stores trainable PTEA state")
    if lane.train_h_safe:
        if state.get("h_safe_state_dict") is None:
            raise ValueError("router checkpoint lacks trainable H-Safe state")
        h_safe.load_state_dict(state["h_safe_state_dict"], strict=True)
    elif state.get("h_safe_state_dict") is not None:
        raise ValueError("frozen-H-Safe lane unexpectedly stores trainable H-Safe state")
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None:
        if state.get("scheduler_state_dict") is None:
            raise ValueError("router checkpoint lacks scheduler state")
        scheduler.load_state_dict(state["scheduler_state_dict"])
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
    if torch.cuda.is_available() and "torch_cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])
    return int(state["micro_step"]), int(state["optimizer_step"])


__all__ = [
    "AlignedRouterRow",
    "BoxMatching",
    "HSafeGeometryMask",
    "HSafeGeometryLoss",
    "H_SAFE_GEOMETRY_LOSS_VERSION",
    "H_SAFE_GEOMETRY_MASK_FORMAT_VERSION",
    "PTEARecoveryLoss",
    "PTEATrainingPaths",
    "ROUTER_PROTOCOL_VERSION",
    "ROUTER_STAGE_A_END_EPOCH",
    "RouterAlignment",
    "assert_router_phase_contract",
    "bbox_soft_map",
    "canonical_fingerprint",
    "h_safe_forward_detached",
    "h_safe_geometry_loss",
    "load_aligned_router_rows",
    "load_h_safe_geometry_mask",
    "load_router_recovery_checkpoint",
    "match_anchors_to_targets",
    "ptea_recovery_loss",
    "router_protocol",
    "save_router_recovery_checkpoint",
    "sha256_file",
    "split_ptea_training_paths",
]
