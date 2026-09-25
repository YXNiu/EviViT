"""Trace-supervised continuous region geometry for EviViT-NativeScale.

This module contains only deterministic geometry and feature extraction.  It
never uses a human box to generate an inference-time region; human views are
used as train-only targets and offline audit labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn


Box = tuple[float, float, float, float]


def normalize_box(values: Sequence[float]) -> Box:
    if len(values) != 4:
        raise ValueError("box must contain four values")
    numbers = [float(value) for value in values]
    scale = 1000.0 if max(abs(value) for value in numbers) > 1.0 else 1.0
    x1, y1, x2, y2 = (value / scale for value in numbers)
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive area")
    return x1, y1, x2, y2


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_center(box: Box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def box_size(box: Box) -> tuple[float, float]:
    return box[2] - box[0], box[3] - box[1]


def intersection_area(first: Box, second: Box) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )


def target_coverage(region: Box, target: Box) -> float:
    return intersection_area(region, target) / max(box_area(target), 1e-12)


def box_iou(first: Box, second: Box) -> float:
    overlap = intersection_area(first, second)
    return overlap / max(box_area(first) + box_area(second) - overlap, 1e-12)


def union_box(*boxes: Box) -> Box:
    if not boxes:
        raise ValueError("at least one box is required")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def expand_box(box: Box, factor: float) -> Box:
    if factor < 1.0:
        raise ValueError("expansion factor must be at least one")
    cx, cy = box_center(box)
    width, height = box_size(box)
    return (
        max(0.0, cx - width * factor / 2.0),
        max(0.0, cy - height * factor / 2.0),
        min(1.0, cx + width * factor / 2.0),
        min(1.0, cy + height * factor / 2.0),
    )


def desired_human_view(
    last_zoom: Box | None,
    final_boxes: Sequence[Box],
    *,
    minimum_core_coverage: float = 0.80,
    final_context_factor: float = 1.25,
) -> tuple[Box | None, Box | None, str]:
    """Return the train target and the smallest final core.

    Last zoom is the primary target.  A final box is only added when the last
    zoom does not already cover it sufficiently.  If last zoom is unavailable,
    the smallest final box receives a modest context margin.
    """

    core = min(final_boxes, key=box_area) if final_boxes else None
    if last_zoom is not None:
        if core is not None and target_coverage(last_zoom, core) < minimum_core_coverage:
            return union_box(last_zoom, core), core, "last_zoom_union_final"
        return last_zoom, core, "last_zoom"
    if core is not None:
        return expand_box(core, final_context_factor), core, "final_context_fallback"
    return None, None, "missing"


def centered_cover_box(
    center: tuple[float, float],
    target: Box,
    *,
    context_margin: float = 0.0,
) -> Box:
    """Smallest clipped box centred at ``center`` that covers ``target``."""

    if context_margin < 0:
        raise ValueError("context margin must be non-negative")
    cx, cy = center
    half_width = max(abs(cx - target[0]), abs(target[2] - cx))
    half_height = max(abs(cy - target[1]), abs(target[3] - cy))
    half_width *= 1.0 + context_margin
    half_height *= 1.0 + context_margin
    return (
        max(0.0, cx - half_width),
        max(0.0, cy - half_height),
        min(1.0, cx + half_width),
        min(1.0, cy + half_height),
    )


@dataclass(frozen=True)
class AnchorMatch:
    anchor_index: int
    anchor_box: Box
    target_box: Box
    centered_box: Box
    center_offset_x: float
    center_offset_y: float
    normalized_center_distance: float
    log_width_ratio: float
    log_height_ratio: float
    required_width: float
    required_height: float
    centered_area_ratio: float
    scale_only_feasible: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "anchor_index": self.anchor_index,
            "anchor_box": list(self.anchor_box),
            "target_box": list(self.target_box),
            "centered_box": list(self.centered_box),
            "center_offset_x": self.center_offset_x,
            "center_offset_y": self.center_offset_y,
            "normalized_center_distance": self.normalized_center_distance,
            "log_width_ratio": self.log_width_ratio,
            "log_height_ratio": self.log_height_ratio,
            "required_width": self.required_width,
            "required_height": self.required_height,
            "centered_area_ratio": self.centered_area_ratio,
            "scale_only_feasible": self.scale_only_feasible,
        }


def match_target_to_anchors(
    anchors: Sequence[Box],
    target: Box,
    *,
    context_margin: float = 0.10,
    maximum_centered_area: float = 0.25,
    maximum_aspect_ratio: float = 4.0,
) -> AnchorMatch:
    if not anchors:
        raise ValueError("at least one anchor is required")
    tx, ty = box_center(target)
    tw, th = box_size(target)
    candidates: list[tuple[tuple[float, float, int], AnchorMatch]] = []
    for index, anchor in enumerate(anchors):
        ax, ay = box_center(anchor)
        aw, ah = box_size(anchor)
        centered = centered_cover_box(
            (ax, ay), target, context_margin=context_margin
        )
        cw, ch = box_size(centered)
        required_width = min(
            1.0,
            2.0
            * max(abs(ax - target[0]), abs(target[2] - ax))
            * (1.0 + context_margin),
        )
        required_height = min(
            1.0,
            2.0
            * max(abs(ay - target[1]), abs(target[3] - ay))
            * (1.0 + context_margin),
        )
        area = box_area(centered)
        aspect = max(cw / max(ch, 1e-12), ch / max(cw, 1e-12))
        dx, dy = tx - ax, ty - ay
        normalized_distance = math.sqrt(
            (dx / max(tw, aw, 1e-6)) ** 2
            + (dy / max(th, ah, 1e-6)) ** 2
        )
        match = AnchorMatch(
            anchor_index=index,
            anchor_box=anchor,
            target_box=target,
            centered_box=centered,
            center_offset_x=dx,
            center_offset_y=dy,
            normalized_center_distance=normalized_distance,
            log_width_ratio=math.log(max(required_width, 1e-6) / max(aw, 1e-6)),
            log_height_ratio=math.log(max(required_height, 1e-6) / max(ah, 1e-6)),
            required_width=required_width,
            required_height=required_height,
            centered_area_ratio=area,
            scale_only_feasible=(
                area <= maximum_centered_area
                and aspect <= maximum_aspect_ratio
                and required_width <= 0.80
                and required_height <= 0.80
                and target_coverage(centered, target) >= 0.999
            ),
        )
        # Prefer the smallest required centred view, then the closest centre,
        # then the earlier PTEA rank for deterministic matching.
        candidates.append(((area, normalized_distance, index), match))
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _map_patch(probability: np.ndarray, box: Box) -> np.ndarray:
    height, width = probability.shape
    x1 = max(0, min(width - 1, int(math.floor(box[0] * width))))
    y1 = max(0, min(height - 1, int(math.floor(box[1] * height))))
    x2 = max(x1 + 1, min(width, int(math.ceil(box[2] * width))))
    y2 = max(y1 + 1, min(height, int(math.ceil(box[3] * height))))
    return probability[y1:y2, x1:x2]


def anchor_feature_vector(
    probability: np.ndarray,
    region: dict[str, Any],
    *,
    image_size: Sequence[int],
    question_length: int,
) -> np.ndarray:
    """Return inference-available features for one frozen PTEA anchor."""

    distribution = np.asarray(probability, dtype=np.float64)
    distribution = np.clip(distribution, 0.0, None)
    distribution /= max(float(distribution.sum()), 1e-12)
    box = normalize_box(region["bbox"])
    patch = _map_patch(distribution, box)
    mass = float(patch.sum())
    positive = patch[patch > 0]
    if positive.size > 1:
        local = positive / positive.sum()
        entropy = float(-(local * np.log(local + 1e-12)).sum()) / math.log(
            positive.size
        )
    else:
        entropy = 0.0
    width, height = box_size(box)
    cx, cy = box_center(box)
    image_width, image_height = (float(value) for value in image_size)
    features = np.asarray(
        [
            float(region.get("source_rank", 1)) / 3.0,
            float(region.get("score", 0.0)),
            cx,
            cy,
            width,
            height,
            box_area(box),
            mass,
            math.log1p(mass / max(box_area(box), 1e-8)),
            float(patch.max()) if patch.size else 0.0,
            entropy,
            float(distribution.max()),
            float(-(distribution * np.log(distribution + 1e-12)).sum())
            / math.log(distribution.size),
            math.log1p(image_width * image_height) / 20.0,
            math.log(max(image_width, 1.0) / max(image_height, 1.0)),
            min(float(question_length), 64.0) / 64.0,
        ],
        dtype=np.float32,
    )
    return features


class TraceScaleHead(nn.Module):
    """Tiny identity-initialized predictor of log width/height residuals."""

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        maximum_absolute_log_scale: float = 1.20,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.maximum_absolute_log_scale = float(maximum_absolute_log_scale)
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.network(features)
        return self.maximum_absolute_log_scale * torch.tanh(residual)


def scaled_centered_box(
    anchor: Box,
    log_width_ratio: float,
    log_height_ratio: float,
    *,
    minimum_side: float = 0.04,
    maximum_side: float = 0.80,
) -> Box:
    """Resize an anchor around its frozen centre and clip to the image."""

    if not 0 < minimum_side <= maximum_side <= 1:
        raise ValueError("side bounds must satisfy 0 < minimum <= maximum <= 1")
    cx, cy = box_center(anchor)
    width, height = box_size(anchor)
    width = min(maximum_side, max(minimum_side, width * math.exp(log_width_ratio)))
    height = min(
        maximum_side, max(minimum_side, height * math.exp(log_height_ratio))
    )
    return (
        max(0.0, cx - width / 2.0),
        max(0.0, cy - height / 2.0),
        min(1.0, cx + width / 2.0),
        min(1.0, cy + height / 2.0),
    )


def load_tracescale_head(
    checkpoint_path: str,
    *,
    device: str | torch.device = "cpu",
) -> tuple[TraceScaleHead, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("model_config", {})
    head = TraceScaleHead(
        input_dim=int(config.get("input_dim", 16)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        maximum_absolute_log_scale=float(
            config.get("maximum_absolute_log_scale", 1.20)
        ),
    )
    head.load_state_dict(checkpoint["state_dict"])
    head.to(device).eval()
    return head, checkpoint


def predict_region_geometry(
    head: TraceScaleHead,
    probability: np.ndarray,
    regions: Sequence[dict[str, Any]],
    *,
    image_size: Sequence[int],
    question_length: int,
    positive_only: bool = True,
    strength: float = 1.0,
) -> list[dict[str, Any]]:
    """Predict auditable, centre-preserving geometry for PTEA anchors.

    ``positive_only`` implements TraceScale-Safe: the learned residual may
    enlarge an existing PTEA region, but it may never remove evidence already
    visible in the frozen region.  ``strength`` is an inference-time ablation;
    one reproduces the learned residual and zero is an exact geometric control.
    """

    if strength < 0:
        raise ValueError("TraceScale strength must be non-negative")
    if not regions:
        return []
    feature_rows = [
        anchor_feature_vector(
            probability,
            region,
            image_size=image_size,
            question_length=question_length,
        )
        for region in regions
    ]
    device = next(head.parameters()).device
    features = torch.tensor(np.stack(feature_rows), dtype=torch.float32, device=device)
    with torch.inference_mode():
        raw = head(features).float().cpu().numpy()
    output: list[dict[str, Any]] = []
    for region, values in zip(regions, raw):
        raw_width, raw_height = (float(values[0]), float(values[1]))
        applied_width = raw_width * strength
        applied_height = raw_height * strength
        if positive_only:
            applied_width = max(0.0, applied_width)
            applied_height = max(0.0, applied_height)
        original = normalize_box(region["bbox"])
        adjusted = scaled_centered_box(
            original,
            applied_width,
            applied_height,
        )
        output.append(
            {
                "original_bbox": list(original),
                "adjusted_bbox": list(adjusted),
                "raw_log_width_ratio": raw_width,
                "raw_log_height_ratio": raw_height,
                "applied_log_width_ratio": applied_width,
                "applied_log_height_ratio": applied_height,
                "original_area_ratio": box_area(original),
                "adjusted_area_ratio": box_area(adjusted),
            }
        )
    return output
