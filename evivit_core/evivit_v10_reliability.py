"""Training-only reliability features and deployment gate for EviViT-v10.

The gate consumes statistics that are available before QA decoding.  It never
uses an external benchmark answer or a model-correctness signal at inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Lightweight audit environments need no GPU stack.
    torch = None  # type: ignore[assignment]

from evivit_core.dense_evidence import decode_top_boxes


FEATURE_NAMES = [
    "candidate_score_entropy",
    "top1_score_margin",
    "top1_mass",
    "top1_mass_fraction",
    "top3_mass_mean",
    "top3_pairwise_iou",
    "candidate_spatial_dispersion",
    "top3_union_area_ratio",
    "top8_scale_diversity",
    "question_word_count",
    "feature_grid_log_aspect",
]


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(values.max())
    exponential = np.exp(np.clip(shifted, -60.0, 60.0))
    return exponential / exponential.sum()


def _normalized_entropy(probability: np.ndarray) -> float:
    if probability.size <= 1:
        return 0.0
    entropy = -float(np.sum(probability * np.log(probability + 1e-12)))
    return entropy / math.log(float(probability.size))


def box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    lx0, ly0, lx1, ly1 = map(float, left)
    rx0, ry0, rx1, ry1 = map(float, right)
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _intersection_area(boxes: Sequence[Sequence[float]]) -> float:
    x0 = max(float(box[0]) for box in boxes)
    y0 = max(float(box[1]) for box in boxes)
    x1 = min(float(box[2]) for box in boxes)
    y1 = min(float(box[3]) for box in boxes)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def top3_union_area_ratio(candidates: Sequence[dict[str, Any]]) -> float:
    """Exact normalized union area of the first three axis-aligned boxes."""

    boxes = [list(map(float, item["bbox"])) for item in candidates[:3]]
    if not boxes:
        return 0.0
    area = sum(_intersection_area([box]) for box in boxes)
    if len(boxes) >= 2:
        area -= sum(
            _intersection_area([boxes[left], boxes[right]])
            for left in range(len(boxes))
            for right in range(left + 1, len(boxes))
        )
    if len(boxes) == 3:
        area += _intersection_area(boxes)
    return area / 1_000_000.0


def candidate_features(
    candidates: Sequence[dict[str, Any]],
    *,
    feature_grid: Sequence[int],
    question: str,
    union_area_ratio: float | None = None,
) -> np.ndarray:
    """Convert one PTEA candidate distribution into the frozen gate features."""

    if len(candidates) < 3:
        raise ValueError("reliability gate requires at least three candidates")
    scores = np.asarray(
        [float(item["score"]) for item in candidates], dtype=np.float64
    )
    masses = np.asarray(
        [float(item["mass"]) for item in candidates], dtype=np.float64
    )
    probabilities = _softmax(scores)
    top = candidates[:3]
    centers = np.asarray(
        [
            [
                (float(item["bbox"][0]) + float(item["bbox"][2])) / 2000.0,
                (float(item["bbox"][1]) + float(item["bbox"][3])) / 2000.0,
            ]
            for item in candidates
        ],
        dtype=np.float64,
    )
    weighted_center = np.sum(probabilities[:, None] * centers, axis=0)
    dispersion = float(
        np.sum(
            probabilities
            * np.sum((centers - weighted_center[None, :]) ** 2, axis=1)
        )
    )
    pairwise_iou = np.mean(
        [
            box_iou(top[0]["bbox"], top[1]["bbox"]),
            box_iou(top[0]["bbox"], top[2]["bbox"]),
            box_iou(top[1]["bbox"], top[2]["bbox"]),
        ]
    )
    top8_scales = {
        round(float(item.get("scale", 0.0)), 6) for item in candidates[:8]
    }
    grid_height = max(float(feature_grid[0]), 1.0)
    grid_width = max(float(feature_grid[1]), 1.0)
    return np.asarray(
        [
            _normalized_entropy(probabilities),
            float(scores[0] - scores[1]),
            float(masses[0]),
            float(masses[0] / max(float(masses.sum()), 1e-12)),
            float(masses[:3].mean()),
            float(pairwise_iou),
            dispersion,
            (
                float(union_area_ratio)
                if union_area_ratio is not None
                else top3_union_area_ratio(candidates)
            ),
            float(len(top8_scales) / 5.0),
            float(len(str(question).split()) / 32.0),
            math.log(grid_height / grid_width),
        ],
        dtype=np.float64,
    )


def features_from_probability(
    probability: torch.Tensor,
    *,
    question: str,
    decode_scales: Sequence[float] = (0.2, 0.25, 0.35, 0.5, 0.67),
    decode_mass_power: float = 0.75,
    decode_area_penalty: float = 0.05,
    topk: int = 32,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Decode cheap numerical candidates and compute deployment features."""

    candidates = decode_top_boxes(
        probability.detach().float().cpu().numpy(),
        scales=decode_scales,
        topk=topk,
        mass_power=decode_mass_power,
        area_penalty=decode_area_penalty,
    )
    features = candidate_features(
        candidates,
        feature_grid=probability.shape,
        question=question,
    )
    return features, candidates


@dataclass(frozen=True)
class ReliabilityGate:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: float

    @classmethod
    def from_report(
        cls, report: dict[str, Any], *, artifact: str = "multifeature"
    ) -> "ReliabilityGate":
        payload = report["deployment_artifacts"][artifact]
        names = tuple(str(value) for value in payload["feature_names"])
        if any(name not in FEATURE_NAMES for name in names):
            raise ValueError(f"unsupported reliability features: {names}")
        return cls(
            feature_names=names,
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=float(payload["bias"]),
        )

    def predict(self, full_features: np.ndarray) -> float:
        indices = [FEATURE_NAMES.index(name) for name in self.feature_names]
        selected = np.asarray(full_features, dtype=np.float64)[indices]
        standardized = (selected - self.mean) / self.std
        logit = float(standardized @ self.weights + self.bias)
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))


def interpolate_fine_budget(
    reliability: float,
    *,
    minimum_tokens: int,
    maximum_tokens: int,
    quantum: int = 64,
) -> int:
    if minimum_tokens < 0 or maximum_tokens < minimum_tokens:
        raise ValueError("invalid fine-token range")
    if quantum <= 0:
        raise ValueError("token quantum must be positive")
    gate = max(0.0, min(1.0, float(reliability)))
    raw = minimum_tokens + gate * (maximum_tokens - minimum_tokens)
    rounded = int(round(raw / quantum) * quantum)
    return max(minimum_tokens, min(maximum_tokens, rounded))


def mix_with_uniform(
    probability: torch.Tensor, reliability: float
) -> torch.Tensor:
    """Flatten an unreliable map without discarding its available signal."""

    if torch is None or not isinstance(probability, torch.Tensor):
        source = np.asarray(probability, dtype=np.float64)
        source = np.clip(source, 0.0, None)
        source = source / max(float(source.sum()), 1e-12)
        uniform = np.full_like(source, 1.0 / source.size)
        gate = max(0.0, min(1.0, float(reliability)))
        mixed = gate * source + (1.0 - gate) * uniform
        return mixed / max(float(mixed.sum()), 1e-12)  # type: ignore[return-value]
    source = probability.float().clamp_min(0)
    source = source / source.sum().clamp_min(1e-12)
    uniform = torch.full_like(source, 1.0 / source.numel())
    gate = max(0.0, min(1.0, float(reliability)))
    mixed = gate * source + (1.0 - gate) * uniform
    return mixed / mixed.sum().clamp_min(1e-12)
