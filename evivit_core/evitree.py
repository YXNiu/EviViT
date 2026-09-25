"""Parent-preserving evidence-adaptive visual budgeting for EviTree.

This module contains the parts of EviTree that can be audited without loading
Qwen3-VL:

* question-conditioned map statistics;
* human-trace-derived evidence-need targets;
* a protected-global continuous visual-token budget;
* a deterministic hierarchical fine-region planner.

The complete low-resolution global stream is never represented by this tree
and must always be retained by the caller.  EviTree plans only *additional*
high-resolution evidence tokens.  This distinction is what separates EviTree
from the retired TraceTree prototype, which replaced coarse parents by pooled
children.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn


NEED_STAT_NAMES = (
    "log_source_pixels_over_4m",
    "absolute_log_aspect_ratio",
    "map_peak",
    "map_entropy",
    "map_top_1pct_mass",
    "map_top_5pct_mass",
    "map_spatial_spread",
    "map_total_variation",
    "map_mode_count_normalized",
    "question_length_normalized",
    "ptea_native_disagreement",
    "ptea_reliability",
)

SCALAR_NEED_FEATURE_NAMES = (
    "log_source_pixels_over_4m",
    "absolute_log_aspect_ratio",
    "question_length_normalized",
    "map_peak",
    "map_entropy",
    "map_top_1pct_mass",
    "map_top_5pct_mass",
    "map_spatial_spread",
    "map_total_variation",
    "map_mode_count_normalized",
)

SPLIT_POLICY_FEATURE_NAMES = (
    "depth_normalized",
    "area_ratio",
    "center_x",
    "center_y",
    "node_evidence_mass",
    "node_density_log",
    "node_peak_log",
    "node_entropy",
    "child_mass_nw",
    "child_mass_ne",
    "child_mass_sw",
    "child_mass_se",
    "child_entropy",
    "child_max_share",
    "child_margin",
)


def _normalise_probability(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("probability must be a non-empty 2D array")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("probability must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        return np.full_like(values, 1.0 / values.size)
    return values / total


def _top_mass(values: np.ndarray, fraction: float) -> float:
    count = max(1, int(math.ceil(values.size * fraction)))
    return float(np.partition(values.reshape(-1), -count)[-count:].sum())


def _mode_count(probability: np.ndarray) -> int:
    """Count separated local peaks above half of the global maximum."""

    values = np.asarray(probability, dtype=np.float64)
    threshold = 0.5 * float(values.max(initial=0.0))
    if threshold <= 0:
        return 0
    padded = np.pad(values, 1, mode="constant", constant_values=-np.inf)
    peaks = np.ones_like(values, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbour = padded[
                1 + dy : 1 + dy + values.shape[0],
                1 + dx : 1 + dx + values.shape[1],
            ]
            peaks &= values >= neighbour
    return int(np.count_nonzero(peaks & (values >= threshold)))


def evidence_map_statistics(probability: np.ndarray) -> dict[str, float]:
    """Return scale-independent statistics of one evidence distribution."""

    distribution = _normalise_probability(probability)
    flat = distribution.reshape(-1)
    positive = flat[flat > 0]
    entropy = -float(np.sum(positive * np.log(positive + 1e-12)))
    entropy /= math.log(flat.size) if flat.size > 1 else 1.0
    yy, xx = np.mgrid[0 : distribution.shape[0], 0 : distribution.shape[1]]
    x = (xx + 0.5) / distribution.shape[1]
    y = (yy + 0.5) / distribution.shape[0]
    center_x = float((distribution * x).sum())
    center_y = float((distribution * y).sum())
    # The largest possible squared distance on the unit canvas is two.
    spread = float(
        (
            distribution
            * ((x - center_x) ** 2 + (y - center_y) ** 2)
        ).sum()
        / 0.5
    )
    uniform = 1.0 / flat.size
    total_variation = 0.5 * float(np.abs(flat - uniform).sum())
    return {
        "map_peak": float(flat.max(initial=0.0)),
        "map_entropy": float(np.clip(entropy, 0.0, 1.0)),
        "map_top_1pct_mass": _top_mass(distribution, 0.01),
        "map_top_5pct_mass": _top_mass(distribution, 0.05),
        "map_spatial_spread": float(np.clip(spread, 0.0, 1.0)),
        "map_total_variation": float(np.clip(total_variation, 0.0, 1.0)),
        "map_mode_count_normalized": min(1.0, _mode_count(distribution) / 8.0),
    }


def need_feature_vector(
    probability: np.ndarray,
    *,
    width: int,
    height: int,
    question: str,
    ptea_native_disagreement: float = 0.0,
    ptea_reliability: float = 1.0,
) -> np.ndarray:
    """Build the frozen scalar input vector for ``EvidenceNeedHead``."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    statistics = evidence_map_statistics(probability)
    values = {
        "log_source_pixels_over_4m": float(
            np.clip(math.log(width * height / float(4 * 1024 * 1024)), -4.0, 4.0)
            / 4.0
        ),
        "absolute_log_aspect_ratio": float(
            np.clip(abs(math.log(width / height)), 0.0, 4.0) / 4.0
        ),
        **statistics,
        "question_length_normalized": min(
            1.0, len(str(question).split()) / 32.0
        ),
        "ptea_native_disagreement": float(
            np.clip(ptea_native_disagreement, 0.0, 1.0)
        ),
        "ptea_reliability": float(np.clip(ptea_reliability, 0.0, 1.0)),
    }
    return np.asarray([values[name] for name in NEED_STAT_NAMES], dtype=np.float32)


def scalar_need_feature_vector(
    probability: np.ndarray,
    *,
    width: int,
    height: int,
    question: str,
) -> np.ndarray:
    """Build the exact deployment vector used by ``ScalarEvidenceNeedHead``."""

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    statistics = evidence_map_statistics(probability)
    values = {
        "log_source_pixels_over_4m": float(
            np.clip(math.log(width * height / float(4 * 1024 * 1024)), -4.0, 4.0)
            / 4.0
        ),
        "absolute_log_aspect_ratio": float(
            np.clip(abs(math.log(width / height)), 0.0, 4.0) / 4.0
        ),
        "question_length_normalized": min(
            1.0, len(str(question).split()) / 32.0
        ),
        **statistics,
    }
    return np.asarray(
        [values[name] for name in SCALAR_NEED_FEATURE_NAMES],
        dtype=np.float32,
    )


class EvidenceNeedHead(nn.Module):
    """Small trainable head that predicts continuous high-resolution need.

    The question input is the contextual summary already produced by PTEA.
    The scalar branch holds image-size and evidence-map statistics.  Qwen-ViT,
    PTEA, and the language model remain frozen in EviTree-A.
    """

    def __init__(
        self,
        question_dim: int,
        *,
        statistic_dim: int = len(NEED_STAT_NAMES),
        hidden_dim: int = 128,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if min(question_dim, statistic_dim, hidden_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        self.question_dim = int(question_dim)
        self.statistic_dim = int(statistic_dim)
        self.question_projection = nn.Sequential(
            nn.LayerNorm(question_dim),
            nn.Linear(question_dim, hidden_dim),
            nn.SiLU(),
        )
        self.statistic_projection = nn.Sequential(
            nn.LayerNorm(statistic_dim),
            nn.Linear(statistic_dim, hidden_dim // 2),
            nn.SiLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, question_summary: torch.Tensor, statistics: torch.Tensor
    ) -> torch.Tensor:
        if question_summary.shape[-1] != self.question_dim:
            raise ValueError("question summary has the wrong final dimension")
        if statistics.shape[-1] != self.statistic_dim:
            raise ValueError("statistics have the wrong final dimension")
        question = self.question_projection(question_summary)
        scalar = self.statistic_projection(statistics)
        return torch.sigmoid(self.output(torch.cat([question, scalar], dim=-1))).squeeze(-1)


class ScalarEvidenceNeedHead(nn.Module):
    """Deployment-first Need head for already question-conditioned map features.

    PTEA has already fused the question into the evidence map, so this compact
    lane does not require a second text encoder.  The larger
    :class:`EvidenceNeedHead` remains available as a later explicit-question
    ablation.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if min(feature_dim, hidden_dim) <= 0:
            raise ValueError("feature and hidden dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_scale", torch.ones(feature_dim))
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def set_normalization(
        self, feature_mean: torch.Tensor, feature_scale: torch.Tensor
    ) -> None:
        if feature_mean.shape != self.feature_mean.shape:
            raise ValueError("feature mean has the wrong shape")
        if feature_scale.shape != self.feature_scale.shape:
            raise ValueError("feature scale has the wrong shape")
        self.feature_mean.copy_(feature_mean)
        self.feature_scale.copy_(feature_scale.clamp_min(1e-6))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError("features have the wrong final dimension")
        normalized = (features.float() - self.feature_mean) / self.feature_scale
        return torch.sigmoid(self.network(normalized).squeeze(-1))


class HierarchicalSplitPolicyHead(nn.Module):
    """Rank dyadic parent nodes by the value of one more spatial split.

    The head only sees statistics of the already question-conditioned PTEA
    probability map.  Human trace maps supervise it during training, but are
    never required at inference time.
    """

    def __init__(
        self,
        feature_dim: int = len(SPLIT_POLICY_FEATURE_NAMES),
        *,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if min(feature_dim, hidden_dim) <= 0:
            raise ValueError("feature and hidden dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_scale", torch.ones(feature_dim))
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def set_normalization(
        self, feature_mean: torch.Tensor, feature_scale: torch.Tensor
    ) -> None:
        if feature_mean.shape != self.feature_mean.shape:
            raise ValueError("feature mean has the wrong shape")
        if feature_scale.shape != self.feature_scale.shape:
            raise ValueError("feature scale has the wrong shape")
        self.feature_mean.copy_(feature_mean)
        self.feature_scale.copy_(feature_scale.clamp_min(1e-6))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError("features have the wrong final dimension")
        normalized = (features.float() - self.feature_mean) / self.feature_scale
        return torch.sigmoid(self.network(normalized).squeeze(-1))


def _array_bounds(
    shape: tuple[int, int],
    box_xyxy: Sequence[float],
) -> tuple[int, int, int, int]:
    height, width = shape
    x1, y1, x2, y2 = map(float, box_xyxy)
    ix1 = min(width - 1, max(0, int(round(x1 * width))))
    iy1 = min(height - 1, max(0, int(round(y1 * height))))
    ix2 = min(width, max(ix1 + 1, int(round(x2 * width))))
    iy2 = min(height, max(iy1 + 1, int(round(y2 * height))))
    return ix1, iy1, ix2, iy2


def split_policy_feature_vector(
    probability: np.ndarray,
    *,
    box_xyxy: Sequence[float],
    depth: int,
    maximum_depth: int = 5,
) -> np.ndarray:
    """Describe one possible tree split using frozen PTEA evidence only."""

    if len(box_xyxy) != 4:
        raise ValueError("box_xyxy must contain four normalized coordinates")
    if not 0 <= depth < maximum_depth:
        raise ValueError("depth must identify a splittable tree node")
    distribution = _normalise_probability(probability)
    x1, y1, x2, y2 = map(float, box_xyxy)
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError("box_xyxy must be a valid normalized box")
    ix1, iy1, ix2, iy2 = _array_bounds(distribution.shape, box_xyxy)
    patch = distribution[iy1:iy2, ix1:ix2]
    node_mass = float(patch.sum())
    area = (x2 - x1) * (y2 - y1)
    local = patch.reshape(-1)
    if node_mass > 0:
        local = local / node_mass
        positive = local[local > 0]
        entropy = -float(np.sum(positive * np.log(positive + 1e-12)))
        entropy /= math.log(local.size) if local.size > 1 else 1.0
        peak_ratio = float(local.max(initial=0.0) * local.size)
    else:
        entropy = 1.0
        peak_ratio = 1.0

    middle_x = 0.5 * (x1 + x2)
    middle_y = 0.5 * (y1 + y2)
    children = (
        (x1, y1, middle_x, middle_y),
        (middle_x, y1, x2, middle_y),
        (x1, middle_y, middle_x, y2),
        (middle_x, middle_y, x2, y2),
    )
    child_masses = []
    for child in children:
        cx1, cy1, cx2, cy2 = _array_bounds(distribution.shape, child)
        child_masses.append(float(distribution[cy1:cy2, cx1:cx2].sum()))
    child_shares = np.asarray(child_masses, dtype=np.float64)
    if child_shares.sum() > 0:
        child_shares /= child_shares.sum()
    else:
        child_shares[:] = 0.25
    positive_children = child_shares[child_shares > 0]
    child_entropy = -float(
        np.sum(positive_children * np.log(positive_children + 1e-12))
    ) / math.log(4.0)
    ordered = np.sort(child_shares)[::-1]
    density_ratio = node_mass / max(area, 1e-12)
    values = (
        depth / max(1, maximum_depth - 1),
        area,
        0.5 * (x1 + x2),
        0.5 * (y1 + y2),
        node_mass,
        math.log1p(max(0.0, density_ratio)),
        math.log1p(max(0.0, peak_ratio)),
        float(np.clip(entropy, 0.0, 1.0)),
        *child_shares.tolist(),
        float(np.clip(child_entropy, 0.0, 1.0)),
        float(ordered[0]),
        float(ordered[0] - ordered[1]),
    )
    return np.asarray(values, dtype=np.float32)


def load_hierarchical_split_policy_head(
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[HierarchicalSplitPolicyHead, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    if checkpoint.get("version") != "evitree_v29_split_policy_v1":
        raise ValueError("unsupported EviTree split-policy checkpoint version")
    config = dict(checkpoint["model_config"])
    if tuple(config["feature_names"]) != SPLIT_POLICY_FEATURE_NAMES:
        raise ValueError(
            "split-policy feature contract differs from deployment contract"
        )
    model = HierarchicalSplitPolicyHead(
        feature_dim=int(config["feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), checkpoint


def load_scalar_evidence_need_head(
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[ScalarEvidenceNeedHead, dict[str, Any]]:
    """Load a versioned Need-head checkpoint and validate its feature ABI."""

    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    if checkpoint.get("version") != "evitree_v29_scalar_need_head_v1":
        raise ValueError("unsupported EviTree Need-head checkpoint version")
    config = dict(checkpoint["model_config"])
    if tuple(config["feature_names"]) != SCALAR_NEED_FEATURE_NAMES:
        raise ValueError(
            "Need-head feature contract differs from the deployment contract"
        )
    model = ScalarEvidenceNeedHead(
        feature_dim=int(config["feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), checkpoint


def continuous_fine_token_budget(
    evidence_need: float,
    *,
    minimum_tokens: int = 512,
    maximum_tokens: int = 4096,
    alignment: int = 64,
) -> int:
    """Map continuous evidence need to an aligned additional fine budget."""

    if minimum_tokens <= 0 or maximum_tokens < minimum_tokens:
        raise ValueError("invalid fine-token interval")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    need = float(np.clip(evidence_need, 0.0, 1.0))
    raw = minimum_tokens + need * (maximum_tokens - minimum_tokens)
    aligned = int(round(raw / alignment) * alignment)
    return max(minimum_tokens, min(maximum_tokens, aligned))


def continuous_native_global_token_budget(
    width: int,
    height: int,
    *,
    base_tokens: int = 0,
    native_fraction: float = 0.1875,
    minimum_tokens: int = 512,
    maximum_tokens: int = 3072,
    maximum_pixels: int = 16 * 1024 * 1024,
    merged_patch_stride: int = 32,
    alignment: int = 64,
) -> int:
    """Map source-image size to a continuous protected global-token budget.

    Qwen first limits the image to ``maximum_pixels`` and then creates one
    merged visual token per roughly ``merged_patch_stride**2`` pixels.  This
    function keeps that native size dependence but allocates only a controlled
    fraction to EviViT's low-resolution global stream. ``base_tokens`` is a
    continuous safety intercept, not a discrete image-size tier: every image
    still receives a size-dependent increment.  The final integer alignment is
    an implementation requirement, not a discrete budget tier.
    """

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if base_tokens < 0:
        raise ValueError("base_tokens must be non-negative")
    if not 0 < native_fraction <= 1:
        raise ValueError("native_fraction must lie in (0, 1]")
    if minimum_tokens <= 0 or maximum_tokens < minimum_tokens:
        raise ValueError("invalid global-token interval")
    if maximum_pixels <= 0 or merged_patch_stride <= 0 or alignment <= 0:
        raise ValueError("pixel, stride, and alignment values must be positive")
    effective_pixels = min(int(width) * int(height), int(maximum_pixels))
    native_tokens = effective_pixels / float(merged_patch_stride**2)
    raw = float(base_tokens) + native_fraction * native_tokens
    aligned = int(round(raw / alignment) * alignment)
    return max(minimum_tokens, min(maximum_tokens, aligned))


def _policy_box_area(box: Sequence[float] | None) -> float | None:
    if box is None or len(box) != 4:
        return None
    x1, y1, x2, y2 = map(float, box)
    width = max(0.0, min(1000.0, x2) - max(0.0, x1))
    height = max(0.0, min(1000.0, y2) - max(0.0, y1))
    if width <= 0 or height <= 0:
        return None
    return width * height / 1_000_000.0


def human_evidence_need_target(
    *,
    width: int,
    height: int,
    last_zoom_box: Sequence[float] | None,
    final_boxes: Iterable[Sequence[float]],
    zoom_episodes: int,
    reset_count: int,
) -> dict[str, float]:
    """Create an auditable need target from the human process.

    Last zoom and final boxes jointly receive 60% of the nominal weight.  This
    follows the established annotation semantics: last zoom is normally the
    first readable view, while the final box is the most precise evidence cue.
    Missing components are removed and the remaining active weights are
    renormalised instead of being silently treated as zero need.
    """

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    last_area = _policy_box_area(last_zoom_box)
    final_areas = [
        area
        for area in (_policy_box_area(box) for box in final_boxes)
        if area is not None
    ]
    components: dict[str, tuple[float, float] | None] = {
        "last_zoom_resolution_need": (
            (0.35, 1.0 - math.sqrt(last_area))
            if last_area is not None
            else None
        ),
        "final_box_resolution_need": (
            (0.25, 1.0 - min(final_areas) ** 0.25)
            if final_areas
            else None
        ),
        "zoom_depth_need": (0.15, min(1.0, max(0, zoom_episodes) / 5.0)),
        "source_size_need": (
            0.15,
            1.0
            / (
                1.0
                + math.exp(
                    -math.log(width * height / float(4 * 1024 * 1024)) / 1.5
                )
            ),
        ),
        "search_ambiguity_need": (
            0.10,
            min(1.0, max(0, reset_count) / 2.0),
        ),
    }
    active = {
        name: value
        for name, value in components.items()
        if value is not None
    }
    denominator = sum(weight for weight, _ in active.values())
    target = (
        sum(weight * value for weight, value in active.values()) / denominator
        if denominator > 0
        else 0.0
    )
    output = {
        name: float(value[1]) if value is not None else float("nan")
        for name, value in components.items()
    }
    output["active_weight"] = denominator
    output["target"] = float(np.clip(target, 0.0, 1.0))
    output["last_zoom_area_ratio"] = (
        float(last_area) if last_area is not None else float("nan")
    )
    output["final_box_min_area_ratio"] = (
        float(min(final_areas)) if final_areas else float("nan")
    )
    return output


@dataclass(frozen=True)
class EviTreeBudget:
    evidence_need: float
    global_tokens: int
    fine_tokens: int
    total_tokens: int
    alignment: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeRatioBudget:
    """Global/Fine budgets derived from Qwen's own native token demand."""

    native_tokens: int
    resized_height: int
    resized_width: int
    global_ratio: float
    fine_ratio: float
    global_tokens: int
    fine_tokens: int
    total_tokens: int
    alignment: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NativeNeedSpreadBudget:
    """Continuous, image- and problem-conditioned Global/Fine budget.

    ``native_tokens`` is the number of merged visual tokens that unmodified
    Qwen would use under the same processor pixel limits.  The evidence-need
    scalar controls the amount of additional Fine reading, while the spatial
    spread of the question-conditioned evidence map controls how much Global
    context must be protected.  Neither stream is selected from a discrete
    template.
    """

    native_tokens: int
    budget_basis_tokens: int
    soft_floor_tokens: int
    resized_height: int
    resized_width: int
    evidence_need: float
    context_need: float
    global_ratio: float
    fine_ratio: float
    global_tokens: int
    fine_tokens: int
    total_tokens: int
    alignment: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def qwen_native_visual_token_geometry(
    width: int,
    height: int,
    *,
    minimum_pixels: int = 4096,
    maximum_pixels: int = 16 * 1024 * 1024,
    patch_size: int = 16,
    merge_size: int = 2,
) -> tuple[int, int, int]:
    """Reproduce Qwen's still-image ``smart_resize`` token geometry.

    The returned token count is *post merger*: it is exactly the number of
    visual tokens that the unmodified Qwen vision encoder would expose for the
    image under the same min/max-pixel contract.  Computing this geometry is
    cheap and does not instantiate a resized image tensor.
    """

    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if minimum_pixels <= 0 or maximum_pixels < minimum_pixels:
        raise ValueError("invalid Qwen min/max-pixel interval")
    if patch_size <= 0 or merge_size <= 0:
        raise ValueError("patch and merge sizes must be positive")
    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {aspect_ratio}"
        )
    factor = int(patch_size * merge_size)
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > maximum_pixels:
        beta = math.sqrt((height * width) / maximum_pixels)
        resized_height = max(
            factor, math.floor(height / beta / factor) * factor
        )
        resized_width = max(
            factor, math.floor(width / beta / factor) * factor
        )
    elif resized_height * resized_width < minimum_pixels:
        beta = math.sqrt(minimum_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    native_tokens = (resized_height // factor) * (resized_width // factor)
    return int(native_tokens), int(resized_height), int(resized_width)


def plan_native_ratio_budget(
    width: int,
    height: int,
    *,
    global_ratio: float,
    fine_ratio: float,
    minimum_global_tokens: int = 64,
    minimum_fine_tokens: int = 64,
    alignment: int = 64,
    minimum_pixels: int = 4096,
    maximum_pixels: int = 16 * 1024 * 1024,
    patch_size: int = 16,
    merge_size: int = 2,
) -> NativeRatioBudget:
    """Allocate both streams as continuous fractions of native Qwen tokens.

    This removes the fixed-3K Fine assumption.  A small image therefore gets
    a small Fine stream, while a large image can spend more evidence tokens.
    The two safety floors only keep each stream structurally valid; they are
    not image-size tiers.
    """

    if not math.isfinite(global_ratio) or not 0 < global_ratio <= 1:
        raise ValueError("global_ratio must lie in (0, 1]")
    if not math.isfinite(fine_ratio) or not 0 < fine_ratio <= 1:
        raise ValueError("fine_ratio must lie in (0, 1]")
    if minimum_global_tokens <= 0 or minimum_fine_tokens <= 0:
        raise ValueError("minimum stream budgets must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    native_tokens, resized_height, resized_width = (
        qwen_native_visual_token_geometry(
            width,
            height,
            minimum_pixels=minimum_pixels,
            maximum_pixels=maximum_pixels,
            patch_size=patch_size,
            merge_size=merge_size,
        )
    )

    def aligned(raw: float, floor: int) -> int:
        value = int(round(raw / alignment) * alignment)
        return max(int(floor), value)

    global_tokens = aligned(global_ratio * native_tokens, minimum_global_tokens)
    fine_tokens = aligned(fine_ratio * native_tokens, minimum_fine_tokens)
    return NativeRatioBudget(
        native_tokens=native_tokens,
        resized_height=resized_height,
        resized_width=resized_width,
        global_ratio=float(global_ratio),
        fine_ratio=float(fine_ratio),
        global_tokens=global_tokens,
        fine_tokens=fine_tokens,
        total_tokens=global_tokens + fine_tokens,
        alignment=int(alignment),
    )


def evidence_context_need(probability: np.ndarray) -> float:
    """Estimate whether answering needs broad/multi-region visual context.

    A diffuse, spatially spread or multi-modal PTEA map should retain more
    Global tokens.  A compact single peak may spend relatively more of the
    native budget on Fine children.  This deterministic statistic is
    deliberately conservative: it changes only the budget split and never
    removes the protected full-image stream.
    """

    statistics = evidence_map_statistics(probability)
    return float(
        np.clip(
            0.45 * statistics["map_entropy"]
            + 0.40 * statistics["map_spatial_spread"]
            + 0.15 * statistics["map_mode_count_normalized"],
            0.0,
            1.0,
        )
    )


def plan_native_need_spread_budget(
    width: int,
    height: int,
    *,
    evidence_need: float,
    context_need: float,
    minimum_global_ratio: float = 0.30,
    maximum_global_ratio: float = 0.50,
    minimum_fine_ratio: float = 0.20,
    maximum_fine_ratio: float = 0.60,
    minimum_global_tokens: int = 64,
    minimum_fine_tokens: int = 64,
    maximum_global_tokens: int = 0,
    maximum_fine_tokens: int = 0,
    soft_floor_tokens: int = 0,
    alignment: int = 64,
    minimum_pixels: int = 4096,
    maximum_pixels: int = 16 * 1024 * 1024,
    patch_size: int = 16,
    merge_size: int = 2,
) -> NativeNeedSpreadBudget:
    """Allocate a continuous native-relative budget using need and topology.

    This resolves two different questions explicitly:

    * ``context_need`` interpolates the protected Global fraction;
    * ``evidence_need`` interpolates the additional Fine fraction.

    Integer alignment is required by Qwen's patch grid, but the underlying
    allocation remains continuous rather than choosing among G1/G2/G3 tiers.
    """

    ratios = (
        minimum_global_ratio,
        maximum_global_ratio,
        minimum_fine_ratio,
        maximum_fine_ratio,
    )
    if not all(math.isfinite(value) for value in ratios):
        raise ValueError("all budget ratios must be finite")
    if not 0 < minimum_global_ratio <= maximum_global_ratio <= 1:
        raise ValueError("invalid Global ratio interval")
    if not 0 < minimum_fine_ratio <= maximum_fine_ratio <= 1:
        raise ValueError("invalid Fine ratio interval")
    if minimum_global_tokens <= 0 or minimum_fine_tokens <= 0:
        raise ValueError("minimum stream budgets must be positive")
    if (
        maximum_global_tokens < 0
        or maximum_fine_tokens < 0
        or (
            maximum_global_tokens
            and maximum_global_tokens < minimum_global_tokens
        )
        or (
            maximum_fine_tokens
            and maximum_fine_tokens < minimum_fine_tokens
        )
    ):
        raise ValueError("invalid maximum stream budgets")
    if soft_floor_tokens < 0:
        raise ValueError("soft_floor_tokens must be non-negative")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    need = float(np.clip(evidence_need, 0.0, 1.0))
    context = float(np.clip(context_need, 0.0, 1.0))
    global_ratio = minimum_global_ratio + context * (
        maximum_global_ratio - minimum_global_ratio
    )
    fine_ratio = minimum_fine_ratio + need * (
        maximum_fine_ratio - minimum_fine_ratio
    )
    native_tokens, resized_height, resized_width = (
        qwen_native_visual_token_geometry(
            width,
            height,
            minimum_pixels=minimum_pixels,
            maximum_pixels=maximum_pixels,
            patch_size=patch_size,
            merge_size=merge_size,
        )
    )
    # A continuous lower envelope avoids a discrete small-image route while
    # preserving the native-size asymptote for large images.  With floor=0
    # this is exactly the original native-relative budget.  For N << floor it
    # approaches the floor smoothly; for N >> floor the overhead vanishes.
    budget_basis_tokens = int(
        round(math.hypot(float(native_tokens), float(soft_floor_tokens)))
    )

    def aligned(raw: float, floor: int, ceiling: int) -> int:
        value = int(round(raw / alignment) * alignment)
        value = max(int(floor), value)
        if ceiling:
            aligned_ceiling = (int(ceiling) // alignment) * alignment
            if aligned_ceiling < floor:
                raise ValueError("aligned stream ceiling is below its floor")
            value = min(value, aligned_ceiling)
        return value

    global_tokens = aligned(
        global_ratio * budget_basis_tokens,
        minimum_global_tokens,
        maximum_global_tokens,
    )
    fine_tokens = aligned(
        fine_ratio * budget_basis_tokens,
        minimum_fine_tokens,
        maximum_fine_tokens,
    )
    return NativeNeedSpreadBudget(
        native_tokens=native_tokens,
        budget_basis_tokens=budget_basis_tokens,
        soft_floor_tokens=int(soft_floor_tokens),
        resized_height=resized_height,
        resized_width=resized_width,
        evidence_need=need,
        context_need=context,
        global_ratio=float(global_ratio),
        fine_ratio=float(fine_ratio),
        global_tokens=global_tokens,
        fine_tokens=fine_tokens,
        total_tokens=global_tokens + fine_tokens,
        alignment=int(alignment),
    )


def plan_evitree_budget(
    evidence_need: float,
    *,
    protected_global_tokens: int = 2048,
    minimum_fine_tokens: int = 512,
    maximum_fine_tokens: int = 4096,
    alignment: int = 64,
) -> EviTreeBudget:
    """Map continuous evidence need to an aligned protected-global budget."""

    if not math.isfinite(evidence_need):
        raise ValueError("evidence_need must be finite")
    if protected_global_tokens <= 0 or minimum_fine_tokens < 0:
        raise ValueError("token budgets must be non-negative with positive global")
    if maximum_fine_tokens < minimum_fine_tokens or alignment <= 0:
        raise ValueError("invalid fine-token range or alignment")
    need = float(np.clip(evidence_need, 0.0, 1.0))
    raw = minimum_fine_tokens + need * (
        maximum_fine_tokens - minimum_fine_tokens
    )
    fine = int(round(raw / alignment)) * alignment
    fine = min(maximum_fine_tokens, max(minimum_fine_tokens, fine))
    return EviTreeBudget(
        evidence_need=need,
        global_tokens=int(protected_global_tokens),
        fine_tokens=fine,
        total_tokens=int(protected_global_tokens + fine),
        alignment=int(alignment),
    )


@dataclass(frozen=True)
class EviTreeLeaf:
    node_id: int
    parent_id: int | None
    path: str
    depth: int
    bbox_xyxy_1000: tuple[int, int, int, int]
    evidence_mass: float
    allocation_mass: float
    token_budget: int

    @property
    def area_ratio(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy_1000
        return (x2 - x1) * (y2 - y1) / 1_000_000.0


@dataclass(frozen=True)
class EviTreePlan:
    leaves: tuple[EviTreeLeaf, ...]
    fine_token_budget: int
    selected_allocation_mass: float
    maximum_depth: int
    reliability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "leaves": [asdict(leaf) for leaf in self.leaves],
        }


@dataclass(frozen=True)
class _TreeNode:
    node_id: int
    parent_id: int | None
    path: str
    depth: int
    x1: float
    y1: float
    x2: float
    y2: float
    evidence_mass: float
    allocation_mass: float

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def _integral(values: np.ndarray) -> np.ndarray:
    return np.pad(values.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def _mass(
    integral: np.ndarray, x1: float, y1: float, x2: float, y2: float
) -> float:
    height, width = integral.shape[0] - 1, integral.shape[1] - 1
    # Every dyadic boundary must map to one shared integer boundary.  Using
    # floor for the left child and ceil for the right child double-counts a
    # boundary cell whenever the continuous boundary lies between pixels.
    ix1 = min(width - 1, max(0, int(round(x1 * width))))
    iy1 = min(height - 1, max(0, int(round(y1 * height))))
    ix2 = min(width, max(ix1 + 1, int(round(x2 * width))))
    iy2 = min(height, max(iy1 + 1, int(round(y2 * height))))
    return float(
        integral[iy2, ix2]
        - integral[iy1, ix2]
        - integral[iy2, ix1]
        + integral[iy1, ix1]
    )


def _split(node: _TreeNode) -> tuple[tuple[float, float, float, float, str], ...]:
    middle_x = 0.5 * (node.x1 + node.x2)
    middle_y = 0.5 * (node.y1 + node.y2)
    return (
        (node.x1, node.y1, middle_x, middle_y, "0"),
        (middle_x, node.y1, node.x2, middle_y, "1"),
        (node.x1, middle_y, middle_x, node.y2, "2"),
        (middle_x, middle_y, node.x2, node.y2, "3"),
    )


def _aligned_allocations(
    weights: Sequence[float], total: int, alignment: int
) -> list[int]:
    if not weights:
        return []
    if total < len(weights) * alignment:
        raise ValueError("budget cannot give one aligned unit to every leaf")
    base = np.full(len(weights), alignment, dtype=np.int64)
    remaining_units = total // alignment - len(weights)
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities /= probabilities.sum()
    raw = probabilities * remaining_units
    extra = np.floor(raw).astype(np.int64)
    missing = int(remaining_units - extra.sum())
    if missing:
        order = np.argsort(-(raw - extra), kind="stable")
        extra[order[:missing]] += 1
    result = (base + alignment * extra).tolist()
    if sum(result) != total:
        raise AssertionError("aligned allocation lost tokens")
    return [int(value) for value in result]


def build_evitree_plan(
    probability: np.ndarray,
    *,
    fine_token_budget: int,
    minimum_leaf_tokens: int = 64,
    maximum_depth: int = 5,
    maximum_leaves: int = 12,
    reliability: float = 0.80,
    target_mass: float = 0.90,
) -> EviTreePlan:
    """Build non-overlapping adaptive fine leaves under an exact budget.

    PTEA supplies the evidence distribution, while a uniform floor receives
    ``1-reliability`` of the allocation mass.  The floor and the separately
    protected global stream prevent a wrong PTEA peak from deleting context.
    """

    if fine_token_budget <= 0 or minimum_leaf_tokens <= 0:
        raise ValueError("fine and minimum leaf budgets must be positive")
    if fine_token_budget % minimum_leaf_tokens:
        raise ValueError("fine budget must align to minimum_leaf_tokens")
    if not 1 <= maximum_depth <= 8:
        raise ValueError("maximum_depth must lie in [1, 8]")
    if maximum_leaves <= 0:
        raise ValueError("maximum_leaves must be positive")
    if not 0.0 <= reliability <= 1.0:
        raise ValueError("reliability must lie in [0, 1]")
    if not 0.0 < target_mass <= 1.0:
        raise ValueError("target_mass must lie in (0, 1]")

    evidence = _normalise_probability(probability)
    uniform = np.full_like(evidence, 1.0 / evidence.size)
    allocation = reliability * evidence + (1.0 - reliability) * uniform
    evidence_integral = _integral(evidence)
    allocation_integral = _integral(allocation)
    next_id = 0

    def node(
        *,
        parent_id: int | None,
        path: str,
        depth: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> _TreeNode:
        nonlocal next_id
        value = _TreeNode(
            node_id=next_id,
            parent_id=parent_id,
            path=path,
            depth=depth,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            evidence_mass=_mass(evidence_integral, x1, y1, x2, y2),
            allocation_mass=_mass(allocation_integral, x1, y1, x2, y2),
        )
        next_id += 1
        return value

    root = node(
        parent_id=None, path="", depth=0, x1=0.0, y1=0.0, x2=1.0, y2=1.0
    )
    frontier = [root]
    maximum_leaves = min(
        maximum_leaves,
        fine_token_budget // minimum_leaf_tokens,
    )
    target_frontier = min(
        4**maximum_depth,
        max(1, maximum_leaves),
    )
    while len(frontier) + 3 <= target_frontier:
        splittable = [item for item in frontier if item.depth < maximum_depth]
        if not splittable:
            break
        # High allocation mass and high density receive finer leaves.  The
        # small depth discount stops a single sharp pixel from consuming every
        # split before secondary evidence modes are considered.
        chosen = max(
            splittable,
            key=lambda item: (
                item.allocation_mass
                * (item.allocation_mass / max(item.area, 1e-12)) ** 0.20
                / (1.0 + 0.15 * item.depth),
                -item.node_id,
            ),
        )
        frontier.remove(chosen)
        for x1, y1, x2, y2, suffix in _split(chosen):
            frontier.append(
                node(
                    parent_id=chosen.node_id,
                    path=chosen.path + suffix,
                    depth=chosen.depth + 1,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    ordered = sorted(
        frontier,
        key=lambda item: (-item.allocation_mass, item.node_id),
    )
    selected: list[_TreeNode] = []
    cumulative = 0.0
    for item in ordered:
        selected.append(item)
        cumulative += item.allocation_mass
        if cumulative >= target_mass or len(selected) >= maximum_leaves:
            break
    selected = selected[:maximum_leaves]
    selected.sort(key=lambda item: (item.y1, item.x1, item.depth, item.node_id))
    budgets = _aligned_allocations(
        [max(item.allocation_mass, 1e-12) ** 0.75 for item in selected],
        fine_token_budget,
        minimum_leaf_tokens,
    )
    leaves = []
    for item, token_budget in zip(selected, budgets, strict=True):
        leaves.append(
            EviTreeLeaf(
                node_id=item.node_id,
                parent_id=item.parent_id,
                path=item.path,
                depth=item.depth,
                bbox_xyxy_1000=(
                    int(round(1000 * item.x1)),
                    int(round(1000 * item.y1)),
                    int(round(1000 * item.x2)),
                    int(round(1000 * item.y2)),
                ),
                evidence_mass=item.evidence_mass,
                allocation_mass=item.allocation_mass,
                token_budget=token_budget,
            )
        )
    return EviTreePlan(
        leaves=tuple(leaves),
        fine_token_budget=fine_token_budget,
        selected_allocation_mass=float(
            sum(item.allocation_mass for item in selected)
        ),
        maximum_depth=maximum_depth,
        reliability=float(reliability),
    )


def build_learned_evitree_plan(
    probability: np.ndarray,
    split_policy: HierarchicalSplitPolicyHead,
    *,
    split_threshold: float,
    fine_token_budget: int,
    minimum_leaf_tokens: int = 64,
    maximum_depth: int = 5,
    maximum_leaves: int = 10,
    reliability: float = 0.80,
) -> EviTreePlan:
    """Build a variable-depth tree using the human-supervised split policy.

    All current frontier leaves are retained.  Consequently the fine path is a
    complete adaptive partition: broad/uncertain evidence remains a large
    high-resolution view, while concentrated evidence is recursively refined.
    The separately protected global stream remains untouched in either case.
    """

    if not 0.0 <= split_threshold <= 1.0:
        raise ValueError("split_threshold must lie in [0, 1]")
    if fine_token_budget % minimum_leaf_tokens:
        raise ValueError("fine budget must align to minimum_leaf_tokens")
    if maximum_leaves <= 0:
        raise ValueError("maximum_leaves must be positive")
    if not 0.0 <= reliability <= 1.0:
        raise ValueError("reliability must lie in [0, 1]")
    maximum_leaves = min(
        maximum_leaves,
        fine_token_budget // minimum_leaf_tokens,
    )
    if maximum_leaves <= 0:
        raise ValueError("fine budget cannot fund one tree leaf")
    evidence = _normalise_probability(probability)
    uniform = np.full_like(evidence, 1.0 / evidence.size)
    allocation = reliability * evidence + (1.0 - reliability) * uniform
    evidence_integral = _integral(evidence)
    allocation_integral = _integral(allocation)
    next_id = 0

    def node(
        *,
        parent_id: int | None,
        path: str,
        depth: int,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> _TreeNode:
        nonlocal next_id
        value = _TreeNode(
            node_id=next_id,
            parent_id=parent_id,
            path=path,
            depth=depth,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            evidence_mass=_mass(evidence_integral, x1, y1, x2, y2),
            allocation_mass=_mass(allocation_integral, x1, y1, x2, y2),
        )
        next_id += 1
        return value

    frontier = [
        node(
            parent_id=None,
            path="",
            depth=0,
            x1=0.0,
            y1=0.0,
            x2=1.0,
            y2=1.0,
        )
    ]
    policy_device = next(split_policy.parameters()).device
    while len(frontier) + 3 <= maximum_leaves:
        splittable = [item for item in frontier if item.depth < maximum_depth]
        if not splittable:
            break
        feature_batch = np.stack(
            [
                split_policy_feature_vector(
                    evidence,
                    box_xyxy=(item.x1, item.y1, item.x2, item.y2),
                    depth=item.depth,
                    maximum_depth=maximum_depth,
                )
                for item in splittable
            ]
        )
        with torch.inference_mode():
            scores = (
                split_policy(
                    torch.from_numpy(feature_batch).to(policy_device)
                )
                .detach()
                .float()
                .cpu()
                .numpy()
            )
        chosen_index = int(np.argmax(scores))
        if float(scores[chosen_index]) < split_threshold:
            break
        chosen = splittable[chosen_index]
        frontier.remove(chosen)
        for x1, y1, x2, y2, suffix in _split(chosen):
            frontier.append(
                node(
                    parent_id=chosen.node_id,
                    path=chosen.path + suffix,
                    depth=chosen.depth + 1,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    frontier.sort(key=lambda item: (item.y1, item.x1, item.depth, item.node_id))
    budgets = _aligned_allocations(
        [max(item.allocation_mass, 1e-12) ** 0.75 for item in frontier],
        fine_token_budget,
        minimum_leaf_tokens,
    )
    leaves = tuple(
        EviTreeLeaf(
            node_id=item.node_id,
            parent_id=item.parent_id,
            path=item.path,
            depth=item.depth,
            bbox_xyxy_1000=(
                int(round(1000 * item.x1)),
                int(round(1000 * item.y1)),
                int(round(1000 * item.x2)),
                int(round(1000 * item.y2)),
            ),
            evidence_mass=item.evidence_mass,
            allocation_mass=item.allocation_mass,
            token_budget=token_budget,
        )
        for item, token_budget in zip(frontier, budgets, strict=True)
    )
    return EviTreePlan(
        leaves=leaves,
        fine_token_budget=fine_token_budget,
        selected_allocation_mass=float(
            sum(item.allocation_mass for item in frontier)
        ),
        maximum_depth=max((item.depth for item in frontier), default=0),
        reliability=float(reliability),
    )


def build_sparse_branch_evitree_plan(
    probability: np.ndarray,
    split_policy: HierarchicalSplitPolicyHead,
    *,
    split_threshold: float,
    fine_token_budget: int,
    minimum_leaf_tokens: int = 64,
    maximum_depth: int = 5,
    maximum_frontier_leaves: int = 10,
    maximum_branches: int = 3,
    minimum_branch_depth: int = 0,
    density_power: float = 0.20,
    recenter_branch_tips: bool = False,
    branch_expansion_factor: float = 1.0,
    maximum_branch_iou: float = 1.0,
    reliability: float = 0.80,
    target_mass: float = 0.90,
) -> EviTreePlan:
    """Re-read only the strongest leaves of a learned adaptive frontier.

    The protected global stream already represents every low-evidence branch.
    Spending Fine tokens on the complete frontier duplicates that context and
    fragments the high-resolution budget.  This decoder therefore uses the
    learned tree only to create adaptive candidate scales, then selects at most
    ``maximum_branches`` non-overlapping high-evidence leaves for re-reading.
    """

    if maximum_branches <= 0:
        raise ValueError("maximum_branches must be positive")
    if not 0 <= minimum_branch_depth <= maximum_depth:
        raise ValueError("minimum_branch_depth must lie in [0, maximum_depth]")
    if not 0.0 <= density_power <= 1.0:
        raise ValueError("density_power must lie in [0, 1]")
    if branch_expansion_factor < 1.0:
        raise ValueError("branch_expansion_factor must be at least 1")
    if not 0.0 <= maximum_branch_iou <= 1.0:
        raise ValueError("maximum_branch_iou must lie in [0, 1]")
    if not 0.0 < target_mass <= 1.0:
        raise ValueError("target_mass must lie in (0, 1]")
    # Build the variable-depth frontier first.  A temporary budget gives every
    # possible frontier leaf one aligned unit; its allocation is discarded.
    frontier_plan = build_learned_evitree_plan(
        probability,
        split_policy,
        split_threshold=split_threshold,
        fine_token_budget=max(
            fine_token_budget,
            maximum_frontier_leaves * minimum_leaf_tokens,
        ),
        minimum_leaf_tokens=minimum_leaf_tokens,
        maximum_depth=maximum_depth,
        maximum_leaves=maximum_frontier_leaves,
        reliability=reliability,
    )
    # The protected Global stream already covers the full image.  A coarse
    # frontier sibling therefore does not need to consume expensive Fine
    # tokens merely to complete a partition.  Optionally expand every coarse
    # branch to a minimum depth and rank the resulting branch tips only.
    # This prevents mass-heavy quarter-image leaves from recreating the
    # original "large box wins" bias.
    evidence = _normalise_probability(probability)
    uniform = np.full_like(evidence, 1.0 / evidence.size)
    allocation = reliability * evidence + (1.0 - reliability) * uniform
    evidence_integral = _integral(evidence)
    allocation_integral = _integral(allocation)
    next_node_id = max(
        (leaf.node_id for leaf in frontier_plan.leaves), default=-1
    ) + 1

    def expand_to_minimum_depth(leaf: EviTreeLeaf) -> list[EviTreeLeaf]:
        nonlocal next_node_id
        if leaf.depth >= minimum_branch_depth:
            return [leaf]
        x1, y1, x2, y2 = (
            value / 1000.0 for value in leaf.bbox_xyxy_1000
        )
        middle_x = 0.5 * (x1 + x2)
        middle_y = 0.5 * (y1 + y2)
        bounds = (
            (x1, y1, middle_x, middle_y, "0"),
            (middle_x, y1, x2, middle_y, "1"),
            (x1, middle_y, middle_x, y2, "2"),
            (middle_x, middle_y, x2, y2, "3"),
        )
        expanded: list[EviTreeLeaf] = []
        for cx1, cy1, cx2, cy2, suffix in bounds:
            child = EviTreeLeaf(
                node_id=next_node_id,
                parent_id=leaf.node_id,
                path=leaf.path + suffix,
                depth=leaf.depth + 1,
                bbox_xyxy_1000=(
                    int(round(1000 * cx1)),
                    int(round(1000 * cy1)),
                    int(round(1000 * cx2)),
                    int(round(1000 * cy2)),
                ),
                evidence_mass=_mass(
                    evidence_integral, cx1, cy1, cx2, cy2
                ),
                allocation_mass=_mass(
                    allocation_integral, cx1, cy1, cx2, cy2
                ),
                token_budget=minimum_leaf_tokens,
            )
            next_node_id += 1
            expanded.extend(expand_to_minimum_depth(child))
        return expanded

    branch_tips = [
        tip
        for leaf in frontier_plan.leaves
        for tip in expand_to_minimum_depth(leaf)
    ]
    ordered = sorted(
        branch_tips,
        key=lambda leaf: (
            -(
                leaf.allocation_mass
                * (
                    leaf.allocation_mass
                    / max(leaf.area_ratio, 1e-12)
                )
                ** density_power
            ),
            leaf.node_id,
        ),
    )
    def recenter(leaf: EviTreeLeaf) -> EviTreeLeaf:
        if not recenter_branch_tips:
            return leaf
        height, width = evidence.shape
        x1, y1, x2, y2 = (
            value / 1000.0 for value in leaf.bbox_xyxy_1000
        )
        ix1, iy1, ix2, iy2 = _array_bounds(
            evidence.shape, (x1, y1, x2, y2)
        )
        patch = evidence[iy1:iy2, ix1:ix2]
        peak_y, peak_x = np.unravel_index(
            int(np.argmax(patch)), patch.shape
        )
        center_x = (ix1 + peak_x + 0.5) / width
        center_y = (iy1 + peak_y + 0.5) / height
        box_width = min(1.0, (x2 - x1) * branch_expansion_factor)
        box_height = min(1.0, (y2 - y1) * branch_expansion_factor)
        rx1 = min(max(0.0, center_x - 0.5 * box_width), 1.0 - box_width)
        ry1 = min(max(0.0, center_y - 0.5 * box_height), 1.0 - box_height)
        rx2 = rx1 + box_width
        ry2 = ry1 + box_height
        return EviTreeLeaf(
            node_id=leaf.node_id,
            parent_id=leaf.parent_id,
            path=leaf.path,
            depth=leaf.depth,
            bbox_xyxy_1000=(
                int(round(1000 * rx1)),
                int(round(1000 * ry1)),
                int(round(1000 * rx2)),
                int(round(1000 * ry2)),
            ),
            evidence_mass=_mass(
                evidence_integral, rx1, ry1, rx2, ry2
            ),
            allocation_mass=_mass(
                allocation_integral, rx1, ry1, rx2, ry2
            ),
            token_budget=leaf.token_budget,
        )

    def branch_iou(first: EviTreeLeaf, second: EviTreeLeaf) -> float:
        ax1, ay1, ax2, ay2 = (
            value / 1000.0 for value in first.bbox_xyxy_1000
        )
        bx1, by1, bx2, by2 = (
            value / 1000.0 for value in second.bbox_xyxy_1000
        )
        overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0.0, min(ay2, by2) - max(ay1, by1)
        )
        union = first.area_ratio + second.area_ratio - overlap
        return overlap / max(union, 1e-12)

    selected: list[EviTreeLeaf] = []
    cumulative = 0.0
    for raw_leaf in ordered:
        leaf = recenter(raw_leaf)
        if selected and max(
            branch_iou(leaf, chosen) for chosen in selected
        ) > maximum_branch_iou:
            continue
        selected.append(leaf)
        cumulative += leaf.allocation_mass
        if cumulative >= target_mass or len(selected) >= maximum_branches:
            break
    if not selected:
        # The first branch can never violate overlap, but keep a fail-closed
        # fallback if future candidate filters are added.
        selected = [recenter(ordered[0])]
    # Small native images can legitimately receive less Fine budget than the
    # nominal maximum number of branches requires.  Preserve the continuous
    # budget instead of inflating it: keep only as many highest-value branches
    # as can receive one aligned token unit.  This makes branch count itself
    # adaptive to image size while retaining the exact requested budget.
    affordable_branches = max(1, fine_token_budget // minimum_leaf_tokens)
    if len(selected) > affordable_branches:
        selected = selected[:affordable_branches]
    selected.sort(
        key=lambda leaf: (
            leaf.bbox_xyxy_1000[1],
            leaf.bbox_xyxy_1000[0],
            leaf.depth,
            leaf.node_id,
        )
    )
    budgets = _aligned_allocations(
        [
            max(leaf.allocation_mass, 1e-12) ** 0.75
            for leaf in selected
        ],
        fine_token_budget,
        minimum_leaf_tokens,
    )
    leaves = tuple(
        EviTreeLeaf(
            node_id=leaf.node_id,
            parent_id=leaf.parent_id,
            path=leaf.path,
            depth=leaf.depth,
            bbox_xyxy_1000=leaf.bbox_xyxy_1000,
            evidence_mass=leaf.evidence_mass,
            allocation_mass=leaf.allocation_mass,
            token_budget=token_budget,
        )
        for leaf, token_budget in zip(selected, budgets, strict=True)
    )
    return EviTreePlan(
        leaves=leaves,
        fine_token_budget=fine_token_budget,
        selected_allocation_mass=float(
            sum(leaf.allocation_mass for leaf in selected)
        ),
        maximum_depth=max((leaf.depth for leaf in selected), default=0),
        reliability=float(reliability),
    )


__all__ = [
    "SCALAR_NEED_FEATURE_NAMES",
    "SPLIT_POLICY_FEATURE_NAMES",
    "EviTreeBudget",
    "EviTreeLeaf",
    "EviTreePlan",
    "NativeRatioBudget",
    "EvidenceNeedHead",
    "NativeNeedSpreadBudget",
    "HierarchicalSplitPolicyHead",
    "ScalarEvidenceNeedHead",
    "NEED_STAT_NAMES",
    "build_evitree_plan",
    "build_learned_evitree_plan",
    "build_sparse_branch_evitree_plan",
    "continuous_fine_token_budget",
    "continuous_native_global_token_budget",
    "evidence_context_need",
    "evidence_map_statistics",
    "human_evidence_need_target",
    "load_hierarchical_split_policy_head",
    "load_scalar_evidence_need_head",
    "need_feature_vector",
    "plan_native_ratio_budget",
    "plan_native_need_spread_budget",
    "plan_evitree_budget",
    "qwen_native_visual_token_geometry",
    "scalar_need_feature_vector",
    "split_policy_feature_vector",
]
