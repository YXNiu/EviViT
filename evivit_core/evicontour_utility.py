"""Lightweight dual-head utility model for Stage-G EviContour nodes."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from evivit_core.dense_evidence import map_mass
from evivit_core.evidence_component_tree import (
    build_evidence_component_tree,
    candidate_records,
)


EVICONTOUR_FEATURE_NAMES = (
    "log_source_pixels",
    "absolute_log_aspect",
    "question_length",
    "tree_size",
    "candidate_count",
    "center_x",
    "center_y",
    "width",
    "height",
    "log_area",
    "log_aspect_ratio",
    "threshold_ratio",
    "peak",
    "log_cell_count",
    "component_mass",
    "component_density",
    "component_energy",
    "component_entropy_sum",
    "fill_ratio",
    "log_parent_area_growth",
    "log_parent_mass_growth",
    "ptea_box_mass",
    "ptea_context_ring_mass",
    "ptea_context_ring_density",
)


def evicontour_feature_vector(
    group: dict[str, Any], candidate: dict[str, Any]
) -> np.ndarray:
    """Build a question-conditioned topology feature vector.

    PTEA already conditions all map-derived values on the full question.  The
    explicit question length is only a small complexity cue, not a replacement
    text encoder.
    """

    width_px, height_px = [max(1, int(value)) for value in group["image_size"]]
    x0, y0, x1, y1 = [float(value) / 1000.0 for value in candidate["bbox"]]
    width, height = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    tree = group["tree"]
    values = {
        "log_source_pixels": math.log1p(width_px * height_px) / 20.0,
        "absolute_log_aspect": min(4.0, abs(math.log(width_px / height_px))) / 4.0,
        "question_length": min(1.0, len(str(group["question"]).split()) / 40.0),
        "tree_size": math.log1p(int(tree["full_nodes"])) / 10.0,
        "candidate_count": math.log1p(int(tree["candidate_count"])) / 6.0,
        "center_x": 0.5 * (x0 + x1),
        "center_y": 0.5 * (y0 + y1),
        "width": width,
        "height": height,
        "log_area": math.log(max(float(candidate["area_ratio"]), 1e-8)),
        "log_aspect_ratio": math.log(max(float(candidate["aspect_ratio"]), 1.0)),
        "threshold_ratio": float(candidate["threshold_ratio"]),
        "peak": float(candidate["peak"]),
        "log_cell_count": math.log1p(int(candidate["cell_count"])),
        "component_mass": float(candidate["component_mass"]),
        "component_density": float(candidate["component_density"]),
        "component_energy": float(candidate["component_energy"]),
        "component_entropy_sum": float(candidate["component_entropy_sum"]),
        "fill_ratio": float(candidate["fill_ratio"]),
        "log_parent_area_growth": math.log1p(max(float(candidate["parent_area_growth"]), 0.0)),
        "log_parent_mass_growth": math.log1p(max(float(candidate["parent_mass_growth"]), 0.0)),
        "ptea_box_mass": float(candidate["ptea_box_mass"]),
        "ptea_context_ring_mass": float(candidate["ptea_context_ring_mass"]),
        "ptea_context_ring_density": float(candidate["ptea_context_ring_density"]),
    }
    return np.asarray([values[name] for name in EVICONTOUR_FEATURE_NAMES], dtype=np.float32)


class EviContourUtilityHead(nn.Module):
    """Shared trunk with separate inspection-relevance and sufficiency heads."""

    def __init__(self, feature_dim: int = len(EVICONTOUR_FEATURE_NAMES), hidden_dim: int = 64) -> None:
        super().__init__()
        if min(feature_dim, hidden_dim) <= 0:
            raise ValueError("feature and hidden dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_scale", torch.ones(feature_dim))
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.relevance_head = nn.Linear(hidden_dim, 1)
        self.sufficiency_head = nn.Linear(hidden_dim, 1)

    def set_normalization(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        if mean.shape != self.feature_mean.shape or scale.shape != self.feature_scale.shape:
            raise ValueError("normalization tensors have the wrong shape")
        self.feature_mean.copy_(mean)
        self.feature_scale.copy_(scale.clamp_min(1e-6))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[-1] != self.feature_dim:
            raise ValueError("features have the wrong final dimension")
        normalized = (features.float() - self.feature_mean) / self.feature_scale
        hidden = self.trunk(normalized)
        return (
            torch.sigmoid(self.relevance_head(hidden).squeeze(-1)),
            torch.sigmoid(self.sufficiency_head(hidden).squeeze(-1)),
        )


def load_evicontour_utility_head(
    checkpoint_path: str, *, map_location: str | torch.device = "cpu"
) -> tuple[EviContourUtilityHead, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if payload.get("format_version") != "evicontour_utility_shared_v1":
        raise ValueError("unsupported EviContour utility checkpoint")
    if tuple(payload.get("feature_names", ())) != EVICONTOUR_FEATURE_NAMES:
        raise ValueError("EviContour checkpoint feature contract mismatch")
    model = EviContourUtilityHead(hidden_dim=int(payload["hidden_dim"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    model.requires_grad_(False)
    return model, payload


def _policy_area(box: Sequence[float]) -> float:
    x0, y0, x1, y1 = [float(value) / 1000.0 for value in box]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _expanded_policy_box(box: Sequence[float], margin: float = 0.20) -> list[int]:
    x0, y0, x1, y1 = [float(value) for value in box]
    width, height = x1 - x0, y1 - y0
    return [
        int(round(max(0.0, x0 - margin * width))),
        int(round(max(0.0, y0 - margin * height))),
        int(round(min(1000.0, x1 + margin * width))),
        int(round(min(1000.0, y1 + margin * height))),
    ]


def build_evicontour_inference_group(
    probability: np.ndarray,
    *,
    image_size: Sequence[int],
    question: str,
    maximum_candidates: int = 128,
) -> dict[str, Any]:
    """Build the exact inference-time feature contract without trace labels."""

    tree = build_evidence_component_tree(
        probability,
        maximum_candidates=maximum_candidates,
        minimum_peak_ratio=0.10,
        minimum_area_growth=0.20,
        minimum_mass_growth=0.05,
        maximum_area_ratio=0.80,
    )
    candidates = candidate_records(tree)
    for candidate in candidates:
        box = candidate["bbox"]
        expanded = _expanded_policy_box(box)
        area = max(_policy_area(box), 1e-8)
        ring_area = max(_policy_area(expanded) - area, 1e-8)
        box_mass = map_mass(probability, box)
        expanded_mass = map_mass(probability, expanded)
        candidate.update(
            {
                "ptea_box_mass": float(box_mass),
                "ptea_context_ring_mass": float(max(0.0, expanded_mass - box_mass)),
                "ptea_context_ring_density": float(
                    max(0.0, expanded_mass - box_mass) / ring_area
                ),
            }
        )
    return {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "question": str(question),
        "tree": {
            "full_nodes": len(tree.nodes),
            "candidate_count": len(candidates),
        },
        "candidates": candidates,
    }


@torch.inference_mode()
def predict_evicontour_portfolio(
    model: EviContourUtilityHead,
    group: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
    max_regions: int = 3,
) -> list[dict[str, Any]]:
    features = np.stack(
        [evicontour_feature_vector(group, candidate) for candidate in group["candidates"]]
    )
    inputs = torch.from_numpy(features).to(device)
    model = model.to(device)
    relevance, sufficiency = model(inputs)
    return select_evicontour_portfolio(
        group["candidates"],
        relevance.detach().cpu().numpy(),
        sufficiency.detach().cpu().numpy(),
        max_regions=max_regions,
    )


@torch.inference_mode()
def predict_evicontour_csr_portfolio(
    model: EviContourUtilityHead,
    group: dict[str, Any],
    *,
    device: str | torch.device = "cpu",
    max_regions: int = 3,
    sufficiency_threshold: float = 0.65,
    sufficiency_tolerance: float = 0.05,
    core_relevance_tolerance: float = 0.05,
    context_ring_mass_ratio: float = 0.20,
    maximum_scope_hops: int = 3,
    branch_sufficiency_weight: float = 0.35,
    core_strategy: str = "g1_anchor",
    minimum_parent_sufficiency_gain: float = 0.03,
) -> list[dict[str, Any]]:
    """Decode Core, Scope and later Resolution as separate decisions.

    This function handles only Core and Scope.  Resolution is deliberately
    applied by the image processor because it depends on the native Qwen grid
    of the selected crop.  Every score is predicted for every node before the
    tree policy is applied; no external QA label is available at inference.
    """

    features = np.stack(
        [evicontour_feature_vector(group, candidate) for candidate in group["candidates"]]
    )
    inputs = torch.from_numpy(features).to(device)
    model = model.to(device)
    relevance, sufficiency = model(inputs)
    return select_evicontour_csr_portfolio(
        group["candidates"],
        relevance.detach().cpu().numpy(),
        sufficiency.detach().cpu().numpy(),
        max_regions=max_regions,
        sufficiency_threshold=sufficiency_threshold,
        sufficiency_tolerance=sufficiency_tolerance,
        core_relevance_tolerance=core_relevance_tolerance,
        context_ring_mass_ratio=context_ring_mass_ratio,
        maximum_scope_hops=maximum_scope_hops,
        branch_sufficiency_weight=branch_sufficiency_weight,
        core_strategy=core_strategy,
        minimum_parent_sufficiency_gain=minimum_parent_sufficiency_gain,
    )


def policy_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in first]
    bx0, by0, bx1, by1 = [float(value) for value in second]
    overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0)
    )
    first_area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    second_area = max(1.0, (bx1 - bx0) * (by1 - by0))
    return overlap / max(first_area + second_area - overlap, 1.0)


def select_evicontour_portfolio(
    candidates: Sequence[dict[str, Any]],
    relevance: Sequence[float],
    sufficiency: Sequence[float],
    *,
    max_regions: int = 3,
    maximum_pairwise_iou: float = 0.55,
    minimum_relative_utility: float = 0.20,
) -> list[dict[str, Any]]:
    """Select complementary regions without forcing all three slots."""

    if len(candidates) != len(relevance) or len(candidates) != len(sufficiency):
        raise ValueError("candidate and score lengths must match")
    utility = [
        0.35 * float(rel) + 0.65 * float(suf) - 0.12 * math.sqrt(max(float(row["area_ratio"]), 0.0))
        for row, rel, suf in zip(candidates, relevance, sufficiency)
    ]
    order = sorted(range(len(candidates)), key=lambda index: utility[index], reverse=True)
    selected: list[dict[str, Any]] = []
    first_utility = utility[order[0]] if order else 0.0
    for index in order:
        if len(selected) >= max_regions:
            break
        # The first slot is a best-available safety anchor.  On low-evidence
        # images every calibrated utility can legitimately be below the stop
        # threshold; returning an empty portfolio would violate the current
        # online Bridge contract.  The threshold therefore controls only
        # optional second/third regions, rather than deleting all evidence.
        if selected and utility[index] < max(
            0.20, minimum_relative_utility * first_utility
        ):
            break
        if any(
            tuple(candidates[index].get("peak_grid", ()))
            == tuple(row.get("peak_grid", ()))
            for row in selected
        ):
            continue
        if any(policy_iou(candidates[index]["bbox"], row["bbox"]) > maximum_pairwise_iou for row in selected):
            continue
        selected.append(
            {
                **candidates[index],
                "predicted_relevance": float(relevance[index]),
                "predicted_sufficiency": float(sufficiency[index]),
                "predicted_utility": float(utility[index]),
            }
        )
    return selected


def _candidate_parent_chain(
    core_index: int,
    candidates: Sequence[dict[str, Any]],
    index_by_node: dict[int, int],
) -> list[int]:
    """Return one compressed candidate chain from a core toward the root."""

    chain: list[int] = []
    seen: set[int] = set()
    current: int | None = int(core_index)
    while current is not None and current not in seen:
        chain.append(current)
        seen.add(current)
        parent_id = candidates[current].get("candidate_parent_id")
        current = (
            index_by_node.get(int(parent_id))
            if parent_id is not None
            else None
        )
    return chain


def _point_inside_policy_box(point: Sequence[int], box: Sequence[int]) -> bool:
    px, py = [int(value) for value in point]
    x0, y0, x1, y1 = [int(value) for value in box]
    return x0 <= px < x1 and y0 <= py < y1


def select_evicontour_csr_portfolio(
    candidates: Sequence[dict[str, Any]],
    relevance: Sequence[float],
    sufficiency: Sequence[float],
    *,
    max_regions: int = 3,
    sufficiency_threshold: float = 0.65,
    sufficiency_tolerance: float = 0.05,
    core_relevance_tolerance: float = 0.05,
    context_ring_mass_ratio: float = 0.20,
    maximum_scope_hops: int = 3,
    branch_sufficiency_weight: float = 0.35,
    core_strategy: str = "g1_anchor",
    minimum_parent_sufficiency_gain: float = 0.03,
    maximum_pairwise_iou: float = 0.65,
    minimum_relative_relevance: float = 0.20,
) -> list[dict[str, Any]]:
    """Select evidence branches first, then calibrate extent on each parent chain.

    G1 ranked all nodes with one mixed utility.  That made a tight child and
    its context-bearing parent compete globally.  CSR instead chooses one
    compact Core per evidence peak, then moves only along that Core's parent
    chain to find the smallest node whose predicted sufficiency is close to
    the best attainable sufficiency on the branch.  A strong context ring can
    trigger one additional parent step when it does not reduce sufficiency.
    """

    if len(candidates) != len(relevance) or len(candidates) != len(sufficiency):
        raise ValueError("candidate and score lengths must match")
    if max_regions <= 0:
        raise ValueError("max_regions must be positive")
    if not 0.0 <= sufficiency_threshold <= 1.0:
        raise ValueError("sufficiency_threshold must lie in [0, 1]")
    if min(sufficiency_tolerance, core_relevance_tolerance) < 0:
        raise ValueError("CSR tolerances must be non-negative")
    if context_ring_mass_ratio < 0:
        raise ValueError("context_ring_mass_ratio must be non-negative")
    if maximum_scope_hops < 0:
        raise ValueError("maximum_scope_hops must be non-negative")
    if not 0.0 <= branch_sufficiency_weight <= 1.0:
        raise ValueError("branch_sufficiency_weight must lie in [0, 1]")
    if core_strategy not in {"g1_anchor", "branch_relevance"}:
        raise ValueError("unknown CSR core strategy")
    if minimum_parent_sufficiency_gain < 0:
        raise ValueError("minimum_parent_sufficiency_gain must be non-negative")
    if not candidates:
        return []

    index_by_node = {
        int(candidate["node_id"]): index for index, candidate in enumerate(candidates)
    }

    if core_strategy == "g1_anchor":
        # G2-R is a conservative causal intervention: retain the G1 portfolio
        # as Core anchors and change only Scope.  A learned, fully decoupled
        # Core selector is reserved for G2-M after this geometry is validated.
        anchors = select_evicontour_portfolio(
            candidates,
            relevance,
            sufficiency,
            max_regions=max_regions,
        )
        anchor_indices = [
            index_by_node[int(anchor["node_id"])] for anchor in anchors
        ]
        selected: list[dict[str, Any]] = []
        for core_index in anchor_indices:
            chain = _candidate_parent_chain(core_index, candidates, index_by_node)[
                : maximum_scope_hops + 1
            ]
            scope_index = core_index
            context_promoted = False
            for parent_index in chain[1:]:
                current_sufficiency = float(sufficiency[scope_index])
                parent_sufficiency = float(sufficiency[parent_index])
                ring_ratio = float(
                    candidates[scope_index].get("ptea_context_ring_mass", 0.0)
                ) / max(
                    float(candidates[scope_index].get("ptea_box_mass", 0.0)),
                    1e-8,
                )
                improves = (
                    parent_sufficiency - current_sufficiency
                    >= minimum_parent_sufficiency_gain
                )
                preserves_with_context = (
                    ring_ratio >= context_ring_mass_ratio
                    and parent_sufficiency
                    >= current_sufficiency - sufficiency_tolerance
                )
                if not (improves or preserves_with_context):
                    break
                scope_index = parent_index
                context_promoted = context_promoted or preserves_with_context
            scope = candidates[scope_index]
            selected.append(
                {
                    **scope,
                    "predicted_relevance": float(relevance[core_index]),
                    "predicted_sufficiency": float(sufficiency[scope_index]),
                    "predicted_utility": float(sufficiency[scope_index]),
                    "csr_core_node_id": int(candidates[core_index]["node_id"]),
                    "csr_scope_node_id": int(scope["node_id"]),
                    "csr_scope_hops": int(chain.index(scope_index)),
                    "csr_branch_max_sufficiency": max(
                        float(sufficiency[index]) for index in chain
                    ),
                    "csr_context_ring_ratio": float(
                        candidates[scope_index].get("ptea_context_ring_mass", 0.0)
                    )
                    / max(
                        float(candidates[scope_index].get("ptea_box_mass", 0.0)),
                        1e-8,
                    ),
                    "csr_context_promoted": context_promoted,
                    "csr_maximum_scope_hops": int(maximum_scope_hops),
                    "csr_core_strategy": core_strategy,
                }
            )
        return selected

    # Core: retain the smallest near-best-relevance node for each PTEA peak.
    # Scope is calibrated later, so broad ancestors cannot win merely because
    # they contain more evidence mass.
    by_peak: dict[tuple[int, int], list[int]] = {}
    for index, candidate in enumerate(candidates):
        peak = tuple(int(value) for value in candidate.get("peak_grid", (0, 0)))
        by_peak.setdefault(peak, []).append(index)
    cores: list[int] = []
    for indices in by_peak.values():
        best_relevance = max(float(relevance[index]) for index in indices)
        eligible = [
            index
            for index in indices
            if float(relevance[index]) >= best_relevance - core_relevance_tolerance
        ]
        cores.append(
            min(
                eligible,
                key=lambda index: (
                    float(candidates[index]["area_ratio"]),
                    -float(relevance[index]),
                    int(candidates[index]["node_id"]),
                ),
            )
        )
    core_chains = {
        index: _candidate_parent_chain(index, candidates, index_by_node)
        for index in cores
    }
    branch_scores = {
        index: (
            (1.0 - branch_sufficiency_weight) * float(relevance[index])
            + branch_sufficiency_weight
            * max(float(sufficiency[item]) for item in core_chains[index])
        )
        for index in cores
    }
    cores.sort(
        key=lambda index: (
            branch_scores[index],
            float(relevance[index]),
            float(candidates[index].get("component_density", 0.0)),
            -float(candidates[index]["area_ratio"]),
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    first_relevance = float(relevance[cores[0]]) if cores else 0.0
    for core_index in cores:
        if len(selected) >= max_regions:
            break
        core_relevance = float(relevance[core_index])
        if selected and core_relevance < max(
            0.20, minimum_relative_relevance * first_relevance
        ):
            break

        full_chain = core_chains[core_index]
        chain = full_chain[: maximum_scope_hops + 1]
        maximum_sufficiency = max(float(sufficiency[index]) for index in chain)
        target = min(
            maximum_sufficiency,
            max(sufficiency_threshold, maximum_sufficiency - sufficiency_tolerance),
        )
        eligible_scope = [
            index for index in chain if float(sufficiency[index]) >= target
        ]
        if eligible_scope:
            scope_index = min(
                eligible_scope,
                key=lambda index: (
                    float(candidates[index]["area_ratio"]),
                    chain.index(index),
                ),
            )
        else:  # Defensive; target is bounded by maximum_sufficiency.
            scope_index = max(chain, key=lambda index: float(sufficiency[index]))

        # A substantial question-conditioned ring means the selected node may
        # omit useful context.  Ascend exactly one candidate level only when
        # the parent remains nearly as sufficient, avoiding unconditional
        # large-box inflation.
        ring_ratio = float(candidates[scope_index].get("ptea_context_ring_mass", 0.0)) / max(
            float(candidates[scope_index].get("ptea_box_mass", 0.0)), 1e-8
        )
        parent_id = candidates[scope_index].get("candidate_parent_id")
        parent_index = index_by_node.get(int(parent_id)) if parent_id is not None else None
        context_promoted = False
        if (
            parent_index is not None
            and parent_index in chain
            and ring_ratio >= context_ring_mass_ratio
            and float(sufficiency[parent_index])
            >= float(sufficiency[scope_index]) - sufficiency_tolerance
        ):
            scope_index = parent_index
            context_promoted = True

        scope = candidates[scope_index]
        core_peak_grid = tuple(int(value) for value in candidates[core_index]["peak_grid"])
        core_peak_policy = [
            int(round(1000.0 * core_peak_grid[0] / max(int(scope.get("map_width", 0)), 1))),
            int(round(1000.0 * core_peak_grid[1] / max(int(scope.get("map_height", 0)), 1))),
        ]
        # Older serialized candidates do not carry map dimensions.  In that
        # case compare peaks directly and rely on IoU for spatial suppression.
        if not scope.get("map_width") or not scope.get("map_height"):
            core_peak_policy = [-1, -1]
        if any(
            core_peak_policy != [-1, -1]
            and _point_inside_policy_box(core_peak_policy, row["bbox"])
            for row in selected
        ):
            continue
        if any(
            policy_iou(scope["bbox"], row["bbox"]) > maximum_pairwise_iou
            for row in selected
        ):
            continue
        selected.append(
            {
                **scope,
                "predicted_relevance": core_relevance,
                "predicted_sufficiency": float(sufficiency[scope_index]),
                "predicted_utility": float(sufficiency[scope_index]),
                "csr_core_node_id": int(candidates[core_index]["node_id"]),
                "csr_scope_node_id": int(scope["node_id"]),
                "csr_scope_hops": int(chain.index(scope_index)),
                "csr_branch_max_sufficiency": maximum_sufficiency,
                "csr_context_ring_ratio": ring_ratio,
                "csr_context_promoted": context_promoted,
                "csr_maximum_scope_hops": int(maximum_scope_hops),
                "csr_branch_score": float(branch_scores[core_index]),
            }
        )
    return selected


__all__ = [
    "EVICONTOUR_FEATURE_NAMES",
    "EviContourUtilityHead",
    "build_evicontour_inference_group",
    "evicontour_feature_vector",
    "load_evicontour_utility_head",
    "predict_evicontour_portfolio",
    "predict_evicontour_csr_portfolio",
    "select_evicontour_csr_portfolio",
    "select_evicontour_portfolio",
]
