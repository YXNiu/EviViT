"""Online question-conditioned evidence allocation for EviViT-v3."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from evivit_core.dense_evidence import decode_top_boxes
from evivit_core.evivit import RegionBudget, allocate_region_budgets
from evivit_core.evivit_v10_reliability import (
    ReliabilityGate,
    features_from_probability,
    interpolate_fine_budget,
    mix_with_uniform,
)
from evivit_core.evitree import (
    HierarchicalSplitPolicyHead,
    build_evitree_plan,
    build_learned_evitree_plan,
    build_sparse_branch_evitree_plan,
)
from evivit_core.evisplit_context_need import context_need_feature_tensor


@dataclass
class MidEvidenceAllocation:
    """Auditable Mid-PTEA map, decoded regions, and exact token budgets."""

    probability_map: torch.Tensor
    candidates: list[dict[str, Any]]
    region_budgets: list[RegionBudget]
    map_entropy: float
    map_peak_probability: float
    evidence_decoder: str
    reliability_gate: float | None = None
    reliability_gate_mode: str = "disabled"
    requested_fine_token_budget: int | None = None
    source_map_entropy: float | None = None
    context_probability_map: torch.Tensor | None = None
    context_map_entropy: float | None = None
    trace_split_context_fraction: float | None = None
    trace_split_context_expansion_factor: float | None = None
    trace_split_context_decoder: str | None = None
    trace_split_budget_mode: str | None = None
    trace_split_secondary_decisive_share: float | None = None
    trace_split_min_context_fraction: float | None = None
    trace_split_max_context_fraction: float | None = None
    trace_split_effective_minimum_region_tokens: int | None = None
    trace_split_context_need_features: list[float] | None = None
    evitree_evidence_need: float | None = None
    evitree_tree_reliability: float | None = None
    evitree_split_threshold: float | None = None
    evitree_split_policy: str | None = None


def _allocate_probability_evidence(
    probability: torch.Tensor,
    *,
    fine_token_budget: int,
    max_regions: int,
    minimum_region_tokens: int,
    decode_scales: Sequence[float],
    decode_mass_power: float,
    decode_area_penalty: float,
    role_prefix: str,
    evidence_decoder: str,
    region_expansion_factor: float,
    topology_union_factor: float,
    contour_relative_threshold: float = 0.40,
    contour_context_margin: float = 0.20,
    contour_minimum_residual_mass: float = 0.05,
    seeded_basin_config: dict[str, Any] | None = None,
    anchored_growth_calibrator: Any | None = None,
    anchored_growth_gate_threshold: float = 0.65,
    evicontour_utility: Any | None = None,
    evicontour_image_size: Sequence[int] | None = None,
    evicontour_question: str | None = None,
    evicontour_csr_sufficiency_threshold: float = 0.65,
    evicontour_csr_sufficiency_tolerance: float = 0.05,
    evicontour_csr_core_relevance_tolerance: float = 0.05,
    evicontour_csr_context_ring_mass_ratio: float = 0.20,
    evicontour_csr_maximum_scope_hops: int = 3,
    evicontour_csr_minimum_parent_sufficiency_gain: float = 0.03,
) -> MidEvidenceAllocation:
    """Decode any normalized map through the exact EviViT budget path."""

    if probability.ndim != 2:
        raise ValueError("probability must have shape [H, W]")
    if fine_token_budget <= 0 or max_regions <= 0:
        raise ValueError("fine_token_budget and max_regions must be positive")
    if region_expansion_factor < 1.0 or topology_union_factor < 1.0:
        raise ValueError("region expansion factors must be at least 1.0")
    probability = probability.float().clamp_min(0)
    total = probability.sum()
    if not torch.isfinite(total) or float(total) <= 0:
        raise ValueError("probability must have positive finite mass")
    probability = probability / total
    if evidence_decoder == "top_boxes":
        candidates = decode_top_boxes(
            probability.cpu().numpy(),
            scales=decode_scales,
            topk=max(5, max_regions),
            mass_power=decode_mass_power,
            area_penalty=decode_area_penalty,
        )
        for index, candidate in enumerate(candidates, 1):
            candidate["portfolio_role"] = f"{role_prefix}_rank_{index}"
    elif evidence_decoder in {
        "residual_focus",
        "topology_union",
        "adaptive_density_mass",
        "adaptive_density_mass_topology",
        "adaptive_pareto",
        "adaptive_pareto_topology",
        "adaptive_contour",
        "seeded_residual_basin",
        "seeded_basin_confidence_fallback",
        "seeded_basin_role_safe_r3",
        "seeded_basin_partial_fill_three",
        "seeded_basin_relaxed_three",
        "anchored_growth_b025",
        "anchored_growth_b035",
        "anchored_growth_calibrated",
        "evicontour_utility",
        "evicontour_csr",
    }:
        if evidence_decoder in {"evicontour_utility", "evicontour_csr"}:
            if evicontour_utility is None:
                raise ValueError("evicontour_utility decoder requires a trained utility head")
            if evicontour_image_size is None or evicontour_question is None:
                raise ValueError("evicontour_utility requires image size and question metadata")
            from evivit_core.evicontour_utility import (
                build_evicontour_inference_group,
                predict_evicontour_csr_portfolio,
                predict_evicontour_portfolio,
            )

            inference_group = build_evicontour_inference_group(
                probability.detach().cpu().numpy(),
                image_size=evicontour_image_size,
                question=evicontour_question,
            )
            if evidence_decoder == "evicontour_csr":
                candidates = predict_evicontour_csr_portfolio(
                    evicontour_utility,
                    inference_group,
                    device="cpu",
                    max_regions=max_regions,
                    sufficiency_threshold=evicontour_csr_sufficiency_threshold,
                    sufficiency_tolerance=evicontour_csr_sufficiency_tolerance,
                    core_relevance_tolerance=(
                        evicontour_csr_core_relevance_tolerance
                    ),
                    context_ring_mass_ratio=(
                        evicontour_csr_context_ring_mass_ratio
                    ),
                    maximum_scope_hops=evicontour_csr_maximum_scope_hops,
                    minimum_parent_sufficiency_gain=(
                        evicontour_csr_minimum_parent_sufficiency_gain
                    ),
                )
            else:
                candidates = predict_evicontour_portfolio(
                    evicontour_utility,
                    inference_group,
                    device="cpu",
                    max_regions=max_regions,
                )
            if not candidates:
                raise RuntimeError("EviContour utility decoder produced no regions")
            for candidate in candidates:
                candidate["score"] = float(candidate["predicted_utility"])
                candidate["mass"] = float(candidate["ptea_box_mass"])
                candidate["portfolio_role"] = f"{role_prefix}_evicontour"
        elif evidence_decoder in {
            "anchored_growth_b025",
            "anchored_growth_b035",
            "anchored_growth_calibrated",
        }:
            from evivit_core.anchored_evidence_growth import (
                AnchoredGrowthConfig,
                decode_calibrated_anchored_evidence_growth,
                decode_anchored_evidence_growth,
            )

            if evidence_decoder == "anchored_growth_calibrated":
                if anchored_growth_calibrator is None:
                    raise ValueError(
                        "calibrated anchored growth requires a calibrator head"
                    )
                candidates = decode_calibrated_anchored_evidence_growth(
                    probability.cpu().numpy(),
                    anchored_growth_calibrator,
                    max_regions=max_regions,
                    gate_threshold=anchored_growth_gate_threshold,
                )
            else:
                candidates = decode_anchored_evidence_growth(
                    probability.cpu().numpy(),
                    max_regions=max_regions,
                    config=AnchoredGrowthConfig(
                        marginal_density_ratio=(
                            0.25 if evidence_decoder.endswith("b025") else 0.35
                        ),
                    ),
                )
            if not candidates:
                raise RuntimeError("anchored growth decoder produced no regions")
            for candidate in candidates:
                candidate["portfolio_role"] = (
                    f"{role_prefix}_{candidate['portfolio_role']}"
                )
        elif evidence_decoder in {
            "seeded_basin_confidence_fallback",
            "seeded_basin_role_safe_r3",
            "seeded_basin_partial_fill_three",
            "seeded_basin_relaxed_three",
        }:
            from evivit_core.seeded_residual_basin import BasinDecoderConfig
            from evivit_core.stage_d_role_safe import decode_role_safe_basins

            config = BasinDecoderConfig(**(seeded_basin_config or {}))
            candidates = decode_role_safe_basins(
                probability.cpu().numpy(),
                max_regions=max_regions,
                config=config,
                mode={
                    "seeded_basin_confidence_fallback": "confidence_fallback",
                    "seeded_basin_role_safe_r3": "replace_r3",
                    "seeded_basin_partial_fill_three": "partial_fill_three",
                    "seeded_basin_relaxed_three": "relaxed_basin_three",
                }[evidence_decoder],
            )
            for candidate in candidates:
                candidate["portfolio_role"] = (
                    f"{role_prefix}_{candidate['portfolio_role']}"
                )
        elif evidence_decoder == "seeded_residual_basin":
            from evivit_core.seeded_residual_basin import (
                BasinDecoderConfig,
                decode_seeded_residual_basins,
            )

            config = BasinDecoderConfig(**(seeded_basin_config or {}))
            candidates = decode_seeded_residual_basins(
                probability.cpu().numpy(),
                max_regions=max_regions,
                config=config,
            )
            if not candidates:
                raise RuntimeError("seeded residual basin decoder produced no regions")
            for candidate in candidates:
                candidate["portfolio_role"] = (
                    f"{role_prefix}_{candidate['portfolio_role']}"
                )
        elif evidence_decoder == "adaptive_contour":
            from evivit_core.simple_evivit_contour import decode_adaptive_contours

            candidates = decode_adaptive_contours(
                probability.cpu().numpy(),
                max_regions=max_regions,
                relative_threshold=contour_relative_threshold,
                context_margin=contour_context_margin,
                minimum_residual_mass=contour_minimum_residual_mass,
            )
            if not candidates:
                raise RuntimeError("adaptive contour decoder produced no regions")
            for candidate in candidates:
                candidate["portfolio_role"] = (
                    f"{role_prefix}_{candidate['portfolio_role']}"
                )
        elif evidence_decoder.startswith("adaptive_"):
            from evivit_core.adaptive_evidence_decoder import (
                decode_adaptive_evidence,
                decode_pareto_evidence,
            )

            decoder = (
                decode_pareto_evidence
                if evidence_decoder.startswith("adaptive_pareto")
                else decode_adaptive_evidence
            )
            candidates = decoder(
                probability.cpu().numpy(),
                max_regions=max_regions,
                include_topology_frame=evidence_decoder.endswith("_topology"),
            )
            if not candidates:
                raise RuntimeError("adaptive evidence decoder produced no regions")
            for candidate in candidates:
                candidate["portfolio_role"] = (
                    f"{role_prefix}_{candidate['portfolio_role']}"
                )
        else:
            # Reuse the frozen v2 numerical decoder, but retain only its compact
            # focus modes.  After selecting one mode it suppresses the surrounding
            # context before locating the next, giving spatial diversity without
            # another network, crop encoder, router, or learned parameter.
            from scripts.build_evimap_portfolio_boxes import decode_portfolio

            if evidence_decoder == "topology_union" and max_regions != 3:
                raise ValueError("topology_union currently requires max_regions=3")
            portfolio = decode_portfolio(
                probability.cpu().numpy(),
                # v2's frozen protocol decodes three focus/context pairs inside an
                # eight-slot portfolio.  Keeping two spare slots also avoids the
                # legacy decoder's exact-capacity boundary before we filter focus.
                top_k=max(2 * max_regions + 2, 8),
                modes=max_regions,
                context_factor=1.8,
                residual_suppression=0.05,
                include_topology_union=False,
                topology_union_factor=topology_union_factor,
            )
            focus = [
                dict(candidate)
                for candidate in portfolio
                if str(candidate.get("portfolio_role", "")).endswith("_focus")
            ]
            if evidence_decoder == "topology_union":
                from scripts.build_evimap_portfolio_boxes import scored_candidate
                from evivit_core.dense_evidence import expand_policy_box

                if not focus:
                    raise RuntimeError("topology decoder produced no focus")
                if len(focus) >= 2:
                    first, second = focus[:2]
                    union_box = [
                        min(first["bbox"][0], second["bbox"][0]),
                        min(first["bbox"][1], second["bbox"][1]),
                        max(first["bbox"][2], second["bbox"][2]),
                        max(first["bbox"][3], second["bbox"][3]),
                    ]
                    topology = scored_candidate(
                        probability.cpu().numpy(),
                        expand_policy_box(union_box, topology_union_factor),
                        role="topology_union",
                    )
                    candidates = [first, second, topology]
                else:
                    # A genuinely single-peaked map has no second object whose
                    # relation can be preserved. Keep the tight focus, its
                    # expanded parent frame, and one distinct broad context view.
                    context = [
                        dict(candidate)
                        for candidate in portfolio
                        if str(candidate.get("portfolio_role", ""))
                        == "mode_1_context"
                    ]
                    broad = [
                        dict(candidate)
                        for candidate in portfolio
                        if str(candidate.get("portfolio_role", "")).startswith(
                            "broad_context_"
                        )
                    ]
                    if not context or not broad:
                        raise RuntimeError("single-peak topology fallback lacks context")
                    context[0]["portfolio_role"] = "topology_parent_context"
                    candidates = [focus[0], context[0], broad[0]]
            else:
                candidates = focus[:max_regions]
            for candidate in candidates:
                candidate["portfolio_role"] = (
                    f"{role_prefix}_{candidate['portfolio_role']}"
                )
    else:
        raise ValueError(f"unknown evidence decoder: {evidence_decoder}")
    if region_expansion_factor > 1.0:
        from scripts.build_evimap_portfolio_boxes import scored_candidate
        from evivit_core.dense_evidence import expand_policy_box

        distribution = probability.cpu().numpy()
        expanded = []
        for candidate in candidates:
            expanded.append(
                scored_candidate(
                    distribution,
                    expand_policy_box(candidate["bbox"], region_expansion_factor),
                    role=str(candidate.get("portfolio_role", "candidate")),
                    source=candidate,
                )
            )
        candidates = expanded
    budgets = allocate_region_budgets(
        candidates,
        total_tokens=fine_token_budget,
        max_regions=max_regions,
        minimum_tokens=minimum_region_tokens,
        mode="evidence",
    )
    distribution = probability.flatten()
    entropy = -(
        distribution.clamp_min(1e-12)
        * distribution.clamp_min(1e-12).log()
    ).sum()
    return MidEvidenceAllocation(
        probability_map=probability,
        candidates=candidates,
        region_budgets=budgets,
        map_entropy=float(entropy.cpu()),
        map_peak_probability=float(probability.max().cpu()),
        evidence_decoder=evidence_decoder,
        requested_fine_token_budget=fine_token_budget,
    )


def _map_entropy(probability: torch.Tensor) -> float:
    distribution = probability.float().flatten().clamp_min(1e-12)
    distribution = distribution / distribution.sum()
    return float((-(distribution * distribution.log()).sum()).cpu())


def _mask_policy_boxes(
    probability: torch.Tensor,
    boxes: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Remove already selected policy boxes from one map."""

    height, width = probability.shape
    result = probability.clone()
    for box in boxes:
        x0, y0, x1, y1 = (int(value) for value in box)
        gx0 = min(width - 1, max(0, int(x0 / 1000 * width)))
        gy0 = min(height - 1, max(0, int(y0 / 1000 * height)))
        gx1 = min(width, max(gx0 + 1, int((x1 * width + 999) / 1000)))
        gy1 = min(height, max(gy0 + 1, int((y1 * height + 999) / 1000)))
        result[gy0:gy1, gx0:gx1] = 0
    if float(result.sum()) <= 0:
        return probability
    return result / result.sum()


def allocate_trace_split_evidence(
    decisive_probability: torch.Tensor,
    context_probability: torch.Tensor,
    *,
    fine_token_budget: int,
    max_regions: int = 3,
    minimum_region_tokens: int = 64,
    context_fraction: float = 0.25,
    context_expansion_factor: float = 1.0,
    context_decoder: str = "top_boxes",
    budget_mode: str = "fixed",
    min_context_fraction: float = 0.10,
    max_context_fraction: float = 0.35,
    context_need_head: Any | None = None,
    adaptive_minimum_region_tokens: bool = False,
    decode_scales: Sequence[float] = (0.2, 0.25, 0.35, 0.5, 0.67),
    decode_mass_power: float = 0.75,
    decode_area_penalty: float = 0.05,
) -> MidEvidenceAllocation:
    """Allocate up to two non-redundant decisive regions and one context region.

    ``max_regions=3`` fixes the slot cap, not the realized per-sample count.
    ``competitive`` mode keeps the total fine-token budget exact, but lets the
    second decisive mode and the context view compete through one analytic
    rule. ``learned`` keeps the same exact budget and regions, but predicts the
    context share with a tiny Trace1144-supervised scalar head.
    """

    if max_regions != 3:
        raise ValueError("trace_split currently requires max_regions=3")
    if not 0.1 <= context_fraction <= 0.5:
        raise ValueError("trace_split context_fraction must be in [0.1, 0.5]")
    if budget_mode not in {"fixed", "competitive", "learned"}:
        raise ValueError(
            "trace_split budget_mode must be fixed, competitive, or learned"
        )
    if not 0.1 <= min_context_fraction <= max_context_fraction <= 0.5:
        raise ValueError(
            "trace_split competitive context range must lie in [0.1, 0.5]"
        )
    if context_expansion_factor < 1.0:
        raise ValueError(
            "trace_split context_expansion_factor must be at least 1.0"
        )
    if context_decoder not in {"top_boxes", "adaptive_density_mass"}:
        raise ValueError(f"unknown trace_split context decoder: {context_decoder}")
    decisive_probability = decisive_probability.float().clamp_min(0)
    decisive_probability /= decisive_probability.sum().clamp_min(1e-12)
    context_probability = context_probability.float().clamp_min(0)
    context_probability /= context_probability.sum().clamp_min(1e-12)
    secondary_decisive_share: float | None = None
    context_need_features: list[float] | None = None
    if budget_mode in {"competitive", "learned"}:
        decisive_probe = _allocate_probability_evidence(
            decisive_probability,
            fine_token_budget=fine_token_budget,
            max_regions=2,
            minimum_region_tokens=minimum_region_tokens,
            decode_scales=decode_scales,
            decode_mass_power=decode_mass_power,
            decode_area_penalty=decode_area_penalty,
            role_prefix="tracesplit_decisive",
            evidence_decoder="residual_focus",
            region_expansion_factor=1.0,
            topology_union_factor=1.0,
        )
        decisive_scores = sorted(
            (float(item.score) for item in decisive_probe.region_budgets),
            reverse=True,
        )
        secondary_decisive_share = (
            decisive_scores[1] if len(decisive_scores) >= 2 else 0.0
        )
        if budget_mode == "competitive":
            normalized_competition = min(
                1.0, max(0.0, secondary_decisive_share / 0.5)
            )
            context_fraction = max_context_fraction - (
                max_context_fraction - min_context_fraction
            ) * normalized_competition
        else:
            if context_need_head is None:
                raise ValueError(
                    "learned trace_split requires context_need_head"
                )
            feature_tensor = context_need_feature_tensor(
                decisive_probability,
                context_probability,
                [item.bbox for item in decisive_probe.region_budgets],
                secondary_decisive_share=secondary_decisive_share,
            )
            context_fraction = float(
                context_need_head(
                    feature_tensor.to(decisive_probability.device).unsqueeze(0)
                )[0]
                .detach()
                .cpu()
            )
            context_need_features = [
                float(value) for value in feature_tensor.detach().cpu()
            ]
    context_tokens = int(round(fine_token_budget * context_fraction))
    decisive_tokens = fine_token_budget - context_tokens
    effective_minimum_region_tokens = int(minimum_region_tokens)
    if adaptive_minimum_region_tokens:
        # Preserve the audited v27b 64-token floor whenever the per-image
        # native-relative budget can afford it.  Only genuinely small images
        # lower the structural floor, and never below the 2x2 merged-token
        # grid required by grid_for_token_budget().
        effective_minimum_region_tokens = max(
            4,
            min(
                int(minimum_region_tokens),
                int(context_tokens),
                int(decisive_tokens // 2),
            ),
        )
    if min(context_tokens, decisive_tokens) < effective_minimum_region_tokens:
        raise ValueError("trace_split token budget is below the region minimum")
    decisive = _allocate_probability_evidence(
        decisive_probability,
        fine_token_budget=decisive_tokens,
        max_regions=2,
        minimum_region_tokens=effective_minimum_region_tokens,
        decode_scales=decode_scales,
        decode_mass_power=decode_mass_power,
        decode_area_penalty=decode_area_penalty,
        role_prefix="tracesplit_decisive",
        evidence_decoder="residual_focus",
        region_expansion_factor=1.0,
        topology_union_factor=1.0,
    )
    residual_context = _mask_policy_boxes(
        context_probability,
        [budget.bbox for budget in decisive.region_budgets],
    )
    context = _allocate_probability_evidence(
        residual_context,
        fine_token_budget=context_tokens,
        max_regions=1,
        minimum_region_tokens=effective_minimum_region_tokens,
        decode_scales=decode_scales,
        decode_mass_power=decode_mass_power,
        decode_area_penalty=decode_area_penalty,
        role_prefix="tracesplit_context",
        evidence_decoder=context_decoder,
        region_expansion_factor=context_expansion_factor,
        topology_union_factor=1.0,
    )
    combined_budgets: list[RegionBudget] = []
    for source, quota in (
        (decisive.region_budgets, 1.0 - context_fraction),
        (context.region_budgets, context_fraction),
    ):
        for budget in source:
            combined_budgets.append(
                RegionBudget(
                    bbox=budget.bbox,
                    token_budget=budget.token_budget,
                    score=budget.score * quota,
                    role=budget.role,
                    source_rank=len(combined_budgets) + 1,
                )
            )
    if sum(item.token_budget for item in combined_budgets) != fine_token_budget:
        raise AssertionError("trace_split lost fine-token budget")
    return MidEvidenceAllocation(
        probability_map=decisive_probability,
        context_probability_map=residual_context,
        candidates=[*decisive.candidates, *context.candidates],
        region_budgets=combined_budgets,
        map_entropy=_map_entropy(decisive_probability),
        context_map_entropy=_map_entropy(residual_context),
        map_peak_probability=float(decisive_probability.max().cpu()),
        evidence_decoder="trace_split",
        requested_fine_token_budget=fine_token_budget,
        trace_split_context_fraction=context_fraction,
        trace_split_context_expansion_factor=context_expansion_factor,
        trace_split_context_decoder=context_decoder,
        trace_split_budget_mode=budget_mode,
        trace_split_secondary_decisive_share=secondary_decisive_share,
        trace_split_min_context_fraction=(
            min_context_fraction if budget_mode == "competitive" else None
        ),
        trace_split_max_context_fraction=(
            max_context_fraction
            if budget_mode in {"competitive", "learned"}
            else None
        ),
        trace_split_effective_minimum_region_tokens=(
            effective_minimum_region_tokens
        ),
        trace_split_context_need_features=context_need_features,
    )


@torch.inference_mode()
def qwen_projected_question_tokens(
    model: Any,
    tokenizer: Any,
    question: str,
    projection: torch.Tensor,
    *,
    device: str,
    max_length: int = 128,
) -> torch.Tensor:
    """Reproduce the frozen 512-D question-token protocol used by Mid-PTEA."""

    clean = " ".join(str(question).replace("<image>", " ").split())
    encoded = tokenizer(
        clean,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    token_ids = encoded.input_ids.to(device)
    embeddings = model.get_input_embeddings()(token_ids).squeeze(0)
    if int(embeddings.shape[-1]) != int(projection.shape[0]):
        raise ValueError("question embedding and projection dimensions do not match")
    return F.normalize(embeddings.float() @ projection.float(), dim=-1).to(
        torch.float16
    )


def _forward_mid_ptea_selector(
    selector: Any,
    visual_feature_grid: torch.Tensor | dict[int, torch.Tensor],
    question_tokens: torch.Tensor,
    *,
    selector_attention_backend: str | None = None,
) -> Any:
    """Run only the neural selector, with no implicit autograd policy."""
    required_blocks = tuple(
        int(value)
        for value in getattr(selector, "required_visual_blocks", ())
    )
    if required_blocks:
        if not isinstance(visual_feature_grid, dict):
            raise ValueError(
                "multi-scale selector requires a block-to-feature-grid mapping"
            )
        missing = set(required_blocks).difference(visual_feature_grid)
        if missing:
            raise ValueError(
                f"multi-scale selector is missing blocks: {sorted(missing)}"
            )
        if any(
            visual_feature_grid[block].ndim != 3
            for block in required_blocks
        ):
            raise ValueError(
                "all multi-scale visual feature grids must have shape [H, W, D]"
            )
    else:
        if not isinstance(visual_feature_grid, torch.Tensor):
            raise ValueError("single-scale selector requires one feature grid")
        if visual_feature_grid.ndim != 3:
            raise ValueError(
                "visual_feature_grid must have shape [H, W, D]"
            )
    if question_tokens.ndim != 2:
        raise ValueError("question_tokens must have shape [L, D]")
    question_mask = torch.ones(
        question_tokens.shape[0],
        dtype=torch.bool,
        device=question_tokens.device,
    )
    if selector_attention_backend is None:
        selector_attention_context = nullcontext()
    else:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except ImportError as error:  # pragma: no cover - old PyTorch guard
            raise RuntimeError(
                "scoped selector attention requires torch.nn.attention.sdpa_kernel"
            ) from error
        backend = {
            "math": SDPBackend.MATH,
            "flash": SDPBackend.FLASH_ATTENTION,
            "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        }.get(selector_attention_backend)
        if backend is None:
            raise ValueError(
                f"unsupported selector attention backend: {selector_attention_backend}"
            )
        selector_attention_context = sdpa_kernel(backend)
    with selector_attention_context:
        return selector(
            visual_feature_grid, question_tokens, question_mask
        )


def forward_mid_ptea_training_paths(
    selector: Any,
    visual_feature_grid: torch.Tensor | dict[int, torch.Tensor],
    question_tokens: torch.Tensor,
    *,
    selector_attention_backend: str | None = None,
) -> Any:
    """Return differentiable maps plus detached copies for discrete decoding.

    This entry point is intentionally not decorated with ``inference_mode``.
    It is reserved for the recovery trainer; normal inference continues to use
    :func:`allocate_mid_ptea_evidence` below.
    """

    from evivit_core.evivit_recovery_router_sft import split_ptea_training_paths

    selector_output = _forward_mid_ptea_selector(
        selector,
        visual_feature_grid,
        question_tokens,
        selector_attention_backend=selector_attention_backend,
    )
    if not isinstance(selector_output, tuple) or len(selector_output) < 2:
        raise ValueError(
            "trainable recovery PTEA requires decisive and context selector outputs"
        )
    return split_ptea_training_paths(selector_output[0][0], selector_output[1][0])


@torch.inference_mode()
def allocate_mid_ptea_evidence(
    selector: Any,
    visual_feature_grid: torch.Tensor | dict[int, torch.Tensor],
    question_tokens: torch.Tensor,
    *,
    fine_token_budget: int,
    max_regions: int = 3,
    minimum_region_tokens: int = 64,
    decode_scales: Sequence[float] = (0.2, 0.25, 0.35, 0.5, 0.67),
    decode_mass_power: float = 0.75,
    decode_area_penalty: float = 0.05,
    evidence_decoder: str = "top_boxes",
    region_expansion_factor: float = 1.0,
    topology_union_factor: float = 1.0,
    selector_attention_backend: str | None = None,
    trace_split_context_fraction: float = 0.25,
    trace_split_context_expansion_factor: float = 1.0,
    trace_split_context_decoder: str = "top_boxes",
    trace_split_budget_mode: str = "fixed",
    trace_split_min_context_fraction: float = 0.10,
    trace_split_max_context_fraction: float = 0.35,
    trace_split_context_need_head: Any | None = None,
    trace_split_adaptive_minimum_region_tokens: bool = False,
    contour_relative_threshold: float = 0.40,
    contour_context_margin: float = 0.20,
    contour_minimum_residual_mass: float = 0.05,
    seeded_basin_config: dict[str, Any] | None = None,
    anchored_growth_calibrator: Any | None = None,
    anchored_growth_gate_threshold: float = 0.65,
    evicontour_utility: Any | None = None,
    evicontour_image_size: Sequence[int] | None = None,
    evicontour_question: str | None = None,
    evicontour_csr_sufficiency_threshold: float = 0.65,
    evicontour_csr_sufficiency_tolerance: float = 0.05,
    evicontour_csr_core_relevance_tolerance: float = 0.05,
    evicontour_csr_context_ring_mass_ratio: float = 0.20,
    evicontour_csr_maximum_scope_hops: int = 3,
    evicontour_csr_minimum_parent_sufficiency_gain: float = 0.03,
) -> MidEvidenceAllocation:
    """Predict one map and allocate a fixed fine-token budget without crops."""

    selector_output = _forward_mid_ptea_selector(
        selector,
        visual_feature_grid,
        question_tokens,
        selector_attention_backend=selector_attention_backend,
    )
    if evidence_decoder == "trace_split":
        if (
            not isinstance(selector_output, tuple)
            or len(selector_output) < 2
        ):
            raise ValueError(
                "trace_split requires decisive and context selector outputs"
            )
        decisive_logits = selector_output[0][0]
        context_logits = selector_output[1][0]
        decisive_probability = torch.softmax(
            decisive_logits.flatten(), dim=0
        ).reshape_as(decisive_logits)
        context_probability = torch.softmax(
            context_logits.flatten(), dim=0
        ).reshape_as(context_logits)
        return allocate_trace_split_evidence(
            decisive_probability,
            context_probability,
            fine_token_budget=fine_token_budget,
            max_regions=max_regions,
            minimum_region_tokens=minimum_region_tokens,
            context_fraction=trace_split_context_fraction,
            context_expansion_factor=trace_split_context_expansion_factor,
            context_decoder=trace_split_context_decoder,
            budget_mode=trace_split_budget_mode,
            min_context_fraction=trace_split_min_context_fraction,
            max_context_fraction=trace_split_max_context_fraction,
            context_need_head=trace_split_context_need_head,
            adaptive_minimum_region_tokens=(
                trace_split_adaptive_minimum_region_tokens
            ),
            decode_scales=decode_scales,
            decode_mass_power=decode_mass_power,
            decode_area_penalty=decode_area_penalty,
        )
    logits = selector_output[0]
    probability = torch.softmax(logits.flatten(), dim=0).reshape_as(logits)
    return _allocate_probability_evidence(
        probability,
        fine_token_budget=fine_token_budget,
        max_regions=max_regions,
        minimum_region_tokens=minimum_region_tokens,
        decode_scales=decode_scales,
        decode_mass_power=decode_mass_power,
        decode_area_penalty=decode_area_penalty,
        role_prefix="mid_ptea",
        evidence_decoder=evidence_decoder,
        region_expansion_factor=region_expansion_factor,
        topology_union_factor=topology_union_factor,
        contour_relative_threshold=contour_relative_threshold,
        contour_context_margin=contour_context_margin,
        contour_minimum_residual_mass=contour_minimum_residual_mass,
        seeded_basin_config=seeded_basin_config,
        anchored_growth_calibrator=anchored_growth_calibrator,
        anchored_growth_gate_threshold=anchored_growth_gate_threshold,
        evicontour_utility=evicontour_utility,
        evicontour_image_size=evicontour_image_size,
        evicontour_question=evicontour_question,
        evicontour_csr_sufficiency_threshold=(
            evicontour_csr_sufficiency_threshold
        ),
        evicontour_csr_sufficiency_tolerance=(
            evicontour_csr_sufficiency_tolerance
        ),
        evicontour_csr_core_relevance_tolerance=(
            evicontour_csr_core_relevance_tolerance
        ),
        evicontour_csr_context_ring_mass_ratio=(
            evicontour_csr_context_ring_mass_ratio
        ),
        evicontour_csr_maximum_scope_hops=(
            evicontour_csr_maximum_scope_hops
        ),
        evicontour_csr_minimum_parent_sufficiency_gain=(
            evicontour_csr_minimum_parent_sufficiency_gain
        ),
    )


@torch.inference_mode()
def reallocate_probability_evidence(
    probability: torch.Tensor,
    *,
    fine_token_budget: int,
    max_regions: int = 3,
    minimum_region_tokens: int = 64,
    decode_scales: Sequence[float] = (0.2, 0.25, 0.35, 0.5, 0.67),
    decode_mass_power: float = 0.75,
    decode_area_penalty: float = 0.05,
    evidence_decoder: str = "residual_focus",
    region_expansion_factor: float = 1.0,
    topology_union_factor: float = 1.0,
    contour_relative_threshold: float = 0.40,
    contour_context_margin: float = 0.20,
    contour_minimum_residual_mass: float = 0.05,
    seeded_basin_config: dict[str, Any] | None = None,
    anchored_growth_calibrator: Any | None = None,
    anchored_growth_gate_threshold: float = 0.65,
) -> MidEvidenceAllocation:
    """Allocate a new budget on an already predicted PTEA probability map.

    EviTree-A uses this public entry point to change only the per-sample fine
    budget.  The PTEA forward pass is therefore executed exactly once.
    """

    return _allocate_probability_evidence(
        probability,
        fine_token_budget=fine_token_budget,
        max_regions=max_regions,
        minimum_region_tokens=minimum_region_tokens,
        decode_scales=decode_scales,
        decode_mass_power=decode_mass_power,
        decode_area_penalty=decode_area_penalty,
        role_prefix="evitree_budget",
        evidence_decoder=evidence_decoder,
        region_expansion_factor=region_expansion_factor,
        topology_union_factor=topology_union_factor,
        contour_relative_threshold=contour_relative_threshold,
        contour_context_margin=contour_context_margin,
        contour_minimum_residual_mass=contour_minimum_residual_mass,
        seeded_basin_config=seeded_basin_config,
        anchored_growth_calibrator=anchored_growth_calibrator,
        anchored_growth_gate_threshold=anchored_growth_gate_threshold,
    )


@torch.inference_mode()
def allocate_evitree_hierarchical_evidence(
    probability: torch.Tensor,
    *,
    fine_token_budget: int,
    minimum_leaf_tokens: int = 64,
    maximum_depth: int = 5,
    maximum_leaves: int = 12,
    reliability: float = 0.80,
    target_mass: float = 0.90,
    evidence_need: float | None = None,
    split_policy: HierarchicalSplitPolicyHead | None = None,
    split_threshold: float | None = None,
    sparse_branches: bool = False,
    maximum_branches: int = 3,
    minimum_branch_depth: int = 0,
    branch_density_power: float = 0.20,
    recenter_branch_tips: bool = False,
    branch_expansion_factor: float = 1.0,
    maximum_branch_iou: float = 1.0,
) -> MidEvidenceAllocation:
    """Decode a PTEA map into non-overlapping hierarchical evidence leaves."""

    probability_np = probability.detach().float().cpu().numpy()
    if split_policy is None:
        plan = build_evitree_plan(
            probability_np,
            fine_token_budget=fine_token_budget,
            minimum_leaf_tokens=minimum_leaf_tokens,
            maximum_depth=maximum_depth,
            maximum_leaves=maximum_leaves,
            reliability=reliability,
            target_mass=target_mass,
        )
        decoder = "evitree_hierarchical"
    else:
        if split_threshold is None:
            raise ValueError("learned EviTree requires split_threshold")
        if sparse_branches:
            plan = build_sparse_branch_evitree_plan(
                probability_np,
                split_policy,
                split_threshold=split_threshold,
                fine_token_budget=fine_token_budget,
                minimum_leaf_tokens=minimum_leaf_tokens,
                maximum_depth=maximum_depth,
                maximum_frontier_leaves=maximum_leaves,
                maximum_branches=maximum_branches,
                minimum_branch_depth=minimum_branch_depth,
                density_power=branch_density_power,
                recenter_branch_tips=recenter_branch_tips,
                branch_expansion_factor=branch_expansion_factor,
                maximum_branch_iou=maximum_branch_iou,
                reliability=reliability,
                target_mass=target_mass,
            )
            decoder = "evitree_sparse_branches"
        else:
            plan = build_learned_evitree_plan(
                probability_np,
                split_policy,
                split_threshold=split_threshold,
                fine_token_budget=fine_token_budget,
                minimum_leaf_tokens=minimum_leaf_tokens,
                maximum_depth=maximum_depth,
                maximum_leaves=maximum_leaves,
                reliability=reliability,
            )
            decoder = "evitree_learned_hierarchy"
    candidates = []
    budgets = []
    for rank, leaf in enumerate(plan.leaves, 1):
        role = f"evitree_leaf_d{leaf.depth}"
        candidate = {
            "bbox": list(leaf.bbox_xyxy_1000),
            "score": leaf.allocation_mass,
            "mass": leaf.evidence_mass,
            "portfolio_role": role,
            "tree_node_id": leaf.node_id,
            "tree_parent_id": leaf.parent_id,
            "tree_path": leaf.path,
            "tree_depth": leaf.depth,
            "tree_area_ratio": leaf.area_ratio,
        }
        candidates.append(candidate)
        budgets.append(
            RegionBudget(
                bbox=leaf.bbox_xyxy_1000,
                token_budget=leaf.token_budget,
                score=leaf.allocation_mass,
                role=role,
                source_rank=rank,
            )
        )
    normalized = probability.float().clamp_min(0)
    normalized = normalized / normalized.sum().clamp_min(1e-12)
    return MidEvidenceAllocation(
        probability_map=normalized,
        candidates=candidates,
        region_budgets=budgets,
        map_entropy=_map_entropy(normalized),
        map_peak_probability=float(normalized.max().cpu()),
        evidence_decoder=decoder,
        requested_fine_token_budget=int(fine_token_budget),
        evitree_evidence_need=evidence_need,
        evitree_tree_reliability=float(reliability),
        evitree_split_threshold=split_threshold,
        evitree_split_policy=(
            (
                "human_trace_supervised_sparse_branches"
                if sparse_branches
                else "human_trace_supervised"
            )
            if split_policy is not None
            else "heuristic"
        ),
    )


@torch.inference_mode()
def allocate_reliability_calibrated_evidence(
    selector: Any,
    visual_feature_grid: torch.Tensor,
    question_tokens: torch.Tensor,
    *,
    question: str,
    gate_mode: str,
    reliability_gate: ReliabilityGate | None,
    minimum_fine_token_budget: int,
    maximum_fine_token_budget: int,
    max_regions: int = 3,
    minimum_region_tokens: int = 64,
    decode_scales: Sequence[float] = (0.2, 0.25, 0.35, 0.5, 0.67),
    decode_mass_power: float = 0.75,
    decode_area_penalty: float = 0.05,
    evidence_decoder: str = "residual_focus",
    region_expansion_factor: float = 1.0,
    topology_union_factor: float = 1.0,
    selector_attention_backend: str | None = None,
    soften_probability: bool = True,
    dynamic_fine_budget: bool = True,
) -> MidEvidenceAllocation:
    """Use training-calibrated PTEA reliability as a soft inference prior."""

    if gate_mode not in {"zero", "one", "learned", "entropy_only"}:
        raise ValueError(f"unknown v10 gate mode: {gate_mode}")
    if gate_mode in {"learned", "entropy_only"} and reliability_gate is None:
        raise ValueError(f"{gate_mode} requires a reliability gate artifact")
    if visual_feature_grid.ndim != 3:
        raise ValueError("visual_feature_grid must have shape [H, W, D]")
    question_mask = torch.ones(
        question_tokens.shape[0],
        dtype=torch.bool,
        device=question_tokens.device,
    )
    selector_attention_context = nullcontext()
    if selector_attention_backend is not None:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backend = {
            "math": SDPBackend.MATH,
            "flash": SDPBackend.FLASH_ATTENTION,
            "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        }.get(selector_attention_backend)
        if backend is None:
            raise ValueError(
                f"unsupported selector attention backend: {selector_attention_backend}"
            )
        selector_attention_context = sdpa_kernel(backend)
    with selector_attention_context:
        logits = selector(
            visual_feature_grid, question_tokens, question_mask
        )[0]
    source_probability = torch.softmax(logits.flatten(), dim=0).reshape_as(logits)
    if gate_mode == "zero":
        gate_value = 0.0
    elif gate_mode == "one":
        gate_value = 1.0
    else:
        features, _ = features_from_probability(
            source_probability,
            question=question,
            decode_scales=decode_scales,
            decode_mass_power=decode_mass_power,
            decode_area_penalty=decode_area_penalty,
            topk=32,
        )
        assert reliability_gate is not None
        gate_value = reliability_gate.predict(features)
    requested_fine_tokens = (
        interpolate_fine_budget(
            gate_value,
            minimum_tokens=minimum_fine_token_budget,
            maximum_tokens=maximum_fine_token_budget,
            quantum=minimum_region_tokens,
        )
        if dynamic_fine_budget
        else maximum_fine_token_budget
    )
    probability = (
        mix_with_uniform(source_probability, gate_value)
        if soften_probability
        else source_probability
    )
    if requested_fine_tokens == 0:
        return MidEvidenceAllocation(
            probability_map=probability,
            candidates=[],
            region_budgets=[],
            map_entropy=_map_entropy(probability),
            map_peak_probability=float(probability.max().cpu()),
            evidence_decoder=evidence_decoder,
            reliability_gate=gate_value,
            reliability_gate_mode=gate_mode,
            requested_fine_token_budget=0,
            source_map_entropy=_map_entropy(source_probability),
        )
    allocation = _allocate_probability_evidence(
        probability,
        fine_token_budget=requested_fine_tokens,
        max_regions=max_regions,
        minimum_region_tokens=minimum_region_tokens,
        decode_scales=decode_scales,
        decode_mass_power=decode_mass_power,
        decode_area_penalty=decode_area_penalty,
        role_prefix=f"v10_{gate_mode}",
        evidence_decoder=evidence_decoder,
        region_expansion_factor=region_expansion_factor,
        topology_union_factor=topology_union_factor,
    )
    allocation.reliability_gate = gate_value
    allocation.reliability_gate_mode = gate_mode
    allocation.requested_fine_token_budget = requested_fine_tokens
    allocation.source_map_entropy = _map_entropy(source_probability)
    return allocation


@torch.inference_mode()
def allocate_control_evidence(
    visual_feature_grid: torch.Tensor,
    *,
    policy: str,
    fine_token_budget: int,
    max_regions: int = 3,
    minimum_region_tokens: int = 64,
    random_seed: int = 0,
    decode_scales: Sequence[float] = (0.2, 0.25, 0.35, 0.5, 0.67),
    decode_mass_power: float = 0.75,
    decode_area_penalty: float = 0.05,
) -> MidEvidenceAllocation:
    """Build equal-budget controls without using the human-trained map head.

    ``uniform`` removes all spatial evidence, ``random`` provides a seeded
    spatial control, and ``feature_norm`` uses question-agnostic frozen-ViT
    activation magnitude.  All policies share the same decoder and token
    allocator as Mid-PTEA, so QA differences cannot be attributed to budgets.
    """

    if visual_feature_grid.ndim != 3:
        raise ValueError("visual_feature_grid must have shape [H, W, D]")
    height, width = visual_feature_grid.shape[:2]
    if policy == "uniform":
        probability = torch.ones(
            height, width, dtype=torch.float32, device=visual_feature_grid.device
        )
    elif policy == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(random_seed))
        probability = torch.rand(height, width, generator=generator).to(
            visual_feature_grid.device
        )
    elif policy == "feature_norm":
        probability = visual_feature_grid.float().pow(2).mean(dim=-1).sqrt()
        probability = probability - probability.min() + 1e-8
    else:
        raise ValueError(f"unknown control evidence policy: {policy}")
    return _allocate_probability_evidence(
        probability,
        fine_token_budget=fine_token_budget,
        max_regions=max_regions,
        minimum_region_tokens=minimum_region_tokens,
        decode_scales=decode_scales,
        decode_mass_power=decode_mass_power,
        decode_area_penalty=decode_area_penalty,
        role_prefix=policy,
        evidence_decoder="top_boxes",
        region_expansion_factor=1.0,
        topology_union_factor=1.0,
    )
