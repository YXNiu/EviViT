"""Fail-closed contracts for the Visual-CoT recovery SFT matrix.

The actual EviViT forward is intentionally reused from
``train_evivit_v3_bridge.py``.  This module contains the small, testable pieces
that must not silently drift between recovery lanes: which components train,
the exact frozen P3/B16 inference contract, optimizer groups, and half-epoch
checkpoint boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterable

import torch


RECOVERY_ADAMW_BETAS = (0.9, 0.95)
RECOVERY_ADAMW_EPS = 1e-8
RECOVERY_EPOCH_ORDER_VERSION = "manifest_prefix_seed_plus_epoch_v1"


@dataclass(frozen=True)
class RecoveryLane:
    name: str
    uses_evivit: bool
    train_language_lora: bool
    train_bridge: bool
    train_ptea: bool = False
    train_h_safe: bool = False

    @property
    def requires_router_aux(self) -> bool:
        """Whether this lane consumes bbox/map supervision."""

        return self.train_ptea or self.train_h_safe


RECOVERY_LANES: dict[str, RecoveryLane] = {
    "B-L": RecoveryLane("B-L", False, True, False),
    "E-L": RecoveryLane("E-L", True, True, False),
    "E-B": RecoveryLane("E-B", True, False, True),
    "E-LB": RecoveryLane("E-LB", True, True, True),
    # Router-supervised lanes use the guarded GPU path in
    # ``train_evivit_v3_bridge.py``.  The separate validation entry point and
    # launch queues remain fail-closed, so these names cannot silently fall
    # through to the answer-only trainer or skip their gradient/geometry gates.
    "E-P": RecoveryLane("E-P", True, False, False, True, False),
    "E-PL": RecoveryLane("E-PL", True, True, False, True, False),
    "E-PLB": RecoveryLane("E-PLB", True, True, True, True, False),
    "E-PLB-C": RecoveryLane("E-PLB-C", True, True, True, True, False),
    "E-PHLB": RecoveryLane("E-PHLB", True, True, True, True, True),
}


def recovery_lane(name: str) -> RecoveryLane:
    try:
        return RECOVERY_LANES[name]
    except KeyError as error:
        raise ValueError(f"unknown recovery lane: {name}") from error


def formal_p3_mismatches(args: Any) -> dict[str, tuple[Any, Any]]:
    """Return every mismatch from the frozen Qwen3-VL-8B P3/B16 contract."""

    expected = {
        "insertion_block": 18,
        "global_token_budget": 2048,
        "fine_token_budget": 3072,
        "max_regions": 3,
        "minimum_region_tokens": 64,
        "evidence_decoder": "trace_split",
        "trace_split_budget_mode": "learned",
        "trace_split_min_context_fraction": 0.10,
        "trace_split_max_context_fraction": 0.25,
        "trace_split_context_expansion_factor": 1.0,
        "trace_split_context_decoder": "top_boxes",
        "patch_exchange": True,
        "patch_exchange_global_scale": 1.5,
        "patch_exchange_local_scale": 2.0,
        "patch_exchange_total_cap_ratio": 1.0,
        "patch_exchange_soft_floor_tokens": 2048,
        "patch_exchange_balanced_soft_floor": True,
        "patch_exchange_preserve_native_global": True,
        "patch_exchange_continuous_soft_floor": True,
        "patch_exchange_continuous_soft_floor_max_tokens": 4096,
        "patch_exchange_continuous_soft_floor_ramp_start_tokens": 4096,
        "patch_exchange_continuous_soft_floor_ramp_end_tokens": 12288,
        "patch_exchange_view_min_pixels": 4096,
        "answer_protocol": "compact_json",
        "fusion": "sparse_bridge",
        "bridge_mode": "bidirectional",
        "visual_output_mode": "append",
    }
    mismatches = {
        key: (getattr(args, key, None), value)
        for key, value in expected.items()
        if getattr(args, key, None) != value
    }
    required_paths = (
        "selector_checkpoint",
        "trace_split_context_need_checkpoint",
        "adaptive_box_head",
    )
    for key in required_paths:
        if getattr(args, key, None) is None:
            mismatches[key] = (None, "required")
    if getattr(args, "warm_start_bridge_checkpoint", None) is None:
        mismatches["warm_start_bridge_checkpoint"] = (None, "required")
    return mismatches


def assert_formal_p3_contract(args: Any) -> None:
    mismatches = formal_p3_mismatches(args)
    if mismatches:
        raise ValueError(f"recovery SFT differs from frozen P3/B16: {mismatches}")


def half_epoch_micro_steps(rows: int, gradient_accumulation: int) -> int:
    """Return an exact half-epoch boundary that is also an optimizer boundary."""

    if rows <= 0 or gradient_accumulation <= 0:
        raise ValueError("rows and gradient accumulation must be positive")
    if rows % 2:
        raise ValueError("half-epoch checkpoints require an even row count")
    boundary = rows // 2
    if boundary % gradient_accumulation:
        raise ValueError(
            "half-epoch boundary must align with gradient accumulation"
        )
    return boundary


def epoch_order(count: int, epoch: int, seed: int) -> list[int]:
    """Generate the shared Base/EviViT order from the original manifest.

    Each epoch starts from ``range(count)``.  In particular, epoch N never
    reshuffles epoch N-1's already-permuted list; this makes the B-L and EviViT
    lanes exactly paired across all three epochs and after resume.
    """

    if count <= 0 or epoch < 0:
        raise ValueError("count must be positive and epoch non-negative")
    values = list(range(count))
    random.Random(seed + epoch).shuffle(values)
    return values


def unique_parameters(groups: Iterable[Iterable[torch.nn.Parameter]]) -> list[torch.nn.Parameter]:
    result: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for group in groups:
        for parameter in group:
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            result.append(parameter)
    return result


def assert_recovery_trainable_scope(
    *,
    lane: RecoveryLane,
    language_named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    bridge_named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    all_named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    ptea_named_parameters: Iterable[tuple[str, torch.nn.Parameter]] = (),
    h_safe_named_parameters: Iterable[tuple[str, torch.nn.Parameter]] = (),
) -> list[str]:
    """Fail unless trainable tensors exactly match the lane whitelist.

    The host ViT, merger, ContextNeed and base LLM are never members of the
    whitelist.  Language parameters are additionally restricted to PEFT LoRA
    tensors so an accidentally unfrozen base projection fails before training.
    """

    language_ids = {
        id(parameter)
        for name, parameter in language_named_parameters
        if parameter.requires_grad and "lora_" in name
    }
    bridge_ids = {
        id(parameter)
        for _, parameter in bridge_named_parameters
        if parameter.requires_grad
    }
    ptea_ids = {
        id(parameter)
        for _, parameter in ptea_named_parameters
        if parameter.requires_grad
    }
    h_safe_ids = {
        id(parameter)
        for _, parameter in h_safe_named_parameters
        if parameter.requires_grad
    }
    expected = set()
    if lane.train_language_lora:
        if not language_ids:
            raise RuntimeError(f"{lane.name} has no trainable language LoRA")
        expected.update(language_ids)
    elif language_ids:
        raise RuntimeError(f"{lane.name} unexpectedly trains language LoRA")
    if lane.train_bridge:
        if not bridge_ids:
            raise RuntimeError(f"{lane.name} has no trainable Bridge parameters")
        expected.update(bridge_ids)
    elif bridge_ids:
        raise RuntimeError(f"{lane.name} unexpectedly trains Bridge parameters")
    if lane.train_ptea:
        if not ptea_ids:
            raise RuntimeError(f"{lane.name} has no trainable PTEA parameters")
        expected.update(ptea_ids)
    elif ptea_ids:
        raise RuntimeError(f"{lane.name} unexpectedly trains PTEA parameters")
    if lane.train_h_safe:
        if not h_safe_ids:
            raise RuntimeError(f"{lane.name} has no trainable H-Safe parameters")
        expected.update(h_safe_ids)
    elif h_safe_ids:
        raise RuntimeError(f"{lane.name} unexpectedly trains H-Safe parameters")

    actual = {
        id(parameter): name
        for name, parameter in all_named_parameters
        if parameter.requires_grad
    }
    unexpected = sorted(name for identity, name in actual.items() if identity not in expected)
    missing = expected.difference(actual)
    if unexpected or missing:
        raise RuntimeError(
            f"{lane.name} trainable scope mismatch: unexpected={unexpected}, "
            f"missing_parameter_count={len(missing)}"
        )
    return sorted(actual.values())


def optimizer_groups(
    *,
    lane: RecoveryLane,
    language_parameters: Iterable[torch.nn.Parameter],
    bridge_parameters: Iterable[torch.nn.Parameter],
    language_learning_rate: float,
    bridge_learning_rate: float,
    ptea_parameters: Iterable[torch.nn.Parameter] = (),
    h_safe_parameters: Iterable[torch.nn.Parameter] = (),
    ptea_learning_rate: float = 1e-5,
    h_safe_learning_rate: float = 5e-6,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    language = unique_parameters([language_parameters])
    bridge = unique_parameters([bridge_parameters])
    ptea = unique_parameters([ptea_parameters])
    h_safe = unique_parameters([h_safe_parameters])
    if lane.train_language_lora:
        if not language:
            raise RuntimeError(f"{lane.name} language optimizer group is empty")
        groups.append({"name": "language_lora", "params": language, "lr": language_learning_rate})
    if lane.train_bridge:
        if not bridge:
            raise RuntimeError(f"{lane.name} Bridge optimizer group is empty")
        groups.append({"name": "bridge", "params": bridge, "lr": bridge_learning_rate})
    if lane.train_ptea:
        if not ptea:
            raise RuntimeError(f"{lane.name} PTEA optimizer group is empty")
        groups.append({"name": "ptea", "params": ptea, "lr": ptea_learning_rate})
    if lane.train_h_safe:
        if not h_safe:
            raise RuntimeError(f"{lane.name} H-Safe optimizer group is empty")
        groups.append({"name": "h_safe", "params": h_safe, "lr": h_safe_learning_rate})
    if not groups:
        raise RuntimeError(f"{lane.name} has no trainable optimizer group")
    return groups
