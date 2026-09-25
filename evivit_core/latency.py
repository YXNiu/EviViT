"""Shared inference-latency instrumentation for frozen EviViT evaluations."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any


class FirstTokenTimer:
    """A Transformers logits processor that timestamps the first decoded token.

    The callback runs after the multimodal prefill has produced its first-token
    logits.  Synchronizing before the timestamp makes the wall-clock reading a
    valid CUDA completion time rather than a CPU launch time.
    """

    def __init__(self, synchronize: Callable[[], None]) -> None:
        self._synchronize = synchronize
        self.first_token_time: float | None = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self.first_token_time is None:
            self._synchronize()
            self.first_token_time = time.perf_counter()
        return scores


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated percentile."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_latency(
    rows: Sequence[dict[str, Any]],
    *,
    warmup_samples: int,
) -> dict[str, Any]:
    """Summarize measured rows after excluding a declared warmup prefix."""

    measured = [row for row in rows if isinstance(row.get("timing"), dict)]
    retained = measured[max(0, int(warmup_samples)) :]
    metrics = (
        "request_to_first_token_seconds",
        "end_to_end_seconds",
        "global_evidence_pass_seconds",
        "evidence_allocation_seconds",
        "fine_reread_and_fusion_seconds",
        "prompt_and_embedding_seconds",
        "evidence_and_vision_seconds",
        "generation_seconds",
        "decode_seconds",
    )
    summary: dict[str, Any] = {
        "recorded_samples": len(measured),
        "warmup_samples": min(max(0, int(warmup_samples)), len(measured)),
        "measured_samples": len(retained),
    }
    for metric in metrics:
        values = [
            float(row["timing"][metric])
            for row in retained
            if row["timing"].get(metric) is not None
        ]
        if not values:
            continue
        summary[metric] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p90": percentile(values, 0.9),
            "min": min(values),
            "max": max(values),
        }
    generated = [
        int(row["timing"]["generated_tokens"])
        for row in retained
        if row["timing"].get("generated_tokens") is not None
    ]
    if generated:
        summary["generated_tokens"] = {
            "mean": statistics.fmean(generated),
            "median": statistics.median(generated),
        }
    return summary
