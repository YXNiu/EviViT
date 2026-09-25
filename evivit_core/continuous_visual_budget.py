"""Continuous source-size-aware visual budgets for EviViT.

The policy is deliberately independent of datasets and model outputs.  It
receives only the original image width/height and one frozen configuration,
then returns an integer total/global/fine token plan.  External QA labels,
PTEA scores, and native-Qwen routing are not inputs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class ContinuousBudgetPlan:
    policy: str
    source_pixels: int
    raw_total_tokens: float
    total_tokens: int
    global_tokens: int
    fine_tokens: int
    global_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_common(
    *,
    width: int,
    height: int,
    reference_pixels: float,
    minimum_total_tokens: int,
    maximum_total_tokens: int,
    global_fraction: float,
) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if reference_pixels <= 0:
        raise ValueError("reference_pixels must be positive")
    if minimum_total_tokens <= 0:
        raise ValueError("minimum_total_tokens must be positive")
    if maximum_total_tokens < minimum_total_tokens:
        raise ValueError("maximum_total_tokens must be >= minimum_total_tokens")
    if not 0.0 < global_fraction < 1.0:
        raise ValueError("global_fraction must be strictly between zero and one")


def raw_total_budget(
    width: int,
    height: int,
    config: dict[str, Any],
) -> float:
    """Return a continuous budget before integer grid conversion."""

    policy = str(config["policy"])
    reference_pixels = float(config["reference_pixels"])
    minimum = int(config["minimum_total_tokens"])
    maximum = int(config["maximum_total_tokens"])
    global_fraction = float(config["global_fraction"])
    _validate_common(
        width=width,
        height=height,
        reference_pixels=reference_pixels,
        minimum_total_tokens=minimum,
        maximum_total_tokens=maximum,
        global_fraction=global_fraction,
    )
    source_pixels = float(width * height)
    ratio = source_pixels / reference_pixels
    if policy == "saturating_log":
        slope = float(config["slope"])
        bias = float(config["bias"])
        if slope <= 0:
            raise ValueError("saturating-log slope must be positive")
        logit = slope * math.log(ratio) + bias
        sigmoid = 1.0 / (1.0 + math.exp(-logit))
        value = minimum + (maximum - minimum) * sigmoid
    elif policy == "power":
        exponent = float(config["exponent"])
        scale = float(config["scale"])
        if exponent <= 0 or scale <= 0:
            raise ValueError("power exponent and scale must be positive")
        value = scale * ratio**exponent
    else:
        raise ValueError(f"unknown continuous budget policy: {policy}")
    return min(float(maximum), max(float(minimum), value))


def global_fraction_for_size(
    width: int,
    height: int,
    config: dict[str, Any],
) -> float:
    """Return the frozen Global share for one source image.

    Stage 2-A uses the default ``constant`` policy.  The conditional Stage
    2-B policy is continuous as well: small source images retain a larger
    Global share for spatial topology, while large source images spend a
    larger share on fine rereading.  No question, map, answer, or dataset
    identity is an input.
    """

    reference_pixels = float(config["reference_pixels"])
    constant_fraction = float(config["global_fraction"])
    _validate_common(
        width=width,
        height=height,
        reference_pixels=reference_pixels,
        minimum_total_tokens=int(config["minimum_total_tokens"]),
        maximum_total_tokens=int(config["maximum_total_tokens"]),
        global_fraction=constant_fraction,
    )
    policy = str(config.get("global_fraction_policy", "constant"))
    if policy == "constant":
        return constant_fraction
    if policy != "calibrated_log_clip":
        raise ValueError(f"unknown global-fraction policy: {policy}")

    slope = float(config["global_fraction_log_slope"])
    offset = float(config["global_fraction_offset"])
    minimum = float(config["minimum_global_fraction"])
    maximum = float(config["maximum_global_fraction"])
    if slope >= 0:
        raise ValueError(
            "calibrated-log Global slope must be negative so smaller images "
            "retain at least as much global context"
        )
    if not 0.0 < minimum < maximum < 1.0:
        raise ValueError("global-fraction bounds must satisfy 0 < min < max < 1")
    ratio = float(width * height) / reference_pixels
    value = offset + slope * math.log(ratio)
    return min(maximum, max(minimum, value))


def plan_continuous_budget(
    width: int,
    height: int,
    config: dict[str, Any],
) -> ContinuousBudgetPlan:
    """Convert one continuous budget into exact integer global/fine targets."""

    raw_total = raw_total_budget(width, height, config)
    total = int(round(raw_total))
    global_fraction = global_fraction_for_size(width, height, config)
    global_tokens = int(round(total * global_fraction))
    global_tokens = min(total - 1, max(1, global_tokens))
    fine_tokens = total - global_tokens
    if global_tokens + fine_tokens != total or fine_tokens <= 0:
        raise AssertionError("continuous budget split lost tokens")
    return ContinuousBudgetPlan(
        policy=str(config["policy"]),
        source_pixels=width * height,
        raw_total_tokens=raw_total,
        total_tokens=total,
        global_tokens=global_tokens,
        fine_tokens=fine_tokens,
        global_fraction=global_fraction,
    )


def _bisect_monotone(
    mean_fn: Callable[[float], float],
    target: float,
    *,
    left: float,
    right: float,
    iterations: int = 120,
) -> float:
    if target < mean_fn(left) or target > mean_fn(right):
        raise ValueError("target mean is outside the calibration bracket")
    for _ in range(iterations):
        middle = 0.5 * (left + right)
        if mean_fn(middle) < target:
            left = middle
        else:
            right = middle
    return 0.5 * (left + right)


def calibrate_saturating_bias(
    source_pixels: Sequence[int],
    *,
    reference_pixels: float,
    minimum_total_tokens: int,
    maximum_total_tokens: int,
    slope: float,
    target_mean_tokens: float,
) -> float:
    """Calibrate only the bias so training-image mean equals the target."""

    if not source_pixels or any(value <= 0 for value in source_pixels):
        raise ValueError("source_pixels must be a non-empty positive sequence")

    def mean_for_bias(bias: float) -> float:
        config = {
            "policy": "saturating_log",
            "reference_pixels": reference_pixels,
            "minimum_total_tokens": minimum_total_tokens,
            "maximum_total_tokens": maximum_total_tokens,
            "global_fraction": 0.4,
            "slope": slope,
            "bias": bias,
        }
        return sum(
            raw_total_budget(value, 1, config) for value in source_pixels
        ) / len(source_pixels)

    return _bisect_monotone(
        mean_for_bias,
        target_mean_tokens,
        left=-40.0,
        right=40.0,
    )


def calibrate_power_scale(
    source_pixels: Sequence[int],
    *,
    reference_pixels: float,
    minimum_total_tokens: int,
    maximum_total_tokens: int,
    exponent: float,
    target_mean_tokens: float,
) -> float:
    """Calibrate a positive power-law scale under fixed clipping bounds."""

    if not source_pixels or any(value <= 0 for value in source_pixels):
        raise ValueError("source_pixels must be a non-empty positive sequence")

    def mean_for_log_scale(log_scale: float) -> float:
        config = {
            "policy": "power",
            "reference_pixels": reference_pixels,
            "minimum_total_tokens": minimum_total_tokens,
            "maximum_total_tokens": maximum_total_tokens,
            "global_fraction": 0.4,
            "exponent": exponent,
            "scale": math.exp(log_scale),
        }
        return sum(
            raw_total_budget(value, 1, config) for value in source_pixels
        ) / len(source_pixels)

    log_scale = _bisect_monotone(
        mean_for_log_scale,
        target_mean_tokens,
        left=math.log(1e-3),
        right=math.log(1e6),
    )
    return math.exp(log_scale)


def calibrate_global_fraction_offset(
    source_pixels: Sequence[int],
    *,
    reference_pixels: float,
    log_slope: float,
    minimum_global_fraction: float,
    maximum_global_fraction: float,
    target_mean_global_fraction: float,
) -> float:
    """Calibrate one offset using training image sizes only.

    The slope and clipping range are preregistered.  Bisection adjusts only
    the offset so the training-set mean Global share matches the fixed A4
    control, preventing Stage 2-B from silently increasing its mean Global
    budget.
    """

    if not source_pixels or any(value <= 0 for value in source_pixels):
        raise ValueError("source_pixels must be a non-empty positive sequence")
    if log_slope >= 0:
        raise ValueError("log_slope must be negative")
    if not 0.0 < minimum_global_fraction < maximum_global_fraction < 1.0:
        raise ValueError("invalid global-fraction bounds")
    if not minimum_global_fraction <= target_mean_global_fraction <= maximum_global_fraction:
        raise ValueError("target mean must lie inside the clipping bounds")

    def mean_for_offset(offset: float) -> float:
        config = {
            "policy": "saturating_log",
            "reference_pixels": reference_pixels,
            "minimum_total_tokens": 1,
            "maximum_total_tokens": 2,
            "global_fraction": target_mean_global_fraction,
            "slope": 1.0,
            "bias": 0.0,
            "global_fraction_policy": "calibrated_log_clip",
            "global_fraction_log_slope": log_slope,
            "global_fraction_offset": offset,
            "minimum_global_fraction": minimum_global_fraction,
            "maximum_global_fraction": maximum_global_fraction,
        }
        return sum(
            global_fraction_for_size(value, 1, config)
            for value in source_pixels
        ) / len(source_pixels)

    return _bisect_monotone(
        mean_for_offset,
        target_mean_global_fraction,
        left=-10.0,
        right=10.0,
    )
