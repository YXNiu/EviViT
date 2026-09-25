"""Deterministic counterfactual box portfolios for EviViT-v6 audits.

The functions in this module are deliberately model-free.  They perturb the
verified AdaptiveBox portfolio around its current solution so that a frozen
VLM can measure whether box geometry still has causal QA headroom.  Candidate
boxes are diagnostic/training-time objects only; they are never exposed to the
test-time model as oracle inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


PolicyBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class BoxPortfolio:
    """One named perturbation of a complete ordered region portfolio."""

    name: str
    boxes: tuple[PolicyBox, ...]
    changed_region: int | None
    operation: str


def _bounded_interval(center: float, side: float) -> tuple[float, float]:
    side = min(1000.0, max(1.0, side))
    center = min(1000.0 - side / 2.0, max(side / 2.0, center))
    return center - side / 2.0, center + side / 2.0


def transform_policy_box(
    box: Sequence[int | float],
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    shift_x_fraction: float = 0.0,
    shift_y_fraction: float = 0.0,
) -> PolicyBox:
    """Scale and translate a policy box while preserving a valid image box.

    Shifts are expressed as fractions of the *original box* width/height.  At
    an image boundary we move the interval inward instead of silently
    shrinking it, matching the safety convention used by AdaptiveBox.
    """

    if len(box) != 4:
        raise ValueError("box must contain xyxy coordinates")
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("box scales must be positive")
    x0, y0, x1, y1 = (float(value) for value in box)
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        raise ValueError(f"invalid policy box: {box}")
    width, height = x1 - x0, y1 - y0
    center_x = (x0 + x1) / 2.0 + shift_x_fraction * width
    center_y = (y0 + y1) / 2.0 + shift_y_fraction * height
    left, right = _bounded_interval(center_x, width * scale_x)
    top, bottom = _bounded_interval(center_y, height * scale_y)
    rounded = (
        int(round(left)),
        int(round(top)),
        int(round(right)),
        int(round(bottom)),
    )
    # Rounding can collapse a one-policy-unit interval at the boundary.
    rx0, ry0, rx1, ry1 = rounded
    if rx1 <= rx0:
        rx1 = min(1000, rx0 + 1)
        rx0 = max(0, rx1 - 1)
    if ry1 <= ry0:
        ry1 = min(1000, ry0 + 1)
        ry0 = max(0, ry1 - 1)
    return rx0, ry0, rx1, ry1


def counterfactual_portfolios(
    boxes: Sequence[Sequence[int | float]],
    *,
    profile: str = "full",
) -> tuple[BoxPortfolio, ...]:
    """Return deterministic local counterfactuals around an ordered portfolio.

    ``quick`` tests scale sensitivity with twelve portfolios.  ``full`` adds
    independent x/y translations for every region.  The identity portfolio is
    always first, which makes resume files and paired comparisons stable.
    """

    if profile not in {"quick", "full"}:
        raise ValueError("profile must be quick or full")
    anchors = tuple(tuple(int(round(float(v))) for v in box) for box in boxes)
    if not anchors:
        raise ValueError("at least one region is required")
    # Validate and canonicalize every anchor through an identity transform.
    anchors = tuple(transform_policy_box(box) for box in anchors)
    output = [BoxPortfolio("identity", anchors, None, "identity")]

    def add_single(index: int, name: str, operation: str, **kwargs: float) -> None:
        changed = list(anchors)
        changed[index] = transform_policy_box(anchors[index], **kwargs)
        output.append(
            BoxPortfolio(
                f"r{index + 1}_{name}", tuple(changed), index, operation
            )
        )

    for index in range(len(anchors)):
        add_single(index, "shrink080", "scale", scale_x=0.8, scale_y=0.8)
        add_single(index, "expand120", "scale", scale_x=1.2, scale_y=1.2)
        add_single(index, "expand150", "scale", scale_x=1.5, scale_y=1.5)

    for scale in (1.2, 1.5):
        output.append(
            BoxPortfolio(
                f"all_expand{int(scale * 100):03d}",
                tuple(
                    transform_policy_box(box, scale_x=scale, scale_y=scale)
                    for box in anchors
                ),
                None,
                "scale_all",
            )
        )

    if profile == "full":
        for index in range(len(anchors)):
            for name, dx, dy in (
                ("left025", -0.25, 0.0),
                ("right025", 0.25, 0.0),
                ("up025", 0.0, -0.25),
                ("down025", 0.0, 0.25),
            ):
                add_single(
                    index,
                    name,
                    "translate",
                    shift_x_fraction=dx,
                    shift_y_fraction=dy,
                )

    # Boundary clipping may make two nominal perturbations identical.  Keep
    # only the first name so every expensive forward pass is causally unique.
    deduplicated: list[BoxPortfolio] = []
    seen: set[tuple[PolicyBox, ...]] = set()
    for portfolio in output:
        if portfolio.boxes in seen:
            continue
        seen.add(portfolio.boxes)
        deduplicated.append(portfolio)
    return tuple(deduplicated)


__all__ = [
    "BoxPortfolio",
    "PolicyBox",
    "counterfactual_portfolios",
    "transform_policy_box",
]
