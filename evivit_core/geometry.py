"""Bounding-box conversion and overlap helpers.

The policy coordinate space is integer xyxy in [0, 1000]. Source annotations
remain in pixel space until an image size is known.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

Box = tuple[float, float, float, float]


def coerce_box(payload: Any) -> Box:
    """Read the bbox variants present in the two annotation-tool schemas."""
    if isinstance(payload, dict):
        for key in ("after", "bbox", "before"):
            if key in payload:
                return coerce_box(payload[key])
        raise ValueError(f"bbox object has no supported coordinates: {payload!r}")
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError(f"bbox must be a four-element sequence: {payload!r}")
    if len(payload) != 4:
        raise ValueError(f"bbox must have four coordinates: {payload!r}")
    values = tuple(float(value) for value in payload)
    if not all(value == value for value in values):
        raise ValueError(f"bbox contains NaN: {payload!r}")
    return values  # type: ignore[return-value]


def ordered(box: Box) -> Box:
    x1, y1, x2, y2 = box
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def clip(box: Box, width: float, height: float) -> Box:
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    x1, y1, x2, y2 = ordered(box)
    return (
        min(max(x1, 0.0), width),
        min(max(y1, 0.0), height),
        min(max(x2, 0.0), width),
        min(max(y2, 0.0), height),
    )


def area(box: Box) -> float:
    x1, y1, x2, y2 = ordered(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(left: Box, right: Box) -> float:
    lx1, ly1, lx2, ly2 = ordered(left)
    rx1, ry1, rx2, ry2 = ordered(right)
    return max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )


def iou(left: Box, right: Box) -> float:
    overlap = intersection(left, right)
    union = area(left) + area(right) - overlap
    return overlap / union if union > 0 else 0.0


def coverage(container: Box, evidence: Box) -> float:
    """Fraction of evidence area visible inside a candidate crop."""
    evidence_area = area(evidence)
    return intersection(container, evidence) / evidence_area if evidence_area > 0 else 0.0


def pixels_to_policy(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = clip(box, width, height)
    return (
        round(1000 * x1 / width),
        round(1000 * y1 / height),
        round(1000 * x2 / width),
        round(1000 * y2 / height),
    )


def policy_to_pixels(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = clip(box, 1000, 1000)
    return (
        round(width * x1 / 1000),
        round(height * y1 / 1000),
        round(width * x2 / 1000),
        round(height * y2 / 1000),
    )

