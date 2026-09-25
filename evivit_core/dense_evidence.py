"""Dense human-evidence maps and cheap box decoding utilities.

The module deliberately separates a trace prior from answer value.  It builds
interpretable spatial targets from the human interaction stream, while leaving
counterfactual QA gain/harm labels to a later stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


def normalize_policy_box(box: Sequence[float]) -> list[int]:
    x1, y1, x2, y2 = [max(0, min(1000, int(round(float(value))))) for value in box]
    if x2 <= x1:
        x2 = min(1000, x1 + 1)
    if y2 <= y1:
        y2 = min(1000, y1 + 1)
    return [x1, y1, x2, y2]


def expand_policy_box(box: Sequence[float], factor: float) -> list[int]:
    """Expand a policy-space box around its center while keeping its size.

    Boundary handling shifts the expanded box back into the image instead of
    merely clipping one side.  This matters for context crops near an edge:
    clipping alone would silently make them less enlarged than central crops.
    """

    if factor < 1.0:
        raise ValueError("box expansion factor must be at least 1.0")
    x1, y1, x2, y2 = normalize_policy_box(box)
    width = min(1000.0, (x2 - x1) * factor)
    height = min(1000.0, (y2 - y1) * factor)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    left = center_x - width / 2.0
    top = center_y - height / 2.0
    left = min(max(0.0, left), 1000.0 - width)
    top = min(max(0.0, top), 1000.0 - height)
    return normalize_policy_box([left, top, left + width, top + height])


def projected_target_pixels(
    candidate: Sequence[float],
    target: Sequence[float],
    *,
    image_size: Sequence[int],
    crop_long_side: int = 1024,
) -> dict[str, float]:
    """Estimate target pixels after a crop is resized for VLM input.

    Coverage alone rewards arbitrarily large crops.  This complementary proxy
    measures how many *rendered* pixels the visible target occupies after the
    crop's longest side is resized to ``crop_long_side``.  It therefore exposes
    the opposite failure mode: a crop can contain the answer but make it tiny.
    """

    if len(image_size) != 2 or int(image_size[0]) <= 0 or int(image_size[1]) <= 0:
        raise ValueError("image_size must be positive [width, height]")
    if crop_long_side <= 0:
        raise ValueError("crop_long_side must be positive")
    image_width, image_height = int(image_size[0]), int(image_size[1])
    cx1, cy1, cx2, cy2 = normalize_policy_box(candidate)
    tx1, ty1, tx2, ty2 = normalize_policy_box(target)
    crop_width = max(1e-6, (cx2 - cx1) / 1000.0 * image_width)
    crop_height = max(1e-6, (cy2 - cy1) / 1000.0 * image_height)
    scale = crop_long_side / max(crop_width, crop_height)
    ix1, iy1 = max(cx1, tx1), max(cy1, ty1)
    ix2, iy2 = min(cx2, tx2), min(cy2, ty2)
    visible_width = max(0.0, (ix2 - ix1) / 1000.0 * image_width * scale)
    visible_height = max(0.0, (iy2 - iy1) / 1000.0 * image_height * scale)
    full_width = (tx2 - tx1) / 1000.0 * image_width * scale
    full_height = (ty2 - ty1) / 1000.0 * image_height * scale
    return {
        "coverage": target_coverage(candidate, target),
        "visible_width_px": visible_width,
        "visible_height_px": visible_height,
        "visible_min_side_px": min(visible_width, visible_height),
        "full_width_px": full_width,
        "full_height_px": full_height,
        "full_min_side_px": min(full_width, full_height),
        "crop_scale": scale,
    }


def pixels_to_policy_box(box: Sequence[float], width: int, height: int) -> list[int]:
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    x1, y1, x2, y2 = [float(value) for value in box]
    return normalize_policy_box(
        [x1 / width * 1000, y1 / height * 1000, x2 / width * 1000, y2 / height * 1000]
    )


def _normalize_peak(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=np.float32)
    peak = float(result.max(initial=0.0))
    if peak > 0:
        result = result / peak
    return result


def normalize_distribution(array: np.ndarray) -> np.ndarray:
    result = np.clip(np.asarray(array, dtype=np.float32), 0.0, None)
    total = float(result.sum())
    if total > 0:
        result = result / total
    return result


def _point_to_cell(x: float, y: float, width: int, height: int, size: int) -> tuple[int, int]:
    px = min(max(float(x), 0.0), float(width))
    py = min(max(float(y), 0.0), float(height))
    hx = min(size - 1, max(0, int(math.floor(px / width * size))))
    hy = min(size - 1, max(0, int(math.floor(py / height * size))))
    return hx, hy


def _add_gaussian(
    heatmap: np.ndarray,
    hx: int,
    hy: int,
    weight: float,
    *,
    radius: int,
) -> None:
    if weight <= 0:
        return
    sigma = max(1.0, radius / 1.5)
    y0, y1 = max(0, hy - radius), min(heatmap.shape[0], hy + radius + 1)
    x0, x1 = max(0, hx - radius), min(heatmap.shape[1], hx + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    heatmap[y0:y1, x0:x1] += weight * np.exp(
        -((xx - hx) ** 2 + (yy - hy) ** 2) / (2.0 * sigma * sigma)
    )


def add_policy_box(heatmap: np.ndarray, box: Sequence[float], weight: float = 1.0) -> None:
    if weight <= 0:
        return
    x1, y1, x2, y2 = normalize_policy_box(box)
    height, width = heatmap.shape
    gx1 = min(width - 1, max(0, int(math.floor(x1 / 1000 * width))))
    gy1 = min(height - 1, max(0, int(math.floor(y1 / 1000 * height))))
    gx2 = min(width, max(gx1 + 1, int(math.ceil(x2 / 1000 * width))))
    gy2 = min(height, max(gy1 + 1, int(math.ceil(y2 / 1000 * height))))
    heatmap[gy1:gy2, gx1:gx2] += weight / max(1, (gx2 - gx1) * (gy2 - gy1))


@dataclass(frozen=True)
class TraceMaps:
    all_trace: np.ndarray
    success_trace: np.ndarray
    failed_trace: np.ndarray
    last_zoom: np.ndarray
    final_box: np.ndarray
    evidence: np.ndarray
    last_zoom_box: list[int] | None
    stats: dict[str, Any]


def build_trace_maps(
    events: Iterable[dict[str, Any]],
    *,
    width: int,
    height: int,
    final_boxes_policy: Sequence[Sequence[float]] = (),
    size: int = 64,
    max_dwell_ms: float = 250.0,
) -> TraceMaps:
    """Build separated trace maps, retaining failed branches as negatives.

    ``success_trace`` is the portion after the last ``zoom_reset``.  For traces
    without a reset, the complete trace is treated as the successful branch.
    ``failed_trace`` contains all spatial evidence before the last reset.
    """

    if width <= 0 or height <= 0 or size <= 0:
        raise ValueError("width, height and size must be positive")
    ordered = sorted((dict(event) for event in events), key=lambda item: float(item.get("t", 0.0)))
    reset_indices = [index for index, event in enumerate(ordered) if event.get("type") == "zoom_reset"]
    last_reset = reset_indices[-1] if reset_indices else -1
    arrays = {
        "all_trace": np.zeros((size, size), dtype=np.float32),
        "success_trace": np.zeros((size, size), dtype=np.float32),
        "failed_trace": np.zeros((size, size), dtype=np.float32),
        "last_zoom": np.zeros((size, size), dtype=np.float32),
        "final_box": np.zeros((size, size), dtype=np.float32),
    }
    last_zoom_box: list[int] | None = None
    point_count = 0
    zoom_box_count = 0

    for index, event in enumerate(ordered):
        category = str(event.get("category", ""))
        event_type = str(event.get("type", ""))
        current_t = float(event.get("t", 0.0))
        next_t = (
            float(ordered[index + 1].get("t", current_t))
            if index + 1 < len(ordered)
            else current_t + 80.0
        )
        dwell = max(20.0, min(float(max_dwell_ms), next_t - current_t)) / max_dwell_ms
        destination_names = ["all_trace"]
        if last_reset < 0 or index > last_reset:
            destination_names.append("success_trace")
        elif index < last_reset:
            destination_names.append("failed_trace")

        if "x" in event and "y" in event and category in {"hover", "drag", "zoom", "canvas"}:
            hx, hy = _point_to_cell(float(event["x"]), float(event["y"]), width, height, size)
            radius = 2 if category == "hover" else 1
            base = 1.0 if category == "hover" else 0.35
            for name in destination_names:
                _add_gaussian(arrays[name], hx, hy, base * dwell, radius=radius)
            point_count += 1

        bbox = event.get("bbox")
        if category == "zoom" and isinstance(bbox, list) and len(bbox) == 4:
            policy_box = pixels_to_policy_box(bbox, width, height)
            weight = 0.50 if event_type == "zoom_end" else 0.18
            for name in destination_names:
                add_policy_box(arrays[name], policy_box, weight)
            zoom_box_count += 1
            if event_type == "zoom_end":
                last_zoom_box = policy_box

    if last_zoom_box is not None:
        add_policy_box(arrays["last_zoom"], last_zoom_box, 1.0)
    for box in final_boxes_policy:
        add_policy_box(arrays["final_box"], box, 1.0)

    for name in arrays:
        arrays[name] = _normalize_peak(arrays[name])

    evidence = _normalize_peak(
        0.25 * arrays["all_trace"]
        + 0.35 * arrays["success_trace"]
        + 0.25 * arrays["last_zoom"]
        + 0.15 * arrays["final_box"]
    )
    stats = {
        "events": len(ordered),
        "resets": len(reset_indices),
        "last_reset_index": last_reset,
        "spatial_points": point_count,
        "zoom_boxes": zoom_box_count,
        "has_last_zoom": last_zoom_box is not None,
        "all_trace_sum": float(arrays["all_trace"].sum()),
        "success_trace_sum": float(arrays["success_trace"].sum()),
        "failed_trace_sum": float(arrays["failed_trace"].sum()),
        "evidence_sum": float(evidence.sum()),
    }
    return TraceMaps(
        all_trace=arrays["all_trace"],
        success_trace=arrays["success_trace"],
        failed_trace=arrays["failed_trace"],
        last_zoom=arrays["last_zoom"],
        final_box=arrays["final_box"],
        evidence=evidence,
        last_zoom_box=last_zoom_box,
        stats=stats,
    )


def policy_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = normalize_policy_box(first)
    bx1, by1, bx2, by2 = normalize_policy_box(second)
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / max(1, area_a + area_b - inter)


def target_coverage(candidate: Sequence[float], target: Sequence[float]) -> float:
    cx1, cy1, cx2, cy2 = normalize_policy_box(candidate)
    tx1, ty1, tx2, ty2 = normalize_policy_box(target)
    inter = max(0, min(cx2, tx2) - max(cx1, tx1)) * max(0, min(cy2, ty2) - max(cy1, ty1))
    return inter / max(1, (tx2 - tx1) * (ty2 - ty1))


def map_mass(heatmap: np.ndarray, box: Sequence[float]) -> float:
    distribution = normalize_distribution(heatmap)
    x1, y1, x2, y2 = normalize_policy_box(box)
    height, width = distribution.shape
    gx1 = min(width - 1, max(0, int(math.floor(x1 / 1000 * width))))
    gy1 = min(height - 1, max(0, int(math.floor(y1 / 1000 * height))))
    gx2 = min(width, max(gx1 + 1, int(math.ceil(x2 / 1000 * width))))
    gy2 = min(height, max(gy1 + 1, int(math.ceil(y2 / 1000 * height))))
    return float(distribution[gy1:gy2, gx1:gx2].sum())


def decode_top_boxes(
    heatmap: np.ndarray,
    *,
    scales: Sequence[float] = (0.25, 0.35, 0.5, 0.67),
    topk: int = 5,
    mass_power: float = 0.5,
    area_penalty: float = 0.05,
    nms_iou: float = 0.5,
) -> list[dict[str, float | list[int]]]:
    """Decode compact boxes from a map without encoding any crop image.

    The search runs only on a small probability grid.  It is therefore a cheap
    numerical decoder, not the expensive candidate-image inference path that
    this project is replacing.
    """

    distribution = normalize_distribution(heatmap)
    height, width = distribution.shape
    if float(distribution.sum()) <= 0:
        return []
    integral = np.pad(distribution.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    proposals: list[dict[str, float | list[int]]] = []
    for scale in scales:
        if not 0 < scale <= 1:
            raise ValueError(f"invalid scale {scale}")
        box_h = max(1, min(height, int(round(height * scale))))
        box_w = max(1, min(width, int(round(width * scale))))
        area_ratio = box_h * box_w / (height * width)
        masses = (
            integral[box_h:, box_w:]
            - integral[:-box_h, box_w:]
            - integral[box_h:, :-box_w]
            + integral[:-box_h, :-box_w]
        )
        scores = masses / max(1e-6, area_ratio**mass_power) - area_penalty * area_ratio
        take = min(scores.size, max(50, topk * 20))
        flat_scores = scores.reshape(-1)
        selected = np.argpartition(flat_scores, -take)[-take:]
        for flat_index in selected:
            y1, x1 = np.unravel_index(int(flat_index), scores.shape)
            y2, x2 = y1 + box_h, x1 + box_w
            proposals.append(
                {
                    "bbox": normalize_policy_box(
                        [x1 / width * 1000, y1 / height * 1000, x2 / width * 1000, y2 / height * 1000]
                    ),
                    "score": float(scores[y1, x1]),
                    "mass": float(masses[y1, x1]),
                    "area_ratio": area_ratio,
                    "scale": float(scale),
                }
            )
    proposals.sort(key=lambda item: (float(item["score"]), float(item["mass"])), reverse=True)
    kept: list[dict[str, float | list[int]]] = []
    for proposal in proposals:
        box = proposal["bbox"]
        assert isinstance(box, list)
        if all(policy_iou(box, existing["bbox"]) < nms_iou for existing in kept):
            kept.append(proposal)
        if len(kept) >= topk:
            break
    return kept
