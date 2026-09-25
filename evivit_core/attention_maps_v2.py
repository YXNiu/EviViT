"""Human-search attention labels from mouse speed, dwell, zoom, and reset traces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from evivit_core.dense_evidence import normalize_policy_box, pixels_to_policy_box


MOVE_TYPES = {"hover_move", "drag_move", "edit_move"}


MAIN_WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    # Primary recipe: the last human zoom is normally already sufficient to answer,
    # while the final box supplies the most precise localization cue.
    "terminal_dominant": {
        "slow_dwell": 0.20,
        "all_zoom": 0.10,
        "last_zoom": 0.40,
        "final_box": 0.30,
    },
    # Preserved as an ablation because it represents the broader search process.
    "process_balanced": {
        "slow_dwell": 0.40,
        "all_zoom": 0.25,
        "last_zoom": 0.25,
        "final_box": 0.10,
    },
    # Layered label fusion: keep terminal evidence dominant while restoring the
    # committed (post-last-reset) human search path and weak global exploration.
    "layered_committed": {
        "terminal": 0.55,
        "committed": 0.30,
        "exploration": 0.15,
    },
}

TERMINAL_WEIGHTS = {"last_zoom": 0.60, "final_box": 0.40}


def normalize_distribution(array: np.ndarray) -> np.ndarray:
    result = np.clip(np.asarray(array, dtype=np.float32), 0.0, None)
    total = float(result.sum())
    return result / total if total > 0 else result


def _point_to_cell(x: float, y: float, width: int, height: int, size: int) -> tuple[int, int]:
    px = min(max(float(x), 0.0), float(width))
    py = min(max(float(y), 0.0), float(height))
    hx = min(size - 1, max(0, int(math.floor(px / width * size))))
    hy = min(size - 1, max(0, int(math.floor(py / height * size))))
    return hx, hy


def _add_gaussian(
    heatmap: np.ndarray,
    x: float,
    y: float,
    *,
    width: int,
    height: int,
    weight: float,
    radius: int,
) -> None:
    if weight <= 0:
        return
    hx, hy = _point_to_cell(x, y, width, height, heatmap.shape[0])
    sigma = max(0.75, radius / 1.5)
    y0, y1 = max(0, hy - radius), min(heatmap.shape[0], hy + radius + 1)
    x0, x1 = max(0, hx - radius), min(heatmap.shape[1], hx + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    heatmap[y0:y1, x0:x1] += weight * np.exp(
        -((xx - hx) ** 2 + (yy - hy) ** 2) / (2.0 * sigma * sigma)
    )


def _add_policy_box(heatmap: np.ndarray, box: Sequence[float], weight: float) -> None:
    if weight <= 0:
        return
    x1, y1, x2, y2 = normalize_policy_box(box)
    height, width = heatmap.shape
    gx1 = min(width - 1, max(0, int(math.floor(x1 / 1000 * width))))
    gy1 = min(height - 1, max(0, int(math.floor(y1 / 1000 * height))))
    gx2 = min(width, max(gx1 + 1, int(math.ceil(x2 / 1000 * width))))
    gy2 = min(height, max(gy1 + 1, int(math.ceil(y2 / 1000 * height))))
    heatmap[gy1:gy2, gx1:gx2] += weight / max(1, (gx2 - gx1) * (gy2 - gy1))


def _combine(channels: Sequence[tuple[np.ndarray, float]]) -> np.ndarray:
    output = np.zeros_like(channels[0][0], dtype=np.float32)
    active_weight = 0.0
    for channel, weight in channels:
        distribution = normalize_distribution(channel)
        if float(distribution.sum()) <= 0 or weight <= 0:
            continue
        output += float(weight) * distribution
        active_weight += float(weight)
    if active_weight > 0:
        output /= active_weight
    return normalize_distribution(output)


def _next_boundary_time(events: list[dict[str, Any]], start: int, fallback: float) -> float:
    boundary_types = {
        "zoom_start", "zoom_reset", "create_start", "select_bbox", "edit_start"
    }
    for event in events[start + 1 :]:
        if str(event.get("type", "")) in boundary_types:
            return float(event.get("t", fallback))
    return fallback


@dataclass(frozen=True)
class AttentionMapsV2:
    slow_dwell: np.ndarray
    all_zoom: np.ndarray
    committed_slow: np.ndarray
    committed_zoom: np.ndarray
    last_zoom: np.ndarray
    final_box: np.ndarray
    exploration: np.ndarray
    committed: np.ndarray
    terminal: np.ndarray
    ambiguity: np.ndarray
    main: np.ndarray
    last_zoom_box: list[int] | None
    stats: dict[str, Any]


def build_attention_maps_v2(
    events: Iterable[dict[str, Any]],
    *,
    width: int,
    height: int,
    final_boxes_policy: Sequence[Sequence[float]] = (),
    size: int = 128,
    velocity_tau_rel_per_second: float = 0.08,
    fast_path_floor: float = 0.10,
    max_interval_ms: float = 500.0,
    main_profile: str = "terminal_dominant",
) -> AttentionMapsV2:
    """Build v2-lite search-attention labels without treating reset branches as negatives."""

    if width <= 0 or height <= 0 or size <= 0:
        raise ValueError("width, height, and size must be positive")
    if velocity_tau_rel_per_second <= 0:
        raise ValueError("velocity_tau_rel_per_second must be positive")
    if main_profile not in MAIN_WEIGHT_PROFILES:
        raise ValueError(
            f"unknown main_profile={main_profile!r}; "
            f"expected one of {sorted(MAIN_WEIGHT_PROFILES)}"
        )
    main_weights = MAIN_WEIGHT_PROFILES[main_profile]
    ordered = sorted((dict(event) for event in events), key=lambda event: float(event.get("t", 0)))
    reset_indices = [
        index for index, event in enumerate(ordered) if str(event.get("type")) == "zoom_reset"
    ]
    last_reset_index = reset_indices[-1] if reset_indices else -1
    arrays = {
        name: np.zeros((size, size), dtype=np.float32)
        for name in (
            "slow_dwell", "ambiguity_slow", "all_zoom", "ambiguity_zoom",
            "committed_slow", "committed_zoom", "last_zoom", "final_box",
        )
    }
    diagonal = math.hypot(width, height)
    out_of_bounds_points = 0
    out_of_bounds_boxes = 0
    movement_points = 0
    zoom_episodes = 0
    velocity_values: list[float] = []
    zoom_start_time: float | None = None
    last_zoom_box: list[int] | None = None

    for index, event in enumerate(ordered):
        event_type = str(event.get("type", ""))
        current_t = float(event.get("t", 0.0))
        if event_type in MOVE_TYPES and "x" in event and "y" in event:
            x, y = float(event["x"]), float(event["y"])
            if not (0 <= x <= width and 0 <= y <= height):
                out_of_bounds_points += 1
            next_t = (
                float(ordered[index + 1].get("t", current_t))
                if index + 1 < len(ordered)
                else current_t + 80.0
            )
            interval_ms = max(0.0, min(max_interval_ms, next_t - current_t))
            velocity = max(0.0, float(event.get("v", 0.0)))
            relative_velocity = velocity / max(diagonal, 1.0)
            velocity_values.append(relative_velocity)
            slow_factor = fast_path_floor + (1.0 - fast_path_floor) * math.exp(
                -relative_velocity / velocity_tau_rel_per_second
            )
            type_factor = 1.0 if event_type == "hover_move" else 0.60
            weight = max(0.005, interval_ms / 1000.0) * slow_factor * type_factor
            _add_gaussian(
                arrays["slow_dwell"], x, y, width=width, height=height,
                weight=weight, radius=2,
            )
            if last_reset_index < 0 or index > last_reset_index:
                _add_gaussian(
                    arrays["committed_slow"], x, y, width=width, height=height,
                    weight=weight, radius=2,
                )
            if 0 <= index < last_reset_index:
                _add_gaussian(
                    arrays["ambiguity_slow"], x, y, width=width, height=height,
                    weight=weight, radius=2,
                )
            movement_points += 1

        if event_type == "zoom_start":
            zoom_start_time = current_t
        elif event_type == "zoom_end":
            bbox = event.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                if not (
                    0 <= float(bbox[0]) <= width and 0 <= float(bbox[2]) <= width
                    and 0 <= float(bbox[1]) <= height and 0 <= float(bbox[3]) <= height
                ):
                    out_of_bounds_boxes += 1
                policy_box = pixels_to_policy_box(bbox, width, height)
                x1, y1, x2, y2 = policy_box
                area_ratio = max(1e-4, (x2 - x1) * (y2 - y1) / 1_000_000.0)
                draw_duration_ms = max(
                    0.0, min(5000.0, current_t - (zoom_start_time or current_t))
                )
                boundary_t = _next_boundary_time(ordered, index, current_t + 1000.0)
                post_zoom_ms = max(0.0, min(8000.0, boundary_t - current_t))
                draw_factor = 1.0 + math.log1p(draw_duration_ms / 500.0)
                inspect_factor = 1.0 + 0.5 * math.log1p(post_zoom_ms / 1000.0)
                depth_factor = min(2.5, 1.0 + 0.25 * math.log(1.0 / area_ratio))
                weight = draw_factor * inspect_factor * depth_factor
                _add_policy_box(arrays["all_zoom"], policy_box, weight)
                if last_reset_index < 0 or index > last_reset_index:
                    _add_policy_box(arrays["committed_zoom"], policy_box, weight)
                if 0 <= index < last_reset_index:
                    _add_policy_box(arrays["ambiguity_zoom"], policy_box, weight)
                last_zoom_box = policy_box
                zoom_episodes += 1
            zoom_start_time = None

    if last_zoom_box is not None:
        _add_policy_box(arrays["last_zoom"], last_zoom_box, 1.0)
    for box in final_boxes_policy:
        _add_policy_box(arrays["final_box"], box, 1.0)

    terminal = _combine(
        [
            (arrays["last_zoom"], TERMINAL_WEIGHTS["last_zoom"]),
            (arrays["final_box"], TERMINAL_WEIGHTS["final_box"]),
        ]
    )
    ambiguity = _combine(
        [(arrays["ambiguity_slow"], 0.55), (arrays["ambiguity_zoom"], 0.45)]
    )
    exploration = _combine(
        [(arrays["slow_dwell"], 2.0 / 3.0), (arrays["all_zoom"], 1.0 / 3.0)]
    )
    committed = _combine(
        [(arrays["committed_slow"], 0.60), (arrays["committed_zoom"], 0.40)]
    )
    if main_profile == "layered_committed":
        main = _combine(
            [
                (terminal, main_weights["terminal"]),
                (committed, main_weights["committed"]),
                (exploration, main_weights["exploration"]),
            ]
        )
    else:
        main = _combine(
            [
                (arrays["slow_dwell"], main_weights["slow_dwell"]),
                (arrays["all_zoom"], main_weights["all_zoom"]),
                (arrays["last_zoom"], main_weights["last_zoom"]),
                (arrays["final_box"], main_weights["final_box"]),
            ]
        )
    output_arrays = {
        "slow_dwell": normalize_distribution(arrays["slow_dwell"]),
        "all_zoom": normalize_distribution(arrays["all_zoom"]),
        "committed_slow": normalize_distribution(arrays["committed_slow"]),
        "committed_zoom": normalize_distribution(arrays["committed_zoom"]),
        "last_zoom": normalize_distribution(arrays["last_zoom"]),
        "final_box": normalize_distribution(arrays["final_box"]),
    }
    velocity_array = np.asarray(velocity_values, dtype=np.float32)
    stats = {
        "events": len(ordered),
        "movement_points": movement_points,
        "zoom_episodes": zoom_episodes,
        "resets": len(reset_indices),
        "last_reset_index": last_reset_index,
        "out_of_bounds_points": out_of_bounds_points,
        "out_of_bounds_boxes": out_of_bounds_boxes,
        "has_last_zoom": last_zoom_box is not None,
        "relative_velocity_median": (
            float(np.median(velocity_array)) if velocity_array.size else 0.0
        ),
        "relative_velocity_p90": (
            float(np.quantile(velocity_array, 0.9)) if velocity_array.size else 0.0
        ),
        "main_entropy": float(
            -(main[main > 0] * np.log(main[main > 0] + 1e-12)).sum()
        ),
        "ambiguity_mass_present": bool(float(ambiguity.sum()) > 0),
        "main_profile": main_profile,
        "main_weights": dict(main_weights),
        "terminal_weights": dict(TERMINAL_WEIGHTS),
    }
    return AttentionMapsV2(
        slow_dwell=output_arrays["slow_dwell"],
        all_zoom=output_arrays["all_zoom"],
        committed_slow=output_arrays["committed_slow"],
        committed_zoom=output_arrays["committed_zoom"],
        last_zoom=output_arrays["last_zoom"],
        final_box=output_arrays["final_box"],
        exploration=exploration,
        committed=committed,
        terminal=terminal,
        ambiguity=ambiguity,
        main=main,
        last_zoom_box=last_zoom_box,
        stats=stats,
    )
