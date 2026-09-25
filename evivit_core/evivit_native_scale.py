"""Pixel-native view planning for EviViT-NativeScale.

The important contract is expressed in pixels, exactly where Qwen's image
processor expresses it.  The resulting visual-token grid is an observation,
not a second optimization target: a view below ``maximum_pixels`` keeps its
native Qwen grid, while a larger view is aspect-preservingly resized once.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from evivit_core.evitree import qwen_native_visual_token_geometry


@dataclass(frozen=True)
class NativePixelViewPlan:
    """Auditable Qwen smart-resize geometry for one source or crop view."""

    source_width: int
    source_height: int
    source_pixels: int
    minimum_pixels: int
    maximum_pixels: int
    resized_width: int
    resized_height: int
    resized_pixels: int
    grid_width: int
    grid_height: int
    realized_tokens: int
    native_grid_preserved: bool
    maximum_applied: bool
    minimum_applied: bool

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class ContinuousNativeGlobalPlan:
    """Continuous, pixel-native Global ceiling for one source image."""

    native_tokens: int
    base_tokens: int
    target_token_ceiling: int
    exponent: float
    base_pixels: int
    native_maximum_pixels: int
    sample_maximum_pixels: int
    view_plan: NativePixelViewPlan

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = asdict(self)
        payload["view_plan"] = self.view_plan.to_dict()
        return payload


def plan_native_pixel_view(
    width: int,
    height: int,
    *,
    minimum_pixels: int = 4096,
    maximum_pixels: int = 16 * 1024 * 1024,
    patch_size: int = 16,
    merge_size: int = 2,
) -> NativePixelViewPlan:
    """Return Qwen's natural post-merger grid under a pixel-only contract."""

    tokens, resized_height, resized_width = qwen_native_visual_token_geometry(
        width,
        height,
        minimum_pixels=minimum_pixels,
        maximum_pixels=maximum_pixels,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    factor = int(patch_size * merge_size)
    grid_height = resized_height // factor
    grid_width = resized_width // factor
    if grid_height * grid_width != tokens:
        raise AssertionError("Qwen pixel plan lost visual-grid tokens")
    source_pixels = int(width * height)
    resized_pixels = int(resized_width * resized_height)
    # Grid alignment may change dimensions by less than one merged stride and
    # is not considered semantic resizing.  These flags report whether a pixel
    # bound, rather than alignment, forced a change.
    maximum_applied = source_pixels > maximum_pixels
    minimum_applied = source_pixels < minimum_pixels
    native_grid_preserved = not maximum_applied and not minimum_applied
    return NativePixelViewPlan(
        source_width=int(width),
        source_height=int(height),
        source_pixels=source_pixels,
        minimum_pixels=int(minimum_pixels),
        maximum_pixels=int(maximum_pixels),
        resized_width=int(resized_width),
        resized_height=int(resized_height),
        resized_pixels=resized_pixels,
        grid_width=int(grid_width),
        grid_height=int(grid_height),
        realized_tokens=int(tokens),
        native_grid_preserved=bool(native_grid_preserved),
        maximum_applied=bool(maximum_applied),
        minimum_applied=bool(minimum_applied),
    )


def plan_continuous_native_global_view(
    width: int,
    height: int,
    *,
    minimum_pixels: int = 4096,
    base_pixels: int = 4 * 1024 * 1024,
    maximum_pixels: int = 16 * 1024 * 1024,
    exponent: float = 0.5,
    patch_size: int = 16,
    merge_size: int = 2,
) -> ContinuousNativeGlobalPlan:
    """Plan a continuous Native-Global view without forcing a token target.

    Let ``N_native`` be the native Qwen token demand under ``maximum_pixels``
    and ``N0`` the token-equivalent of ``base_pixels``.  The ceiling is
    ``N_native`` for small inputs and ``N0*(N_native/N0)**exponent`` for large
    inputs.  Only the corresponding pixel ceiling is passed to Qwen
    smart-resize; the realized token grid remains an observed consequence.
    """

    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    if minimum_pixels <= 0:
        raise ValueError("minimum_pixels must be positive")
    if not minimum_pixels <= base_pixels <= maximum_pixels:
        raise ValueError(
            "continuous Global pixels must satisfy minimum <= base <= maximum"
        )
    if not 0.0 <= exponent <= 1.0:
        raise ValueError("continuous Global exponent must lie in [0, 1]")
    factor = int(patch_size * merge_size)
    if factor <= 0:
        raise ValueError("patch_size * merge_size must be positive")

    native_plan = plan_native_pixel_view(
        width,
        height,
        minimum_pixels=minimum_pixels,
        maximum_pixels=maximum_pixels,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    base_tokens = max(1, int(base_pixels // (factor * factor)))
    native_tokens = int(native_plan.realized_tokens)
    if native_tokens <= base_tokens:
        target_token_ceiling = native_tokens
    else:
        target_token_ceiling = int(
            math.ceil(
                base_tokens
                * (float(native_tokens) / float(base_tokens)) ** exponent
            )
        )
    target_token_ceiling = min(native_tokens, max(1, target_token_ceiling))
    # When the continuous ceiling reaches the measured native demand, reuse
    # the exact native pixel maximum.  Reconstructing it as tokens*stride^2
    # can be slightly smaller because Qwen aspect/alignment rounding makes the
    # measured grid area lower than the configured pixel maximum.
    if target_token_ceiling >= native_tokens:
        sample_maximum_pixels = int(maximum_pixels)
    else:
        sample_maximum_pixels = min(
            int(maximum_pixels),
            max(int(minimum_pixels), target_token_ceiling * factor * factor),
        )
    view_plan = plan_native_pixel_view(
        width,
        height,
        minimum_pixels=minimum_pixels,
        maximum_pixels=sample_maximum_pixels,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    return ContinuousNativeGlobalPlan(
        native_tokens=native_tokens,
        base_tokens=base_tokens,
        target_token_ceiling=target_token_ceiling,
        exponent=float(exponent),
        base_pixels=int(base_pixels),
        native_maximum_pixels=int(maximum_pixels),
        sample_maximum_pixels=int(sample_maximum_pixels),
        view_plan=view_plan,
    )


__all__ = [
    "ContinuousNativeGlobalPlan",
    "NativePixelViewPlan",
    "plan_continuous_native_global_view",
    "plan_native_pixel_view",
]
