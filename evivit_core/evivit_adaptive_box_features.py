"""Shared online/offline feature contract for EviViT-v6 AdaptiveBox."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from evivit_core.tracescale import normalize_box


def map_crop(tensor: torch.Tensor, box: Sequence[float], size: int = 4) -> torch.Tensor:
    height, width = tensor.shape[-2:]
    x1, y1, x2, y2 = normalize_box(box)
    gx1 = max(0, min(width - 1, int(math.floor(x1 * width))))
    gy1 = max(0, min(height - 1, int(math.floor(y1 * height))))
    gx2 = max(gx1 + 1, min(width, int(math.ceil(x2 * width))))
    gy2 = max(gy1 + 1, min(height, int(math.ceil(y2 * height))))
    patch = tensor[..., gy1:gy2, gx1:gx2]
    return F.adaptive_avg_pool2d(patch.unsqueeze(0), (size, size)).squeeze(0)


def pooled_hidden(hidden: torch.Tensor, box: Sequence[float]) -> torch.Tensor:
    return map_crop(hidden, box, size=2).flatten()


def role_one_hot(index: int, total: int = 3) -> torch.Tensor:
    result = torch.zeros(total, dtype=torch.float32)
    result[min(index, total - 1)] = 1.0
    return result


def geometry_features(region: Any, *, image_size: Sequence[int]) -> torch.Tensor:
    x1, y1, x2, y2 = normalize_box(region.bbox)
    width, height = x2 - x1, y2 - y1
    image_width, image_height = (float(value) for value in image_size)
    return torch.tensor(
        [
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            width,
            height,
            width * height,
            math.log(max(width / max(height, 1e-8), 1e-8)),
            float(region.score),
            float(region.source_rank) / 3.0,
            math.log1p(image_width * image_height) / 20.0,
            math.log(max(image_width, 1.0) / max(image_height, 1.0)),
        ],
        dtype=torch.float32,
    )


def selector_hidden(
    selector: Any,
    visual: torch.Tensor,
    text: torch.Tensor,
    *,
    attention_backend: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = torch.ones(text.shape[0], dtype=torch.bool, device=text.device)
    if attention_backend is None:
        attention_context = nullcontext()
    else:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        backend = {
            "math": SDPBackend.MATH,
            "flash": SDPBackend.FLASH_ATTENTION,
            "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        }.get(attention_backend)
        if backend is None:
            raise ValueError(f"unsupported attention backend: {attention_backend}")
        attention_context = sdpa_kernel(backend)
    if hasattr(selector, "shared"):
        backbone = selector.shared
        with attention_context:
            hidden, _ = backbone.aligned_hidden(visual, text, mask)
            projected_text = backbone.text_projection(text.float().unsqueeze(0))
            projected_text = backbone.text_context(projected_text)
        decisive = hidden + backbone.spatial(hidden)
        context = hidden + selector.context_spatial(hidden)
    elif all(
        hasattr(selector, name)
        for name in ("aligned_hidden", "text_projection", "text_context", "spatial")
    ):
        # Frozen v4/T2 uses the original single-branch PTEA.  Preserve that
        # representation exactly instead of silently replacing it with the
        # later v24 dual decisive/context selector used by v6.
        with attention_context:
            hidden, _ = selector.aligned_hidden(visual, text, mask)
            projected_text = selector.text_projection(text.float().unsqueeze(0))
            projected_text = selector.text_context(projected_text)
        decisive = hidden + selector.spatial(hidden)
        context = decisive
    else:
        raise ValueError("unsupported PTEA feature contract for AdaptiveBox")
    question = projected_text.mean(dim=1).squeeze(0)
    return decisive.squeeze(0), context.squeeze(0), question


def tree_region_features(allocation: Any, index: int) -> torch.Tensor:
    """Return lightweight topology metadata for one T2 sparse branch."""

    candidate = (
        allocation.candidates[index]
        if index < len(getattr(allocation, "candidates", []))
        else {}
    )
    depth = float(candidate.get("tree_depth", 0.0))
    path = candidate.get("tree_path") or []
    return torch.tensor(
        [
            depth / 5.0,
            float(candidate.get("tree_area_ratio", 0.0)),
            min(len(path), 5) / 5.0,
            float(candidate.get("mass", 0.0)),
            float(candidate.get("score", 0.0)),
            min(len(allocation.region_budgets), 3) / 3.0,
        ],
        dtype=torch.float32,
    )


def adaptive_box_feature_vectors(
    selector: Any,
    visual: torch.Tensor,
    text: torch.Tensor,
    allocation: Any,
    *,
    image_size: Sequence[int],
    attention_backend: str | None = None,
) -> torch.Tensor:
    decisive_hidden, context_hidden, question = selector_hidden(
        selector, visual, text, attention_backend=attention_backend
    )
    decisive_probability = allocation.probability_map.float()
    context_probability = (
        allocation.context_probability_map.float()
        if allocation.context_probability_map is not None
        else decisive_probability
    )
    rows = []
    for index, region in enumerate(allocation.region_budgets):
        box = normalize_box(region.bbox)
        decisive_features = pooled_hidden(decisive_hidden, box).float().cpu()
        decisive_map = map_crop(
            decisive_probability.unsqueeze(0), box, size=4
        ).flatten().cpu()
        if hasattr(selector, "shared"):
            parts = [
                decisive_features,
                pooled_hidden(context_hidden, box).float().cpu(),
                question.float().cpu(),
                decisive_map,
                map_crop(
                    context_probability.unsqueeze(0), box, size=4
                ).flatten().cpu(),
                geometry_features(region, image_size=image_size),
                role_one_hot(index),
            ]
        else:
            parts = [
                decisive_features,
                question.float().cpu(),
                decisive_map,
                geometry_features(region, image_size=image_size),
                role_one_hot(index),
                tree_region_features(allocation, index),
            ]
        rows.append(torch.cat(parts))
    if not rows:
        raise ValueError("AdaptiveBox received no legal PTEA regions")
    return torch.stack(rows)


__all__ = [
    "adaptive_box_feature_vectors",
    "geometry_features",
    "map_crop",
    "pooled_hidden",
    "role_one_hot",
    "selector_hidden",
    "tree_region_features",
]
