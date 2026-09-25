#!/usr/bin/env python3
"""Build cached PTEA-anchored training vectors for EviViT-v6 AdaptiveBox."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.predict_patch_text_topk_boxes import load_features, load_selector  # noqa: E402
from scripts.train_patch_text_evidence_map import load_question_tokens  # noqa: E402
from evivit_core.evisplit_context_need import load_context_need_head  # noqa: E402
from evivit_core.evivit_adaptive_box_features import (  # noqa: E402
    adaptive_box_feature_vectors,
)
from evivit_core.evivit_adaptive_box import residual_target  # noqa: E402
from evivit_core.evivit_v3_online import allocate_mid_ptea_evidence  # noqa: E402
from evivit_core.tracescale import (  # noqa: E402
    desired_human_view,
    match_target_to_anchors,
    normalize_box,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--question-tokens", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--context-need-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fine-token-budget", type=int, default=3072)
    parser.add_argument("--minimum-region-tokens", type=int, default=64)
    parser.add_argument("--easy-weight", type=float, default=0.25)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    if len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, got {len(rows)}")
    features = load_features(args.features_dir)
    # The canonical train1144 dense cache preserves the original annotation
    # index in IDs such as ``visual_probe_train_560_idx476_v1`` whereas the
    # trace manifest and question-token cache use ``visual_probe_train_560``.
    # Expose a collision-checked alias so the same frozen features can be
    # reused instead of being re-extracted from Qwen-ViT.
    aliased_features = dict(features)
    for feature_id, feature_row in features.items():
        alias = re.sub(r"_idx\d+_v1$", "", feature_id)
        if alias == feature_id:
            continue
        if alias in aliased_features and aliased_features[alias] is not feature_row:
            raise RuntimeError(f"dense-feature alias collision for {alias}")
        aliased_features[alias] = feature_row
    features = aliased_features
    questions = load_question_tokens(args.question_tokens)
    missing = [str(row["id"]) for row in rows if str(row["id"]) not in features or str(row["id"]) not in questions]
    if missing:
        raise RuntimeError(f"missing {len(missing)} cached inputs; first={missing[:5]}")
    input_dim = int(next(iter(features.values()))["visual"].shape[-1])
    selector = load_selector(args.selector_checkpoint, input_dim, args.device)
    context_need = load_context_need_head(
        args.context_need_checkpoint, device=args.device
    )

    examples: list[dict[str, Any]] = []
    feature_dim = None
    for index, row in enumerate(rows, 1):
        row_id = str(row["id"])
        visual = torch.as_tensor(features[row_id]["visual"], device=args.device)
        text = torch.as_tensor(
            questions[row_id]["question_tokens"], device=args.device
        )
        with torch.inference_mode():
            allocation = allocate_mid_ptea_evidence(
                selector,
                visual,
                text,
                fine_token_budget=args.fine_token_budget,
                max_regions=3,
                minimum_region_tokens=args.minimum_region_tokens,
                evidence_decoder="trace_split",
                region_expansion_factor=1.0,
                topology_union_factor=1.0,
                trace_split_budget_mode="learned",
                trace_split_min_context_fraction=0.10,
                trace_split_max_context_fraction=0.25,
                trace_split_context_need_head=context_need,
            )
        anchors = [normalize_box(region.bbox) for region in allocation.region_budgets]
        if not 1 <= len(anchors) <= 3:
            raise RuntimeError(
                f"{row_id}: expected one to three legal v5 anchors, got {len(anchors)}"
            )
        last_zoom = normalize_box(row["last_zoom_box"]) if row.get("last_zoom_box") else None
        final_boxes = [normalize_box(box) for box in row.get("final_boxes_policy", [])]
        explicit = row.get("adaptive_box_target_policy")
        if explicit:
            target = normalize_box(explicit)
            core = min(final_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1])) if final_boxes else None
            target_source = str(row.get("target_source", "explicit"))
        else:
            target, core, target_source = desired_human_view(last_zoom, final_boxes)
        if target is None:
            raise RuntimeError(f"{row_id}: no adaptive-box target")
        match = match_target_to_anchors(anchors, target, context_margin=0.10)
        anchor_tensor = torch.tensor(anchors, dtype=torch.float32)
        target_tensor = anchor_tensor.clone()
        target_tensor[match.anchor_index] = torch.tensor(target, dtype=torch.float32)
        encoded_targets = residual_target(anchor_tensor, target_tensor)

        vectors = adaptive_box_feature_vectors(
            selector,
            visual,
            text,
            allocation,
            image_size=row.get("image_size") or features[row_id].get("image_size") or [1, 1],
        )
        anchor_rows = []
        for anchor_index, region in enumerate(allocation.region_budgets):
            box = anchors[anchor_index]
            vector = vectors[anchor_index]
            feature_dim = int(vector.numel()) if feature_dim is None else feature_dim
            if int(vector.numel()) != feature_dim:
                raise RuntimeError("AdaptiveBox feature dimensions diverged")
            anchor_rows.append(
                {
                    "bbox": list(box),
                    "role": str(region.role),
                    "score": float(region.score),
                    "source_rank": int(region.source_rank),
                    "token_budget": int(region.token_budget),
                    # Records are accumulated until the final torch.save.  Keep
                    # cached supervision on CPU; retaining one CUDA tensor per
                    # anchor makes memory grow linearly with the dataset.
                    "features": vector.detach().to(
                        device="cpu", dtype=torch.float16
                    ),
                    "target_residual": encoded_targets[anchor_index],
                    "matched": anchor_index == match.anchor_index,
                }
            )
        sample_weight = float(row.get("sample_weight", 1.0))
        if str(row.get("supervision_source", "")).startswith("vlmr1"):
            sample_weight = args.easy_weight
        examples.append(
            {
                "id": row_id,
                "image": str(row.get("image", "")),
                "question": str(row.get("question", "")),
                "source": str(row.get("supervision_source", "visualprobe_trace1144")),
                "sample_weight": sample_weight,
                "target_source": target_source,
                "target_box": torch.tensor(target, dtype=torch.float32),
                "final_box": torch.tensor(core, dtype=torch.float32) if core is not None else None,
                "matched_anchor": int(match.anchor_index),
                "anchors": anchor_rows,
                "map_entropy": float(allocation.map_entropy),
                # Keep the six deployment-time map statistics used by the
                # learned context allocator.  Earlier caches retained only
                # raw entropy, which is not comparable across image sizes
                # because the PTEA grid contains a variable number of cells.
                # These fields let later conditional-routing experiments use
                # normalized entropy and peak prominence without re-running
                # the Qwen visual tower.
                "map_cells": int(allocation.probability_map.numel()),
                "map_peak_probability": float(
                    allocation.map_peak_probability
                ),
                "context_map_entropy": (
                    float(allocation.context_map_entropy)
                    if allocation.context_map_entropy is not None
                    else None
                ),
                "context_need_features": list(
                    allocation.trace_split_context_need_features or []
                ),
                "context_fraction": float(allocation.trace_split_context_fraction or 0.0),
            }
        )
        if index % 50 == 0 or index == len(rows):
            print(json.dumps({"progress": index, "total": len(rows), "id": row_id}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": "evivit_v6_adaptivebox_examples_v1",
            "feature_dim": feature_dim,
            "records": examples,
            "config": vars(args),
        },
        args.output,
    )
    summary = {
        "version": "evivit_v6_adaptivebox_examples_v1",
        "rows": len(examples),
        "anchor_rows": sum(len(row["anchors"]) for row in examples),
        "feature_dim": feature_dim,
        "matched_anchor_counts": {
            str(key): sum(row["matched_anchor"] == key for row in examples)
            for key in range(3)
        },
        "sources": sorted({row["source"] for row in examples}),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
