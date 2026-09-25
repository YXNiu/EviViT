#!/usr/bin/env python3
"""Predict and persist ranked PTEA boxes without running the QA model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_patch_text_evidence_map import load_question_tokens  # noqa: E402
from evivit_core.dense_evidence import decode_top_boxes, expand_policy_box  # noqa: E402
from evivit_core.patch_text_evidence import (  # noqa: E402
    DualPatchTextEvidenceHead,
    MultiScalePatchTextEvidenceHead,
    PatchTextEvidenceHead,
    controlled_question_tokens,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_features(directory: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("dense_features_shard_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for row in payload.get("records", []):
            result[str(row["id"])] = row
    if not result:
        raise RuntimeError(f"no dense feature shards found in {directory}")
    return result


def load_selector(
    path: Path, input_dim: int, device: str
) -> (
    PatchTextEvidenceHead
    | MultiScalePatchTextEvidenceHead
    | DualPatchTextEvidenceHead
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_config = checkpoint.get("model_config") or {}
    visual_input_dim = int(model_config.get("visual_input_dim", input_dim))
    text_input_dim = int(model_config.get("text_input_dim", input_dim))
    if int(input_dim) != visual_input_dim:
        raise ValueError(
            f"requested visual input dim {input_dim} does not match checkpoint "
            f"dimension {visual_input_dim}"
        )
    architecture = str(
        checkpoint.get("selector_architecture", "single_scale_patch_text")
    )
    if architecture == "single_scale_patch_text":
        model = PatchTextEvidenceHead(
            input_dim=None,
            visual_input_dim=visual_input_dim,
            text_input_dim=text_input_dim,
            hidden_dim=int(config["hidden_dim"]),
            text_layers=int(config["text_layers"]),
            text_heads=int(config["text_heads"]),
            dropout=float(config["dropout"]),
        ).to(device)
    elif architecture == "multiscale_patch_text_evidence":
        model = MultiScalePatchTextEvidenceHead(
            input_dim=None,
            visual_input_dim=visual_input_dim,
            text_input_dim=text_input_dim,
            hidden_dim=int(model_config["hidden_dim"]),
            text_layers=int(model_config["text_layers"]),
            text_heads=int(model_config["text_heads"]),
            dropout=float(model_config["dropout"]),
            detail_block=int(model_config["detail_block"]),
            semantic_block=int(model_config["semantic_block"]),
            max_relative_residual=float(
                model_config["max_relative_residual"]
            ),
        ).to(device)
    elif architecture == "dual_residual_context_ptea":
        model = DualPatchTextEvidenceHead(
            input_dim=None,
            visual_input_dim=visual_input_dim,
            text_input_dim=text_input_dim,
            hidden_dim=int(model_config["hidden_dim"]),
            text_layers=int(model_config["text_layers"]),
            text_heads=int(model_config["text_heads"]),
            dropout=float(model_config["dropout"]),
        ).to(device)
    else:
        raise ValueError(f"unsupported selector architecture: {architecture}")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--selector-features", type=Path, required=True)
    parser.add_argument("--question-tokens", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--probability-output",
        type=Path,
        help="Optional torch file containing one float16 evidence map per sample.",
    )
    parser.add_argument("--selector-max-pixels", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--decode-scales", default="0.2,0.25,0.35,0.5,0.67")
    parser.add_argument("--decode-mass-power", type=float, default=0.75)
    parser.add_argument("--decode-area-penalty", type=float, default=0.05)
    parser.add_argument("--box-expansion-factor", type=float, default=1.0)
    parser.add_argument(
        "--question-mode", choices=["correct", "shuffled", "zero"], default="correct"
    )
    parser.add_argument("--question-control-seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    rows = read_jsonl(args.manifest)
    features = load_features(args.selector_features)
    tokens = load_question_tokens(args.question_tokens)
    missing = [
        str(row["id"]) for row in rows
        if str(row["id"]) not in features or str(row["id"]) not in tokens
    ]
    if missing:
        raise RuntimeError(f"missing {len(missing)} selector inputs; first={missing[:5]}")
    tokens, token_source_ids = controlled_question_tokens(
        rows,
        tokens,
        mode=args.question_mode,
        seed=args.question_control_seed,
    )
    input_dim = int(next(iter(features.values()))["visual"].shape[-1])
    selector = load_selector(args.selector_checkpoint, input_dim, args.device)
    scales = tuple(float(value) for value in args.decode_scales.split(",") if value)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    map_tokens = []
    probability_maps: dict[str, torch.Tensor] = {}
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            row_id = str(row["id"])
            visual = torch.as_tensor(features[row_id]["visual"], device=args.device)
            question_tokens = torch.as_tensor(
                tokens[row_id]["question_tokens"], device=args.device
            )
            question_mask = torch.ones(
                question_tokens.shape[0], dtype=torch.bool, device=args.device
            )
            with torch.inference_mode():
                logits = selector(visual, question_tokens, question_mask)[0]
                probability = torch.softmax(logits.flatten(), dim=0).reshape_as(logits)
            if args.probability_output is not None:
                probability_maps[row_id] = probability.detach().to("cpu", torch.float16)
            decoded = decode_top_boxes(
                probability.float().cpu().numpy(),
                scales=scales,
                topk=args.top_k,
                mass_power=args.decode_mass_power,
                area_penalty=args.decode_area_penalty,
            )
            if args.box_expansion_factor != 1.0:
                for candidate in decoded:
                    source_bbox = list(candidate["bbox"])
                    candidate["source_bbox"] = source_bbox
                    candidate["bbox"] = expand_policy_box(
                        source_bbox, args.box_expansion_factor
                    )
                    x1, y1, x2, y2 = candidate["bbox"]
                    candidate["area_ratio"] = (x2 - x1) * (y2 - y1) / 1_000_000.0
                    candidate["box_expansion_factor"] = args.box_expansion_factor
            map_tokens.append(int(probability.numel()))
            output = {
                "id": row["id"],
                "eval_tier": row.get("eval_tier"),
                "image": row["image"],
                "question": row["question"],
                "answer": row.get("answer", ""),
                "selector_version": f"ptea_a_{args.question_mode}",
                "selector_checkpoint": str(args.selector_checkpoint),
                "selector_features": str(args.selector_features),
                "selector_max_pixels": args.selector_max_pixels,
                "question_mode": args.question_mode,
                "question_token_source_id": token_source_ids[row_id],
                "question_control_seed": args.question_control_seed,
                "feature_grid": list(features[row_id]["feature_grid"]),
                "map_tokens": int(probability.numel()),
                "decode_top_k": args.top_k,
                "decode_scales": list(scales),
                "decode_mass_power": args.decode_mass_power,
                "decode_area_penalty": args.decode_area_penalty,
                "box_expansion_factor": args.box_expansion_factor,
                "candidates": decoded,
            }
            handle.write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
            if index % 20 == 0 or index == len(rows):
                print(json.dumps({"progress": index, "total": len(rows), "id": row_id}), flush=True)

    if args.probability_output is not None:
        args.probability_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": "ptea_probability_maps_v1",
                "selector_checkpoint": str(args.selector_checkpoint),
                "selector_max_pixels": args.selector_max_pixels,
                "question_mode": args.question_mode,
                "maps": probability_maps,
            },
            args.probability_output,
        )

    summary = {
        "rows": len(rows),
        "selector_version": f"ptea_a_{args.question_mode}",
        "selector_checkpoint": str(args.selector_checkpoint),
        "selector_features": str(args.selector_features),
        "selector_max_pixels": args.selector_max_pixels,
        "question_mode": args.question_mode,
        "question_control_seed": args.question_control_seed,
        "decode_top_k": args.top_k,
        "decode_scales": list(scales),
        "decode_mass_power": args.decode_mass_power,
        "decode_area_penalty": args.decode_area_penalty,
        "box_expansion_factor": args.box_expansion_factor,
        "mean_map_tokens": sum(map_tokens) / len(map_tokens) if map_tokens else 0.0,
        "output": str(args.output),
        "probability_output": (
            str(args.probability_output) if args.probability_output is not None else None
        ),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
