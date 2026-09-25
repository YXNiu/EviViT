#!/usr/bin/env python3
"""Train EviSplit's complementary context head on all Trace1144 records.

The verified v4 PTEA is frozen as the decisive branch.  Only a duplicated
spatial residual and evidence projection learn the human exploration/context
distribution.  Scientific decisions are made by three Full515 evaluations,
not by the in-training fit diagnostics reported here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_dense_trace_map import (  # noqa: E402
    load_features,
    read_jsonl,
    resize_distribution,
    shape_batches,
)
from scripts.train_evivit_vnext_h2_dual_map import (  # noqa: E402
    context_target_array,
    distribution_loss,
    evaluate,
)
from scripts.train_patch_text_evidence_map import (  # noqa: E402
    load_question_tokens,
    padded_text_batch,
)
from evivit_core.patch_text_evidence import DualPatchTextEvidenceHead  # noqa: E402


def batch_inputs(
    batch: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    question_features: dict[str, dict[str, Any]],
    maps: dict[str, np.ndarray],
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    visual = torch.stack(
        [
            torch.as_tensor(features[str(row["id"])]["visual"])
            for row in batch
        ]
    ).to(device)
    text, text_mask = padded_text_batch(
        batch, question_features, device=device
    )
    height, width = visual.shape[1:3]
    context = torch.stack(
        [
            resize_distribution(
                context_target_array(maps, int(row["label_index"])),
                height,
                width,
                device,
            )
            for row in batch
        ]
    )
    return visual, text, text_mask, context


def atomic_checkpoint(
    path: Path,
    *,
    model: DualPatchTextEvidenceHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "version": "evivit_tracesplit_context_ptea_v1",
            "selector_architecture": "dual_residual_context_ptea",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "model_config": {
                "visual_input_dim": model.visual_input_dim,
                "text_input_dim": model.text_input_dim,
                "hidden_dim": model.hidden_dim,
                "text_layers": config["text_layers"],
                "text_heads": config["text_heads"],
                "dropout": config["dropout"],
            },
            "fit_diagnostics": diagnostics,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--question-tokens", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=1144)
    parser.add_argument("--matched-valid-final-box-subset", action="store_true",
                        help="Explicit paired 1143-row protocol excluding the missing final annotation.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--text-layers", type=int, default=1)
    parser.add_argument("--text-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--cosine-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.expected_rows != 1144 and not (
        args.matched_valid_final_box_subset and args.expected_rows == 1143
    ):
        raise ValueError(
            "formal EviSplit protocol requires expected_rows=1144"
        )
    if args.epochs not in {3, 5}:
        raise ValueError(
            "formal EviSplit protocol supports only preregistered 3- or "
            "5-epoch checkpoint curves"
        )
    if args.output_dir.exists():
        raise RuntimeError(
            f"refusing to overwrite existing output: {args.output_dir}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    rows = read_jsonl(args.manifest)
    if len(rows) != args.expected_rows:
        raise RuntimeError(
            f"expected {args.expected_rows} Trace rows, got {len(rows)}"
        )
    features = load_features(args.features_dir)
    questions = load_question_tokens(args.question_tokens)
    missing = [
        str(row["id"])
        for row in rows
        if str(row["id"]) not in features or str(row["id"]) not in questions
    ]
    if missing:
        raise RuntimeError(f"missing EviSplit inputs: {len(missing)}")
    with np.load(args.maps) as payload:
        required = {"terminal", "exploration", "ambiguity"}
        absent = required.difference(payload.files)
        if absent:
            raise RuntimeError(f"missing trace maps: {sorted(absent)}")
        maps = {name: payload[name] for name in payload.files}

    warm = torch.load(args.warm_start, map_location="cpu", weights_only=False)
    visual_dim = int(next(iter(features.values()))["visual"].shape[-1])
    text_dim = int(
        next(iter(questions.values()))["question_tokens"].shape[-1]
    )
    model = DualPatchTextEvidenceHead(
        input_dim=None,
        visual_input_dim=visual_dim,
        text_input_dim=text_dim,
        hidden_dim=args.hidden_dim,
        text_layers=args.text_layers,
        text_heads=args.text_heads,
        dropout=args.dropout,
    ).to(args.device)
    model.initialize_from_single(warm["model"])
    model.shared.requires_grad_(False)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    config = {
        **vars(args),
        "format_version": "evivit_tracesplit_context_train_v1",
        "training_rows": len(rows),
        "frozen_decisive": (
            "exact warm-start PTEA shared branch; backbone/block identity "
            "is recorded by the warm-start checkpoint"
        ),
        "context_target": (
            "0.70*exploration+0.30*ambiguity when ambiguity exists; "
            "otherwise exploration"
        ),
        "inference_contract": (
            "two frozen-v4 decisive regions plus one masked context region; "
            "same G2F3-R3, Bridge and LLM"
        ),
        "small_subset_policy": (
            "engineering checks only; never an architecture decision"
        ),
        "trainable_parameters": sum(p.numel() for p in trainable),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    log_path = args.output_dir / "train_metrics.jsonl"
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.shared.eval()
        losses: list[float] = []
        context_cosines: list[float] = []
        for batch in shape_batches(
            rows, features, batch_size=args.batch_size, shuffle=True
        ):
            visual, text, text_mask, context_target = batch_inputs(
                batch,
                features,
                questions,
                maps,
                device=args.device,
            )
            optimizer.zero_grad(set_to_none=True)
            _, context_logits = model(visual, text, text_mask)
            loss, _ = distribution_loss(
                context_logits,
                context_target,
                cosine_weight=args.cosine_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            with torch.no_grad():
                probability = torch.softmax(
                    context_logits.flatten(1), dim=1
                )
                context_cosines.extend(
                    F.cosine_similarity(
                        probability, context_target.flatten(1), dim=1
                    )
                    .cpu()
                    .tolist()
                )
        diagnostics = evaluate(
            model,
            rows,
            features,
            questions,
            maps,
            device=args.device,
            batch_size=args.batch_size,
            decode_scales=(0.2, 0.25, 0.35, 0.5, 0.67),
            decode_mass_power=0.75,
            decode_area_penalty=0.05,
        )
        diagnostics["scope"] = "full_training_fit_not_validation"
        record = {
            "epoch": epoch,
            "train_loss": mean(losses),
            "online_train_context_cosine": mean(context_cosines),
            "fit_diagnostics": diagnostics,
            "elapsed_sec": time.perf_counter() - started,
            "peak_gpu_reserved_mib": (
                torch.cuda.max_memory_reserved() / 1024**2
                if torch.device(args.device).type == "cuda"
                else 0.0
            ),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        atomic_checkpoint(
            args.output_dir / f"checkpoint_epoch_{epoch:02d}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            diagnostics=diagnostics,
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)
    (args.output_dir / "TRAINING_COMPLETED").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
