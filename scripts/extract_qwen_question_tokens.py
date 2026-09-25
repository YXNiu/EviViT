#!/usr/bin/env python3
"""Cache per-token Qwen question embeddings without loading the full 4B model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch.nn import functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.extract_qwen_dense_features import build_projection, clean_question, read_jsonl


def load_embedding_rows(model_dir: Path, token_ids: list[int]) -> dict[int, torch.Tensor]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    key = "model.language_model.embed_tokens.weight"
    shard = model_dir / index["weight_map"][key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(key)
    unique_ids = sorted(set(token_ids))
    selected = weight[torch.as_tensor(unique_ids, dtype=torch.long)].float()
    return {token_id: row for token_id, row in zip(unique_ids, selected)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--projection-seed", type=int, default=20260715)
    parser.add_argument("--max-question-length", type=int, default=128)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    tokenized: list[dict[str, Any]] = []
    all_ids: list[int] = []
    for row in rows:
        question = clean_question(str(row["question"]))
        encoded = tokenizer(
            question,
            add_special_tokens=True,
            truncation=True,
            max_length=args.max_question_length,
        )
        ids = [int(value) for value in encoded["input_ids"]]
        all_ids.extend(ids)
        tokenized.append({"id": str(row["id"]), "question": question, "token_ids": ids})

    embedding_rows = load_embedding_rows(args.model, all_ids)
    hidden_size = int(next(iter(embedding_rows.values())).numel())
    projection = build_projection(
        hidden_size, args.projection_dim, args.projection_seed, device="cpu"
    )
    projected = {
        token_id: F.normalize(vector @ projection, dim=-1).to(torch.float16)
        for token_id, vector in embedding_rows.items()
    }
    records = []
    for row in tokenized:
        ids = row["token_ids"]
        records.append(
            {
                **row,
                "token_strings": tokenizer.convert_ids_to_tokens(ids),
                "question_tokens": torch.stack([projected[token_id] for token_id in ids]),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "records": records,
            "meta": {
                "manifest": str(args.manifest),
                "model": str(args.model),
                "projection_dim": args.projection_dim,
                "projection_seed": args.projection_seed,
                "rows": len(records),
                "unique_token_ids": len(embedding_rows),
            },
        },
        args.output,
    )
    print(json.dumps({"output": str(args.output), **torch.load(args.output, weights_only=False)["meta"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
