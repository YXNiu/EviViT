#!/usr/bin/env python3
"""Local text-only semantic QA judge using a Qwen3-VL checkpoint.

The judge never receives images.  It sees only question, gold answer, and model
prediction, and emits a binary semantic-equivalence decision.  Multiple jobs
share one model load and one on-disk cache.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evivit_core.gpu_safety import check_gpu  # noqa: E402


PROMPT_VERSION = "qwen3vl8b_text_judge_strict_v1"
SYSTEM_PROMPT = (
    "You are a strict visual-question-answering evaluator. The image is not "
    "available and you must not answer the visual question yourself. Decide "
    "only whether the candidate prediction is semantically equivalent to the "
    "reference answer for the given question. Ignore harmless case, punctuation, "
    "articles, singular/plural, common abbreviations, and number-word formatting. "
    "A prediction with extra contradictory information is incorrect. Analyze "
    "silently. Output exactly one word: CORRECT or INCORRECT."
)


def judge_lock_path(gpu: int) -> Path:
    lock_root = Path(os.environ.get("EVIVIT_JUDGE_LOCK_DIR", tempfile.gettempdir()))
    return lock_root / f"evivit_qwen3vl8b_judge_gpu{gpu}.lock"


def judge_output_lock_path(output: Path) -> Path:
    """Return a lock shared by every GPU/process writing one judge output."""
    return output.with_suffix(output.suffix + ".judge.lock")


def validate_resume_rows(
    output: Path,
    rows: list[dict[str, Any]],
    existing: list[dict[str, Any]],
    *,
    deterministic_inference: bool,
    deterministic_attention_backend: str = "math",
) -> None:
    if len(existing) > len(rows):
        raise ValueError(
            f"resume output has more rows than predictions: "
            f"{output} ({len(existing)} > {len(rows)})"
        )
    if [str(row["id"]) for row in existing] != [
        str(row["id"]) for row in rows[: len(existing)]
    ]:
        raise ValueError(f"resume prefix mismatch: {output}")
    if any(
        bool(row.get("judge_deterministic_inference", False))
        != deterministic_inference
        for row in existing
    ):
        raise ValueError(f"resume determinism protocol mismatch: {output}")
    if any(row.get("judge_deterministic_attention_backend", "math") != deterministic_attention_backend for row in existing):
        raise ValueError(f"resume attention-backend mismatch: {output}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def cache_key(
    row: dict[str, Any], model: str, *, deterministic_inference: bool = False,
    deterministic_attention_backend: str = "math",
) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "question": str(row.get("question", "")),
        "answer": str(row.get("answer", "")),
        "prediction": str(row.get("prediction", row.get("raw_prediction", ""))),
        "valid_rollout": row.get("valid_rollout"),
        "terminal_answer": row.get("terminal_answer"),
    }
    if deterministic_inference:
        payload["inference_protocol"] = f"sdpa_{deterministic_attention_backend}_deterministic_v1"
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def forced_incorrect_reason(row: dict[str, Any]) -> str | None:
    """Return a deterministic failure reason that must bypass semantic judging."""

    prediction = str(row.get("prediction", row.get("raw_prediction", ""))).strip()
    if row.get("valid_rollout") is False:
        return "invalid_rollout"
    if row.get("terminal_answer") is False:
        return "no_terminal_answer"
    if not prediction:
        return "empty_prediction"
    return None


def user_prompt(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Question: {row.get('question', '')}",
            f"Reference answer: {row.get('answer', '')}",
            f"Candidate prediction: {row.get('prediction', row.get('raw_prediction', ''))}",
            "Decision:",
        ]
    )


def parse_decision(raw: str) -> bool | None:
    value = str(raw).strip()
    if value.startswith("```"):
        value = value.strip("`").strip()
    upper = value.upper()
    match = re.match(r"^\s*(INCORRECT|CORRECT)\b", upper)
    if match:
        return match.group(1) == "CORRECT"
    json_match = re.search(r'"correct"\s*:\s*(true|false)', value, flags=re.I)
    if json_match:
        return json_match.group(1).lower() == "true"
    return None


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {
        str(row["cache_key"]): row
        for row in read_jsonl(path)
        if row.get("cache_key") and not row.get("judge_error")
    }


def render_chat(tokenizer: Any, row: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(row)},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def compact_retry_row(row: dict[str, Any], max_candidate_chars: int = 6000) -> dict[str, Any]:
    """Bound only a parser-error retry while retaining both answer boundaries.

    Some failed host rollouts repeat chart values until the judge prompt reaches
    its token ceiling.  The primary strict judgment always sees the original
    prediction.  If and only if that judgment emits no parseable decision, this
    deterministic retry keeps the beginning and end of the same candidate so
    the judge can still identify a terminal answer or contradictory runaway.
    """

    candidate = str(row.get("prediction", row.get("raw_prediction", "")))
    if len(candidate) <= max_candidate_chars:
        return dict(row)
    side = max_candidate_chars // 2
    compact = (
        candidate[:side]
        + "\n[... deterministic middle truncation for judge retry ...]\n"
        + candidate[-side:]
    )
    result = dict(row)
    result["prediction"] = compact
    result["raw_prediction"] = compact
    return result


def summarize(rows: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    total = len(rows)
    tiers: dict[str, dict[str, int]] = {}
    for row in rows:
        tier = str(row.get("eval_tier") or "unknown")
        entry = tiers.setdefault(tier, {"rows": 0, "judge": 0})
        entry["rows"] += 1
        entry["judge"] += int(bool(row.get("judge_correct")))
    return {
        "version": PROMPT_VERSION,
        "rows": total,
        "judge_correct": sum(bool(row.get("judge_correct")) for row in rows),
        "judge_accuracy": (
            sum(bool(row.get("judge_correct")) for row in rows) / total if total else 0.0
        ),
        "judge_errors": sum(bool(row.get("judge_error")) for row in rows),
        "tiers": tiers,
        "elapsed_seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, action="append", required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen3-VL-8B-Instruct"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("outputs/judge_cache/qwen3vl8b_text_judge_v1.jsonl"),
    )
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument(
        "--memory-budget-mib",
        type=int,
        default=50000,
        help=(
            "Optional per-process GPU preflight/fraction cap in MiB; set 0 to "
            "use CUDA normally without an artificial project-side memory cap."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deterministic-inference", action="store_true")
    parser.add_argument("--deterministic-attention-backend", choices=("math", "flash"), default="math")
    args = parser.parse_args()
    if len(args.predictions) != len(args.output):
        raise ValueError("--predictions and --output counts differ")

    jobs: list[tuple[Path, Path, list[dict[str, Any]], list[dict[str, Any]]]] = []
    pending_total = 0
    for predictions, output in zip(args.predictions, args.output):
        rows = read_jsonl(predictions)
        if args.limit > 0:
            rows = rows[: args.limit]
        if not rows or len({str(row["id"]) for row in rows}) != len(rows):
            raise ValueError(f"invalid prediction rows: {predictions}")
        existing = read_jsonl(output) if args.resume and output.is_file() else []
        validate_resume_rows(
            output,
            rows,
            existing,
            deterministic_inference=args.deterministic_inference,
            deterministic_attention_backend=args.deterministic_attention_backend,
        )
        jobs.append((predictions, output, rows, existing))
        pending_total += len(rows) - len(existing)
    if pending_total == 0:
        for _, output, _, existing in jobs:
            summary = summarize(existing, 0.0)
            summary["deterministic_inference"] = args.deterministic_inference
            summary["max_input_tokens"] = args.max_input_tokens
            summary["max_new_tokens"] = args.max_new_tokens
            summary["attention_backend"] = (
                f"sdpa_{args.deterministic_attention_backend}_deterministic"
                if args.deterministic_inference
                else "sdpa_default"
            )
            output.with_suffix(".summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps({"status": "already_complete", "summaries_refreshed": True}))
        return 0

    # Serialize heavyweight judge model loads per physical GPU.  Multiple
    # experiment watchers can become ready at the same instant after a trainer
    # exits; without this lock they can all pass the same preflight snapshot
    # and then jointly OOM while loading duplicate 8B checkpoints.
    lock_path = judge_lock_path(args.gpu)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    judge_lock = lock_path.open("a+", encoding="utf-8")
    print(
        json.dumps({"judge_gpu_lock": str(lock_path), "status": "waiting"}),
        flush=True,
    )
    fcntl.flock(judge_lock.fileno(), fcntl.LOCK_EX)
    print(
        json.dumps({"judge_gpu_lock": str(lock_path), "status": "acquired"}),
        flush=True,
    )

    if args.memory_budget_mib < 0:
        raise ValueError("--memory-budget-mib must be non-negative")
    if args.memory_budget_mib:
        decision = check_gpu(
            args.gpu, args.memory_budget_mib, ROOT / "configs/gpu_safety.json"
        )
        print(json.dumps({"gpu_preflight": decision.as_dict()}), flush=True)
        if not decision.allowed:
            return 2
    else:
        print(
            json.dumps(
                {
                    "gpu_preflight": {
                        "allowed": True,
                        "mode": "cuda_unbounded_no_project_cap",
                        "gpu": args.gpu,
                    }
                }
            ),
            flush=True,
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch

    if args.deterministic_inference:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(args.deterministic_attention_backend == "flash")
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(args.deterministic_attention_backend == "math")

    device = "cuda:0"
    total_mib = torch.cuda.get_device_properties(0).total_memory / 1024**2
    if args.memory_budget_mib:
        torch.cuda.set_per_process_memory_fraction(
            min(0.95, args.memory_budget_mib / total_mib), device=0
        )

    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": device},
        local_files_only=True,
    ).eval()
    model_name = args.model.name
    cache = load_cache(args.cache)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    for predictions, output, rows, _stale_existing in jobs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output_lock_path = judge_output_lock_path(output)
        output_lock = output_lock_path.open("a+", encoding="utf-8")
        print(
            json.dumps(
                {"judge_output_lock": str(output_lock_path), "status": "waiting"}
            ),
            flush=True,
        )
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX)
        print(
            json.dumps(
                {"judge_output_lock": str(output_lock_path), "status": "acquired"}
            ),
            flush=True,
        )

        # Refresh only after acquiring the output-specific lock.  GPU locks
        # serialize model loads on one card but do not protect the same output
        # from a runner assigned to another card.  Without this refresh, a
        # waiter can use stale resume length and append duplicate rows.
        existing = read_jsonl(output) if args.resume and output.is_file() else []
        validate_resume_rows(
            output,
            rows,
            existing,
            deterministic_inference=args.deterministic_inference,
            deterministic_attention_backend=args.deterministic_attention_backend,
        )
        completed = len(existing)
        mode = "a" if completed else "w"
        judged = list(existing)
        with output.open(mode, encoding="utf-8") as out_handle, args.cache.open(
            "a", encoding="utf-8"
        ) as cache_handle:
            remaining = rows[completed:]
            for start in range(0, len(remaining), args.batch_size):
                batch = remaining[start : start + args.batch_size]
                for row in batch:
                    reason = forced_incorrect_reason(row)
                    if reason is None:
                        continue
                    key = cache_key(
                        row,
                        model_name,
                        deterministic_inference=args.deterministic_inference,
                        deterministic_attention_backend=args.deterministic_attention_backend,
                    )
                    if key in cache:
                        continue
                    item = {
                        "cache_key": key,
                        "prompt_version": PROMPT_VERSION,
                        "judge_model": model_name,
                        "question": row.get("question"),
                        "answer": row.get("answer"),
                        "prediction": row.get("prediction", row.get("raw_prediction")),
                        "correct": False,
                        "judge_error": False,
                        "raw_judge": "INCORRECT",
                        "forced_incorrect_reason": reason,
                        "deterministic_inference": args.deterministic_inference,
                    }
                    cache[key] = item
                    cache_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    cache_handle.flush()
                uncached = [
                    row
                    for row in batch
                    if forced_incorrect_reason(row) is None
                    if cache_key(
                        row,
                        model_name,
                        deterministic_inference=args.deterministic_inference,
                        deterministic_attention_backend=args.deterministic_attention_backend,
                    )
                    not in cache
                ]
                if uncached:
                    prompts = [render_chat(tokenizer, row) for row in uncached]
                    encoded = tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=args.max_input_tokens,
                    ).to(device)
                    with torch.inference_mode():
                        generated = model.generate(
                            **encoded,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                        )
                    continuation = generated[:, encoded.input_ids.shape[1] :]
                    decoded = tokenizer.batch_decode(
                        continuation,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for row, raw in zip(uncached, decoded):
                        key = cache_key(
                            row,
                            model_name,
                            deterministic_inference=args.deterministic_inference,
                            deterministic_attention_backend=args.deterministic_attention_backend,
                        )
                        parsed = parse_decision(raw)
                        retry_raw = ""
                        retry_mode = None
                        if parsed is None:
                            retry_row = compact_retry_row(row)
                            retry_prompt = render_chat(tokenizer, retry_row)
                            retry_encoded = tokenizer(
                                [retry_prompt],
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=args.max_input_tokens,
                            ).to(device)
                            with torch.inference_mode():
                                retry_generated = model.generate(
                                    **retry_encoded,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=False,
                                )
                            retry_continuation = retry_generated[
                                :, retry_encoded.input_ids.shape[1] :
                            ]
                            retry_raw = tokenizer.batch_decode(
                                retry_continuation,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False,
                            )[0]
                            parsed = parse_decision(retry_raw)
                            retry_mode = "compact_candidate_boundary_v1"
                        item = {
                            "cache_key": key,
                            "prompt_version": PROMPT_VERSION,
                            "judge_model": model_name,
                            "question": row.get("question"),
                            "answer": row.get("answer"),
                            "prediction": row.get("prediction", row.get("raw_prediction")),
                            "correct": bool(parsed) if parsed is not None else False,
                            "judge_error": parsed is None,
                            "raw_judge": retry_raw if retry_mode else raw,
                            "judge_primary_raw": raw if retry_mode else None,
                            "judge_retry_mode": retry_mode,
                            "deterministic_inference": args.deterministic_inference,
                        }
                        if parsed is not None:
                            cache[key] = item
                            cache_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                            cache_handle.flush()
                for row in batch:
                    key = cache_key(
                        row,
                        model_name,
                        deterministic_inference=args.deterministic_inference,
                        deterministic_attention_backend=args.deterministic_attention_backend,
                    )
                    cached = cache.get(key)
                    judged_row = dict(row)
                    judged_row.update(
                        {
                            "judge_model": model_name,
                            "judge_prompt_version": PROMPT_VERSION,
                            "judge_correct": bool(cached.get("correct")) if cached else False,
                            "judge_error": cached is None,
                            "raw_judge": cached.get("raw_judge", "") if cached else "",
                            "judge_cache_key": key,
                            "judge_deterministic_inference": (
                                args.deterministic_inference
                            ),
                            "judge_deterministic_attention_backend": args.deterministic_attention_backend,
                            "judge_forced_incorrect_reason": (
                                cached.get("forced_incorrect_reason") if cached else None
                            ),
                        }
                    )
                    judged.append(judged_row)
                    out_handle.write(json.dumps(judged_row, ensure_ascii=False) + "\n")
                    out_handle.flush()
                print(
                    json.dumps(
                        {
                            "job": str(predictions),
                            "completed": len(judged),
                            "rows": len(rows),
                            "errors": sum(bool(row.get("judge_error")) for row in judged),
                        }
                    ),
                    flush=True,
                )
        summary = summarize(judged, time.perf_counter() - started)
        summary["deterministic_inference"] = args.deterministic_inference
        summary["max_input_tokens"] = args.max_input_tokens
        summary["max_new_tokens"] = args.max_new_tokens
        summary["attention_backend"] = (
            f"sdpa_{args.deterministic_attention_backend}_deterministic"
            if args.deterministic_inference
            else "sdpa_default"
        )
        output.with_suffix(".summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_UN)
        output_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
