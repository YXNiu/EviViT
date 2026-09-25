#!/usr/bin/env python3
"""Evaluate base Qwen3-VL original-only QA on an Eval manifest."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evivit_core.gpu_safety import check_gpu  # noqa: E402
from evivit_core.latency import FirstTokenTimer, summarize_latency  # noqa: E402
from evivit_core.qwen_family import load_qwen_model_class  # noqa: E402
from evivit_core.evivit_recovery_fastdev import recovery_row_metadata  # noqa: E402


SYSTEM_PROMPT = (
    "You are a careful fine-grained visual question answering assistant. "
    "Answer using only the image evidence. Return compact JSON only."
)


NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def configure_deterministic_inference(attention_backend: str = "math") -> None:
    """Lock one SDPA backend under PyTorch's deterministic-algorithm guard."""

    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if attention_backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    elif attention_backend == "mem_efficient":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    elif attention_backend == "flash":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
    else:
        raise ValueError(f"unsupported deterministic SDPA backend: {attention_backend}")


def deterministic_backend_name(attention_backend: str) -> str:
    return f"sdpa_{attention_backend}_deterministic"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
    return rows


def read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def clean_question(text: str) -> str:
    return " ".join(str(text).replace("<image>", " ").split())


def normalized_answer(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def relaxed_answer(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value).lower())
    cleaned = []
    for token in tokens:
        if token in {"a", "an", "the"}:
            continue
        cleaned.append(NUMBER_WORDS.get(token, token))
    return " ".join(cleaned)


def relaxed_answer_match(predicted: str, target: str) -> bool:
    raw_pred = str(predicted).strip().lower()
    raw_gold = str(target).strip().lower()
    if re.fullmatch(r"[a-d]", raw_gold):
        return raw_pred == raw_gold
    pred = relaxed_answer(predicted)
    gold = relaxed_answer(target)
    if not pred or not gold:
        return False
    return pred == gold or pred in gold or gold in pred


def strip_code_fence(text: str) -> str:
    value = str(text).strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def extract_first_json_object(text: str) -> str:
    value = strip_code_fence(text)
    if value.startswith("{") and value.endswith("}"):
        return value
    start = value.find("{")
    if start < 0:
        return value
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(value[start:], start):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]
    return value


def parse_answer(raw: str) -> tuple[str, str, str | None]:
    # Reasoning checkpoints such as Vero deliberately emit a long <think>
    # trace followed by the deployable answer.  Parse that final channel
    # before looking for JSON that may occur incidentally in the reasoning.
    lower = raw.lower()
    answer_start = lower.rfind("<answer>")
    if answer_start >= 0:
        content_start = answer_start + len("<answer>")
        answer_end = lower.find("</answer>", content_start)
        content = raw[content_start : answer_end if answer_end >= 0 else None].strip()
        boxed_start = content.rfind(r"\boxed{")
        if boxed_start >= 0:
            value_start = boxed_start + len(r"\boxed{")
            depth = 1
            index = value_start
            while index < len(content) and depth:
                if content[index] == "{":
                    depth += 1
                elif content[index] == "}":
                    depth -= 1
                index += 1
            if depth == 0:
                boxed = content[value_start : index - 1].strip()
                return boxed, content, None
        if content:
            return content, content, None
    extracted = extract_first_json_object(raw)
    try:
        payload = json.loads(extracted)
    except json.JSONDecodeError:
        compact = strip_code_fence(raw).strip()
        # Fallback: treat the whole generation as an answer, but record the parse issue.
        return compact, extracted, "non_json_answer"
    if not isinstance(payload, dict):
        if payload is None:
            return "", extracted, "json_null_answer"
        if isinstance(payload, bool):
            return (
                "true" if payload else "false",
                extracted,
                "json_scalar_answer",
            )
        if isinstance(payload, (str, int, float)):
            return str(payload).strip(), extracted, "json_scalar_answer"
        if (
            isinstance(payload, list)
            and len(payload) == 1
            and isinstance(payload[0], (str, int, float, bool))
        ):
            value = payload[0]
            if isinstance(value, bool):
                answer = "true" if value else "false"
            else:
                answer = str(value).strip()
            return answer, extracted, "json_singleton_list_answer"
        return (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            extracted,
            "json_non_object_answer",
        )
    answer = payload.get("answer")
    if answer is None:
        return "", extracted, "missing_answer"
    return str(answer).strip(), extracted, None


def parse_muir_choice(raw: str) -> tuple[str, str, str | None]:
    """Parse the direct multiple-choice letter under MuirBench's protocol."""

    value = strip_code_fence(raw).strip()
    direct = re.fullmatch(r"\s*([A-Ea-e])\s*[.)]?\s*", value)
    if direct:
        return direct.group(1).upper(), value, None
    tagged = re.findall(
        r"(?:answer|choice|option)\s*(?:is|:)?\s*\$?([A-E])\b",
        value,
        flags=re.IGNORECASE,
    )
    if tagged:
        return tagged[-1].upper(), value, "muir_choice_verbose"
    standalone = re.findall(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", value.upper())
    if standalone:
        return standalone[-1], value, "muir_choice_fallback"
    return value, value, "muir_choice_unparsed"


def parse_blink_choice(raw: str) -> tuple[str, str, str | None]:
    """Parse BLINK's official parenthesized multiple-choice answer."""

    prediction, extracted, error = parse_muir_choice(raw)
    if error is None:
        return prediction, extracted, None
    return prediction, extracted, error.replace("muir_choice", "blink_choice")


def qa_messages(
    question: str, image: Image.Image, prompt_variant: str = "canonical"
) -> list[dict[str, Any]]:
    return qa_messages_multi(question, [image], prompt_variant=prompt_variant)


def qa_messages_multi(
    question: str,
    images: Sequence[Image.Image],
    prompt_variant: str = "canonical",
) -> list[dict[str, Any]]:
    """Build one prompt with ordered, independent image observations.

    PerceptionBench places all image placeholders before the question.  Keep
    that ordering instead of composing a contact sheet or inserting extra
    figure-label text that would change the official question semantics.
    """

    if not images:
        raise ValueError("at least one image is required")
    image_content = [{"type": "image", "image": image} for image in images]
    if prompt_variant == "muir_mcq":
        segments = str(question).split("<image>")
        if len(segments) - 1 != len(images):
            raise ValueError(
                "MuirBench prompt/image mismatch: "
                f"{len(segments) - 1} placeholders for {len(images)} images"
            )
        content: list[dict[str, Any]] = []
        if segments[0]:
            content.append({"type": "text", "text": segments[0]})
        for image, segment in zip(images, segments[1:]):
            content.append({"type": "image", "image": image})
            if segment:
                content.append({"type": "text", "text": segment})
        return [{"role": "user", "content": content}]
    if prompt_variant == "blink_mcq":
        return [
            {
                "role": "user",
                "content": [
                    *image_content,
                    {"type": "text", "text": str(question).strip()},
                ],
            }
        ]
    if prompt_variant == "vero_native":
        return [
            {
                "role": "user",
                "content": [*image_content, {"type": "text", "text": clean_question(question)}],
            }
        ]
    if prompt_variant == "vision_opd_official":
        # Match Vision-OPD eval/infer.py: one user turn, image first, and the
        # released query with only an optional <image> placeholder removed.
        # Do not inject EviViT's compact-JSON system prompt or collapse MCQ
        # line breaks.
        official_question = str(question).replace("<image>", "").strip()
        return [
            {
                "role": "user",
                "content": [
                    *image_content,
                    {"type": "text", "text": official_question},
                ],
            }
        ]
    if prompt_variant == "evivit_compact_json":
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    *image_content,
                    {
                        "type": "text",
                        "text": (
                            f"Question: {clean_question(question)}\n"
                            'Return exactly one compact JSON object: {"answer":"..."}\n'
                            "Do not explain."
                        ),
                    },
                ],
            },
        ]
    if prompt_variant == "canonical":
        system_prompt = SYSTEM_PROMPT
        instructions = [
            'Return exactly one compact JSON object: {"answer":"..."}',
            "Do not explain.",
        ]
    elif prompt_variant == "direct":
        system_prompt = (
            "You are a visual question answering assistant. Inspect the image and "
            "give the shortest semantically correct answer in compact JSON."
        )
        instructions = [
            'Answer directly as {"answer":"..."}.',
            "Use no explanation or extra fields.",
        ]
    elif prompt_variant == "evidence_check":
        system_prompt = (
            "You are a careful fine-grained visual question answering assistant. "
            "Check the visible evidence in the full image before answering. Return compact JSON only."
        )
        instructions = [
            "Check small text, objects, and spatial context when relevant.",
            'Return exactly {"answer":"..."} with no explanation.',
        ]
    else:
        raise ValueError(f"unsupported prompt_variant={prompt_variant!r}")
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                *image_content,
                {
                    "type": "text",
                    "text": "\n".join(
                        [f"Question: {clean_question(question)}", *instructions]
                    ),
                },
            ],
        },
    ]


def summarize(rows: list[dict[str, Any]], elapsed: float, peak_reserved_mib: float) -> dict[str, Any]:
    total = len(rows)
    exact = sum(1 for row in rows if row.get("exact_correct"))
    relaxed = sum(1 for row in rows if row.get("relaxed_correct"))
    valid = sum(1 for row in rows if not row.get("parse_error"))
    return {
        "total": total,
        "exact_correct": exact,
        "relaxed_correct": relaxed,
        "exact_accuracy": exact / total if total else 0.0,
        "relaxed_accuracy": relaxed / total if total else 0.0,
        "valid_json_rate": valid / total if total else 0.0,
        "elapsed_seconds": round(elapsed, 3),
        "peak_reserved_mib": peak_reserved_mib,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/derived/evaluation/eval90_manifest.jsonl"),
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-VL-4B-Instruct"))
    parser.add_argument(
        "--adapter",
        type=Path,
        help=(
            "Optional PEFT/LoRA adapter evaluated on the unchanged original-image "
            "path. This is used by the matched B0 answer-only control."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/baselines/eval90/base_qwen_original_predictions.jsonl"),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--memory-budget-mib", type=int, default=62000)
    parser.add_argument("--max-pixels", type=int, default=1024 * 1024)
    parser.add_argument("--min-pixels", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Pass enable_thinking=False to model families that support it. "
            "This is required for the matched Qwen3.5/Vision-OPD answer-only protocol."
        ),
    )
    parser.add_argument(
        "--prompt-variant",
        choices=[
            "canonical",
            "evivit_compact_json",
            "direct",
            "evidence_check",
            "vero_native",
            "muir_mcq",
            "blink_mcq",
            "vision_opd_official",
        ],
        default="canonical",
    )
    parser.add_argument(
        "--record-token-logprob",
        action="store_true",
        help="Store mean/min generated-token log probability for zero-zoom screening.",
    )
    parser.add_argument(
        "--record-timing",
        action="store_true",
        help="Synchronize CUDA and record per-request TTFT/end-to-end latency.",
    )
    parser.add_argument("--timing-warmup-samples", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--deterministic-inference",
        action="store_true",
        help="Use deterministic math-SDPA kernels for reproducible evaluation.",
    )
    parser.add_argument(
        "--deterministic-attention-backend",
        choices=("math", "mem_efficient", "flash"),
        default="math",
        help=(
            "SDPA backend locked under deterministic algorithms. Math is the "
            "strict default; efficient backends require a separate repeatability audit."
        ),
    )
    args = parser.parse_args()

    decision = check_gpu(args.gpu, args.memory_budget_mib, ROOT / "configs/gpu_safety.json")
    print(json.dumps({"gpu_preflight": decision.as_dict()}, ensure_ascii=False), flush=True)
    if not decision.allowed:
        return 2

    rows = read_jsonl(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=True
    )
    model_type = str(model_config.model_type)
    if model_type == "qwen3_5":
        preprocess_protocol = (
            "qwen35_strict_minmax_nonthinking_v1"
            if args.disable_thinking
            else "qwen35_strict_minmax_thinking_v1"
        )
    elif model_type == "qwen3_vl":
        preprocess_protocol = "qwen3vl_strict_minmax_v1"
    else:
        raise ValueError(f"unsupported multimodal model_type={model_type!r}")

    existing = read_existing(args.output) if args.resume else []
    for row in existing:
        if row.get("adapter") != (str(args.adapter) if args.adapter else None):
            raise ValueError(f"adapter mismatch while resuming {args.output}")
        existing_variant = row.get("prompt_variant", "canonical")
        if existing_variant != args.prompt_variant:
            raise ValueError(
                f"cannot resume {args.prompt_variant!r} into output containing "
                f"{existing_variant!r}: {args.output}"
            )
        if row.get("preprocess_protocol") != preprocess_protocol:
            raise ValueError(
                f"cannot resume strict min/max preprocessing into a legacy output: {args.output}"
            )
        if int(row.get("max_pixels", -1)) != args.max_pixels or int(
            row.get("min_pixels", -1)
        ) != args.min_pixels:
            raise ValueError(f"pixel-budget mismatch while resuming {args.output}")
        if bool(row.get("deterministic_inference", False)) != bool(
            args.deterministic_inference
        ):
            raise ValueError(
                f"deterministic-inference mismatch while resuming {args.output}"
            )
    completed = {str(row.get("id")) for row in existing}
    remaining = [row for row in rows if str(row.get("id")) not in completed]
    print(
        json.dumps(
            {
                "resume": {
                    "enabled": bool(args.resume),
                    "existing_rows": len(existing),
                    "remaining_rows": len(remaining),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.deterministic_inference:
        configure_deterministic_inference(args.deterministic_attention_backend)

    import torch
    from transformers import AutoProcessor

    if model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration as ModelClass
    else:
        from transformers import Qwen3VLForConditionalGeneration as ModelClass

    exposed_total_mib = torch.cuda.get_device_properties(0).total_memory / 1024**2
    if args.memory_budget_mib:
        torch.cuda.set_per_process_memory_fraction(
            min(0.95, args.memory_budget_mib / exposed_total_mib), device=0
        )
    torch.cuda.reset_peak_memory_stats(0)
    device = "cuda:0"

    processor = AutoProcessor.from_pretrained(
        str(args.model),
        local_files_only=True,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )
    image_processor = processor.image_processor
    if hasattr(image_processor, "max_pixels"):
        actual_max_pixels = int(image_processor.max_pixels)
        actual_min_pixels = int(image_processor.min_pixels)
        pixel_budget_api = "min_pixels_max_pixels"
    else:
        size = image_processor.size
        actual_max_pixels = int(size.longest_edge)
        actual_min_pixels = int(size.shortest_edge)
        pixel_budget_api = "size_shortest_longest_edge"
    if actual_max_pixels != args.max_pixels or actual_min_pixels != args.min_pixels:
        raise RuntimeError("processor did not retain the requested min/max pixel budget")
    model = load_qwen_model_class(
        ModelClass,
        args.model,
        device=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    checkpoint_key_mapping_count = int(
        getattr(model, "_evivit_checkpoint_key_mapping_count", 0)
    )
    checkpoint_compatibility_base = getattr(
        model, "_evivit_checkpoint_compatibility_base", None
    )
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, str(args.adapter), is_trainable=False, local_files_only=True
        )
    model.eval()
    merge_size = int(model.config.vision_config.spatial_merge_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_rows = list(existing)
    mode = "a" if args.resume and existing else "w"
    started = time.perf_counter()
    with args.output.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(remaining, 1):
            if args.record_timing:
                torch.cuda.synchronize(0)
            request_started = time.perf_counter()
            image_values = row.get("images") or [row["image"]]
            image_paths = [Path(str(value)) for value in image_values]
            image_paths = [
                path if path.is_absolute() else args.project_root / path
                for path in image_paths
            ]
            images = [Image.open(path).convert("RGB") for path in image_paths]
            messages = qa_messages_multi(
                str(row["question"]), images, prompt_variant=args.prompt_variant
            )
            template_kwargs = {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            if model_type == "qwen3_5":
                template_kwargs["enable_thinking"] = not args.disable_thinking
            inputs = processor.apply_chat_template(messages, **template_kwargs).to(device)
            grid_rows = inputs["image_grid_thw"].detach().cpu().tolist()
            visual_tokens = [
                int(t) * (int(h) // merge_size) * (int(w) // merge_size)
                for t, h, w in grid_rows
            ]
            if len(grid_rows) != len(images):
                raise RuntimeError(
                    f"expected {len(images)} image grids, got {len(grid_rows)}: {grid_rows}"
                )
            with torch.inference_mode():
                first_token_timer = FirstTokenTimer(lambda: torch.cuda.synchronize(0))
                if args.record_timing:
                    torch.cuda.synchronize(0)
                generation_started = time.perf_counter()
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=args.record_token_logprob,
                    output_scores=args.record_token_logprob,
                    logits_processor=[first_token_timer] if args.record_timing else None,
                )
                if args.record_timing:
                    torch.cuda.synchronize(0)
                generation_finished = time.perf_counter()
            sequences = generated.sequences if args.record_token_logprob else generated
            trimmed = [
                output[len(source) :] for source, output in zip(inputs.input_ids, sequences)
            ]
            token_logprobs: list[float] = []
            if args.record_token_logprob:
                generated_ids = trimmed[0]
                for token_id, scores in zip(generated_ids, generated.scores):
                    value = torch.log_softmax(scores[0].float(), dim=-1)[token_id]
                    token_logprobs.append(float(value.detach().cpu()))
            raw = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            if args.prompt_variant == "muir_mcq":
                prediction, extracted, parse_error = parse_muir_choice(raw)
            elif args.prompt_variant == "blink_mcq":
                prediction, extracted, parse_error = parse_blink_choice(raw)
            else:
                prediction, extracted, parse_error = parse_answer(raw)
            target = str(row.get("answer", ""))
            out = {
                "id": row["id"],
                "eval_tier": row.get("eval_tier"),
                "image": row.get("image"),
                "images": row.get("images"),
                "image_count": len(images),
                "question": row.get("question"),
                "answer": target,
                "raw_prediction": raw,
                "extracted_prediction": extracted,
                "prediction": prediction,
                "normalized_prediction": normalized_answer(prediction),
                "normalized_answer": normalized_answer(target),
                "relaxed_prediction": relaxed_answer(prediction),
                "relaxed_answer": relaxed_answer(target),
                "exact_correct": normalized_answer(prediction) == normalized_answer(target),
                "relaxed_correct": relaxed_answer_match(prediction, target),
                "parse_error": parse_error,
                "method": "base_qwen_original_only",
                "model": str(args.model),
                "adapter": str(args.adapter) if args.adapter else None,
                "prompt_variant": args.prompt_variant,
                "disable_thinking": args.disable_thinking,
                "model_type": model_type,
                "preprocess_protocol": preprocess_protocol,
                "max_pixels": args.max_pixels,
                "min_pixels": args.min_pixels,
                "pixel_budget_api": pixel_budget_api,
                "image_grid_thw": grid_rows[0] if len(grid_rows) == 1 else grid_rows,
                "visual_tokens_by_image": visual_tokens,
                "visual_tokens": sum(visual_tokens),
                "mean_token_logprob": (
                    sum(token_logprobs) / len(token_logprobs) if token_logprobs else None
                ),
                "min_token_logprob": min(token_logprobs) if token_logprobs else None,
                "crop_count": 0,
                "deterministic_inference": args.deterministic_inference,
                "attention_backend": (
                    deterministic_backend_name(args.deterministic_attention_backend)
                    if args.deterministic_inference
                    else "sdpa_default"
                ),
                "checkpoint_key_mapping_count": checkpoint_key_mapping_count,
                "checkpoint_compatibility_base": checkpoint_compatibility_base,
            }
            out.update(recovery_row_metadata(row))
            if args.record_timing:
                if first_token_timer.first_token_time is None:
                    raise RuntimeError("first-token timer was not invoked")
                out["timing"] = {
                    "request_to_first_token_seconds": (
                        first_token_timer.first_token_time - request_started
                    ),
                    "end_to_end_seconds": generation_finished - request_started,
                    "generation_seconds": generation_finished - generation_started,
                    "decode_seconds": (
                        generation_finished - first_token_timer.first_token_time
                    ),
                    "generated_tokens": int(trimmed[0].numel()),
                }
            out_rows.append(out)
            handle.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "progress": index,
                        "remaining": len(remaining),
                        "id": out["id"],
                        "pred": prediction,
                        "gold": target,
                        "exact": out["exact_correct"],
                        "relaxed": out["relaxed_correct"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize(0)
    summary = summarize(
        out_rows,
        elapsed=time.perf_counter() - started,
        peak_reserved_mib=round(torch.cuda.max_memory_reserved(0) / 1024**2, 1),
    )
    summary.update(
        {
            "manifest": str(args.manifest),
            "project_root": str(args.project_root),
            "model": str(args.model),
            "adapter": str(args.adapter) if args.adapter else None,
            "output": str(args.output),
            "limit": args.limit,
            "max_pixels": args.max_pixels,
            "min_pixels": args.min_pixels,
            "pixel_budget_api": pixel_budget_api,
            "model_type": model_type,
            "checkpoint_key_mapping_count": checkpoint_key_mapping_count,
            "checkpoint_compatibility_base": checkpoint_compatibility_base,
            "preprocess_protocol": preprocess_protocol,
            "disable_thinking": args.disable_thinking,
            "mean_visual_tokens": (
                sum(int(row["visual_tokens"]) for row in out_rows) / len(out_rows)
                if out_rows else 0.0
            ),
            "max_visual_tokens": max(
                (int(row["visual_tokens"]) for row in out_rows), default=0
            ),
            "max_new_tokens": args.max_new_tokens,
            "prompt_variant": args.prompt_variant,
            "record_token_logprob": args.record_token_logprob,
            "record_timing": args.record_timing,
            "deterministic_inference": args.deterministic_inference,
            "attention_backend": (
                deterministic_backend_name(args.deterministic_attention_backend)
                if args.deterministic_inference
                else "sdpa_default"
            ),
            "latency": (
                summarize_latency(
                    out_rows, warmup_samples=args.timing_warmup_samples
                )
                if args.record_timing
                else None
            ),
        }
    )
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
