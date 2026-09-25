"""Auditable GRPO primitives shared by native Qwen-ViT and EviViT.

The module separates three concerns that are easy to accidentally confound:

1. build one frozen visual prompt (native Qwen-ViT or EviViT);
2. sample and score concise answer continuations from that same prompt;
3. compute outcome-only group-relative policy optimization.

No image is re-encoded while sampling or scoring the K completions.  The
language model still sees the full visual span, including Qwen3-VL DeepStack
features and multimodal rotary positions.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


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


SEMANTIC_REWARD_JUDGE_VERSION = "qwen3vl8b_text_reward_strict_v1"
SEMANTIC_REWARD_JUDGE_SYSTEM = (
    "You are a strict visual-question-answering reward judge. The reference "
    "answer is authoritative. Decide only whether the candidate is "
    "semantically equivalent to that reference for the given question. "
    "Ignore harmless case, punctuation, articles, singular/plural, common "
    "abbreviations, and number-word formatting. Extra contradictory "
    "information is incorrect. Analyze silently and output exactly one word: "
    "CORRECT or INCORRECT."
)


@dataclass
class FrozenVisualPrompt:
    """One frozen multimodal prompt ready for language-model reuse."""

    input_ids: Any
    inputs_embeds: Any
    attention_mask: Any
    position_ids: Any
    visual_mask: Any
    deepstack_visual_embeds: list[Any]
    next_text_position: int
    visual_tokens: int

    @property
    def prompt_tokens(self) -> int:
        return int(self.input_ids.shape[1])


@dataclass(frozen=True)
class SampledCompletion:
    token_ids: tuple[int, ...]
    text: str


def parse_semantic_reward_decision(raw: str) -> bool | None:
    """Parse the frozen judge's intentionally tiny output vocabulary."""

    value = str(raw).strip()
    if value.startswith("```"):
        value = value.strip("`").strip()
    match = re.match(r"^\s*(INCORRECT|CORRECT)\b", value.upper())
    if match is None:
        return None
    return match.group(1) == "CORRECT"


def load_semantic_reward_judge(model_path: Path, *, device: str) -> tuple[Any, Any]:
    """Load one frozen Qwen3-VL judge beside the trainable policy host."""

    import torch
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": device},
        local_files_only=True,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def semantic_reward_judge_batch(
    tokenizer: Any,
    model: Any,
    *,
    question: str,
    target: str,
    candidates: Sequence[str],
    device: str,
    max_input_tokens: int = 1024,
    max_new_tokens: int = 8,
) -> list[dict[str, Any]]:
    """Judge semantic equivalence in one deterministic text-only batch.

    The image is deliberately omitted: the frozen judge compares a candidate
    against an authoritative reference answer and therefore cannot inject its
    own visual guess into the reward.  This is the same strict protocol used
    by the project's final QA audits.
    """

    import torch

    prompts = []
    for candidate in candidates:
        user = "\n".join(
            [
                f"Question: {question}",
                f"Reference answer: {target}",
                f"Candidate prediction: {candidate}",
                "Decision:",
            ]
        )
        prompts.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SEMANTIC_REWARD_JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    continuation = generated[:, encoded.input_ids.shape[1] :]
    raw_values = tokenizer.batch_decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    results = []
    for candidate, raw in zip(candidates, raw_values):
        decision = parse_semantic_reward_decision(raw)
        results.append(
            {
                "candidate": str(candidate),
                "correct": bool(decision) if decision is not None else False,
                "judge_error": decision is None,
                "raw_judge": str(raw),
                "prompt_version": SEMANTIC_REWARD_JUDGE_VERSION,
            }
        )
    return results


def outcome_reward(*, exact: bool, relaxed: bool, valid: bool) -> float:
    """Return the preregistered outcome-only reward without double counting."""

    if not valid:
        return -0.2
    if exact:
        return 1.0
    if relaxed:
        return 0.5
    return 0.0


def relaxed_reward_answer(value: str) -> str:
    """Normalize articles and small number words for safe relaxed rewards."""

    tokens = re.findall(r"[a-z0-9]+", str(value).lower())
    return " ".join(
        NUMBER_WORDS.get(token, token)
        for token in tokens
        if token not in {"a", "an", "the"}
    )


def strict_relaxed_reward_match(predicted: str, target: str) -> bool:
    """Require normalized equality; substring matches are unsafe RL rewards."""

    prediction = relaxed_reward_answer(predicted)
    answer = relaxed_reward_answer(target)
    return bool(prediction and answer and prediction == answer)


def compare_policy_reference_weights(model: Any) -> dict[str, Any]:
    """Audit paired policy/reference LoRA tensors."""

    parameters = dict(model.named_parameters())
    policy_names = sorted(
        name for name in parameters if "lora_" in name and ".default." in name
    )
    if not policy_names:
        raise ValueError("no default policy LoRA tensors found")
    missing = []
    maximum = 0.0
    elements = 0
    dtype_pairs: dict[str, int] = {}
    for policy_name in policy_names:
        reference_name = policy_name.replace(".default.", ".reference.")
        if reference_name not in parameters:
            missing.append(reference_name)
            continue
        left = parameters[policy_name].detach().float()
        right = parameters[reference_name].detach().float()
        if left.shape != right.shape:
            raise ValueError(f"adapter tensor shape mismatch: {policy_name}")
        maximum = max(maximum, float((left - right).abs().max().cpu()))
        elements += int(left.numel())
        dtype_key = f"{parameters[policy_name].dtype}->{parameters[reference_name].dtype}"
        dtype_pairs[dtype_key] = dtype_pairs.get(dtype_key, 0) + 1
    if missing:
        raise ValueError(f"reference adapter tensors are missing: {missing[:8]}")
    return {
        "paired_tensors": len(policy_names),
        "paired_elements": elements,
        "max_abs_diff": maximum,
        "bitwise_equal": maximum == 0.0,
        "dtype_pairs": dtype_pairs,
    }


def synchronize_policy_reference_weights(model: Any) -> None:
    """Make the fresh reference an exact tensor clone of the initial policy."""

    parameters = dict(model.named_parameters())
    policy_names = sorted(
        name for name in parameters if "lora_" in name and ".default." in name
    )
    for policy_name in policy_names:
        reference_name = policy_name.replace(".default.", ".reference.")
        if reference_name not in parameters:
            raise ValueError(f"missing reference tensor {reference_name}")
        parameters[reference_name].data = parameters[policy_name].detach().clone()


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_fixed_reference_state(model: Any, path: Path) -> str:
    """Persist the immutable initial reference independently of policy resumes."""

    import torch

    parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name and ".reference." in name
    }
    if not parameters:
        raise ValueError("no reference adapter tensors found")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(parameters, temporary)
    os.replace(temporary, path)
    return _sha256_file(path)


def restore_fixed_reference_state(model: Any, path: Path) -> dict[str, Any]:
    """Restore the immutable reference without touching the resumed policy."""

    import torch

    if not path.is_file():
        raise FileNotFoundError(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    parameters = dict(model.named_parameters())
    expected = {
        name for name in parameters if "lora_" in name and ".reference." in name
    }
    if set(saved) != expected:
        raise ValueError("saved fixed-reference tensor names do not match the model")
    maximum = 0.0
    elements = 0
    for name in sorted(expected):
        target = parameters[name]
        value = saved[name]
        if target.shape != value.shape:
            raise ValueError(f"fixed-reference tensor shape mismatch: {name}")
        target.data = value.to(device=target.device).clone()
        maximum = max(
            maximum,
            float((target.detach().cpu().float() - value.float()).abs().max()),
        )
        elements += int(value.numel())
    return {
        "paired_tensors": len(expected),
        "paired_elements": elements,
        "restore_max_abs_diff": maximum,
        "bitwise_equal": maximum == 0.0,
        "sha256": _sha256_file(path),
    }


def group_advantages(
    rewards: Sequence[float], *, epsilon: float = 1e-4
) -> list[float]:
    """Population-standardize rewards within one rollout group.

    A zero-variance group intentionally receives all-zero advantages: the
    group provides no preference signal and must not create a policy update.
    """

    if len(rewards) < 2:
        raise ValueError("GRPO requires at least two rollouts per prompt")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    values = [float(value) for value in rewards]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("rewards must be finite")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance == 0.0:
        return [0.0 for _ in values]
    scale = math.sqrt(variance) + epsilon
    return [(value - mean) / scale for value in values]


def summarize_reward_probe(
    records: Sequence[dict[str, Any]], rollouts_per_prompt: int
) -> dict[str, Any]:
    """Summarize whether K sampled answers contain usable GRPO preferences."""

    if not records or rollouts_per_prompt < 2:
        raise ValueError("reward probe requires non-empty K>=2 groups")
    groups = len(records)
    rollouts = [item for record in records for item in record["rollouts"]]
    count = len(rollouts)
    if count != groups * rollouts_per_prompt:
        raise ValueError("reward probe has an inconsistent rollout count")

    def compact(value: str) -> str:
        import re

        return "".join(re.findall(r"[a-z0-9]+", str(value).lower()))

    return {
        "groups": groups,
        "rollouts": count,
        "valid_answer_rate": sum(bool(item["valid"]) for item in rollouts) / count,
        "exact_rollout_rate": sum(bool(item["exact"]) for item in rollouts) / count,
        "relaxed_rollout_rate": sum(bool(item["relaxed"]) for item in rollouts) / count,
        "mean_reward": sum(float(item["reward"]) for item in rollouts) / count,
        "mean_completion_tokens": sum(int(item["tokens"]) for item in rollouts) / count,
        "mean_unique_answer_fraction": sum(
            len({compact(item["prediction"]) for item in record["rollouts"]})
            / rollouts_per_prompt
            for record in records
        )
        / groups,
        "groups_with_any_correct_rate": sum(
            any(bool(item["relaxed"]) for item in record["rollouts"])
            for record in records
        )
        / groups,
        "nonzero_reward_variance_group_rate": sum(
            len({float(item["reward"]) for item in record["rollouts"]}) > 1
            for record in records
        )
        / groups,
        "all_wrong_group_rate": sum(
            all(float(item["reward"]) == 0.0 for item in record["rollouts"])
            for record in records
        )
        / groups,
        "all_correct_group_rate": sum(
            all(bool(item["relaxed"]) for item in record["rollouts"])
            for record in records
        )
        / groups,
    }


def _repeat_visual_prompt(prompt: FrozenVisualPrompt, repeats: int) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    return {
        "input_ids": prompt.input_ids.repeat(repeats, 1),
        "inputs_embeds": prompt.inputs_embeds.repeat(repeats, 1, 1),
        "attention_mask": prompt.attention_mask.repeat(repeats, 1),
        "position_ids": prompt.position_ids.repeat(1, repeats, 1),
        "visual_mask": prompt.visual_mask.repeat(repeats, 1),
        # Qwen expects visual rows flattened in batch-major order.
        "deepstack_visual_embeds": [
            feature.repeat(repeats, 1) for feature in prompt.deepstack_visual_embeds
        ],
    }


def prepare_native_qwen_visual_prompt(qwen: Any, batch: dict[str, Any]) -> FrozenVisualPrompt:
    """Encode a native Qwen3-VL image once and preserve exact MRoPE metadata."""

    import torch

    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    missing = required - set(batch)
    if missing:
        raise ValueError(f"native visual batch is missing {sorted(missing)}")
    with torch.no_grad():
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        inputs_embeds = qwen.get_input_embeddings()(input_ids)
        image_rows, deepstack = qwen.model.get_image_features(
            batch["pixel_values"], batch["image_grid_thw"]
        )
        image_embeds = torch.cat(image_rows, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = qwen.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_mask = image_mask[..., 0]
        position_ids, rope_deltas = qwen.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=batch["image_grid_thw"],
            attention_mask=attention_mask,
        )
        if input_ids.shape[0] != 1 or rope_deltas.numel() != 1:
            raise ValueError("visual-prompt preparation currently requires batch size one")
        next_text_position = int(input_ids.shape[1] + rope_deltas.item())
    return FrozenVisualPrompt(
        input_ids=input_ids.detach(),
        inputs_embeds=inputs_embeds.detach(),
        attention_mask=attention_mask.detach(),
        position_ids=position_ids.detach(),
        visual_mask=visual_mask.detach(),
        deepstack_visual_embeds=[feature.detach() for feature in deepstack],
        next_text_position=next_text_position,
        visual_tokens=int(visual_mask.sum().item()),
    )


def prepare_evivit_visual_prompt(
    qwen: Any,
    tokenizer: Any,
    prompt_text: str,
    image_embeds: Any,
    deepstack_visual_embeds: list[Any],
    visual_coordinates: Any,
    *,
    device: str,
) -> FrozenVisualPrompt:
    """Build one coordinate-aware EviViT prompt from already frozen features."""

    import torch

    # Import here so pure reward/advantage tests do not require torch.
    from scripts.eval_evivit_unified_tokens import (
        build_position_ids,
        expand_single_image_token,
    )

    raw_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    image_token_id = int(qwen.config.image_token_id)
    expanded_ids, visual_start, visual_end = expand_single_image_token(
        raw_ids, image_token_id, int(image_embeds.shape[0])
    )
    input_ids = torch.tensor([expanded_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        inputs_embeds = qwen.get_input_embeddings()(input_ids)
        visual_mask = input_ids.eq(image_token_id)
        inputs_embeds = inputs_embeds.masked_scatter(
            visual_mask.unsqueeze(-1), image_embeds.to(inputs_embeds.dtype)
        )
        position_ids, next_text_position = build_position_ids(
            input_ids.shape[1],
            visual_start,
            visual_end,
            visual_coordinates,
            device=device,
        )
    return FrozenVisualPrompt(
        input_ids=input_ids.detach(),
        inputs_embeds=inputs_embeds.detach(),
        attention_mask=attention_mask.detach(),
        position_ids=position_ids.detach(),
        visual_mask=visual_mask.detach(),
        deepstack_visual_embeds=[feature.detach() for feature in deepstack_visual_embeds],
        next_text_position=int(next_text_position),
        visual_tokens=int(visual_mask.sum().item()),
    )


def prompt_next_token_logits(qwen: Any, prompt: FrozenVisualPrompt) -> Any:
    """Return next-token logits from a reusable frozen visual prompt."""

    import torch

    cache_position = torch.arange(prompt.prompt_tokens, device=prompt.input_ids.device)
    output = qwen.model.language_model(
        inputs_embeds=prompt.inputs_embeds,
        attention_mask=prompt.attention_mask,
        position_ids=prompt.position_ids,
        cache_position=cache_position,
        visual_pos_masks=prompt.visual_mask,
        deepstack_visual_embeds=[
            row.to(prompt.inputs_embeds.dtype)
            for row in prompt.deepstack_visual_embeds
        ],
        use_cache=False,
    )
    return qwen.lm_head(output.last_hidden_state[:, -1])


def _sample_top_p(logits: Any, *, temperature: float, top_p: float) -> Any:
    import torch

    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probabilities, sorted_indices = torch.sort(
        probabilities, dim=-1, descending=True
    )
    cumulative = sorted_probabilities.cumsum(dim=-1)
    remove = cumulative - sorted_probabilities >= top_p
    sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
    sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    sampled_sorted = torch.multinomial(sorted_probabilities, num_samples=1)
    return sorted_indices.gather(-1, sampled_sorted)


def sample_completions(
    qwen: Any,
    tokenizer: Any,
    prompt: FrozenVisualPrompt,
    *,
    count: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    rollout_batch_size: int = 1,
) -> list[SampledCompletion]:
    """Sample K completions without re-running the frozen visual encoder."""

    import torch

    if count < 2 or max_new_tokens <= 0 or rollout_batch_size <= 0:
        raise ValueError("invalid rollout sampling configuration")
    eos_value = qwen.generation_config.eos_token_id
    eos_ids = {int(eos_value)} if isinstance(eos_value, int) else {
        int(value) for value in (eos_value or [])
    }
    if not eos_ids:
        raise ValueError("generation config has no EOS token")
    eos_fill = min(eos_ids)
    completions: list[SampledCompletion] = []
    with torch.no_grad():
        for offset in range(0, count, rollout_batch_size):
            batch_size = min(rollout_batch_size, count - offset)
            repeated = _repeat_visual_prompt(prompt, batch_size)
            prompt_length = prompt.prompt_tokens
            output = qwen.model.language_model(
                inputs_embeds=repeated["inputs_embeds"],
                attention_mask=repeated["attention_mask"],
                position_ids=repeated["position_ids"],
                cache_position=torch.arange(prompt_length, device=prompt.input_ids.device),
                visual_pos_masks=repeated["visual_mask"],
                deepstack_visual_embeds=[
                    row.to(repeated["inputs_embeds"].dtype)
                    for row in repeated["deepstack_visual_embeds"]
                ],
                use_cache=True,
            )
            past = output.past_key_values
            logits = qwen.lm_head(output.last_hidden_state[:, -1])
            active = torch.ones(batch_size, dtype=torch.bool, device=logits.device)
            token_rows: list[list[int]] = [[] for _ in range(batch_size)]
            for step in range(max_new_tokens):
                sampled = _sample_top_p(
                    logits, temperature=temperature, top_p=top_p
                ).squeeze(-1)
                sampled = torch.where(
                    active,
                    sampled,
                    torch.full_like(sampled, eos_fill),
                )
                for row_index in range(batch_size):
                    if bool(active[row_index].item()):
                        token_rows[row_index].append(int(sampled[row_index].item()))
                is_eos = torch.zeros_like(active)
                for eos_id in eos_ids:
                    is_eos |= sampled.eq(eos_id)
                active &= ~is_eos
                if not bool(active.any().item()) or step + 1 >= max_new_tokens:
                    break
                token_embed = qwen.get_input_embeddings()(sampled.unsqueeze(-1))
                full_length = prompt_length + step + 1
                token_position = torch.full(
                    (3, batch_size, 1),
                    prompt.next_text_position + step,
                    dtype=torch.long,
                    device=prompt.input_ids.device,
                )
                output = qwen.model.language_model(
                    inputs_embeds=token_embed,
                    attention_mask=torch.ones(
                        batch_size,
                        full_length,
                        dtype=torch.long,
                        device=prompt.input_ids.device,
                    ),
                    position_ids=token_position,
                    past_key_values=past,
                    cache_position=torch.tensor(
                        [full_length - 1], device=prompt.input_ids.device
                    ),
                    use_cache=True,
                )
                past = output.past_key_values
                logits = qwen.lm_head(output.last_hidden_state[:, -1])
            for token_ids in token_rows:
                completions.append(
                    SampledCompletion(
                        token_ids=tuple(token_ids),
                        text=tokenizer.decode(
                            token_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ).strip(),
                    )
                )
            del output, past, logits
    if len(completions) != count:
        raise RuntimeError("rollout sampler returned the wrong completion count")
    return completions


def completion_log_probs(
    qwen: Any,
    prompt: FrozenVisualPrompt,
    completion_ids: Any,
    completion_mask: Any,
) -> Any:
    """Score completion tokens while reusing frozen visual embeddings."""

    import torch

    if completion_ids.ndim != 2 or completion_mask.shape != completion_ids.shape:
        raise ValueError("completion ids/mask must be equal rank-2 tensors")
    batch_size, completion_length = completion_ids.shape
    if completion_length <= 0:
        raise ValueError("completion must contain at least one token")
    repeated = _repeat_visual_prompt(prompt, batch_size)
    completion_embeds = qwen.get_input_embeddings()(completion_ids)
    inputs_embeds = torch.cat([repeated["inputs_embeds"], completion_embeds], dim=1)
    # Gradient checkpointing needs a differentiable input even though the
    # embedding table and all visual features stay frozen.
    if torch.is_grad_enabled() and any(
        parameter.requires_grad for parameter in qwen.parameters()
    ):
        inputs_embeds.requires_grad_(True)
    attention_mask = torch.cat(
        [repeated["attention_mask"], completion_mask.to(torch.long)], dim=1
    )
    scalar_positions = torch.arange(
        prompt.next_text_position,
        prompt.next_text_position + completion_length,
        dtype=torch.long,
        device=completion_ids.device,
    ).view(1, 1, -1).expand(3, batch_size, -1)
    position_ids = torch.cat([repeated["position_ids"], scalar_positions], dim=-1)
    visual_mask = torch.cat(
        [
            repeated["visual_mask"],
            torch.zeros_like(completion_mask, dtype=torch.bool),
        ],
        dim=1,
    )
    output = qwen.model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_pos_masks=visual_mask,
        deepstack_visual_embeds=[
            row.to(inputs_embeds.dtype)
            for row in repeated["deepstack_visual_embeds"]
        ],
        use_cache=False,
    )
    start = prompt.prompt_tokens - 1
    selected_hidden = output.last_hidden_state[:, start : start + completion_length]
    logits = qwen.lm_head(selected_hidden)
    return torch.log_softmax(logits.float(), dim=-1).gather(
        -1, completion_ids.unsqueeze(-1)
    ).squeeze(-1)


def grpo_clipped_loss(
    *,
    current_log_probs: Any,
    old_log_probs: Any,
    reference_log_probs: Any,
    completion_mask: Any,
    advantages: Any,
    clip_epsilon: float,
    kl_coefficient: float,
) -> tuple[Any, dict[str, Any]]:
    """Length-normalized clipped GRPO loss with a fixed-reference KL term."""

    import torch

    if current_log_probs.shape != old_log_probs.shape or current_log_probs.shape != reference_log_probs.shape:
        raise ValueError("policy, old-policy, and reference log-prob shapes differ")
    if completion_mask.shape != current_log_probs.shape:
        raise ValueError("completion mask shape differs from log probabilities")
    if advantages.ndim != 1 or advantages.shape[0] != current_log_probs.shape[0]:
        raise ValueError("advantages must contain one value per completion")
    if not 0 < clip_epsilon < 1 or kl_coefficient < 0:
        raise ValueError("invalid clipping or KL configuration")
    mask = completion_mask.to(current_log_probs.dtype)
    lengths = mask.sum(dim=-1).clamp_min(1.0)
    ratio = torch.exp(current_log_probs - old_log_probs.detach())
    expanded_advantage = advantages.unsqueeze(-1)
    unclipped = ratio * expanded_advantage
    clipped = torch.clamp(
        ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
    ) * expanded_advantage
    policy_token_loss = -torch.minimum(unclipped, clipped)
    ref_minus_policy = reference_log_probs - current_log_probs
    kl_token = torch.exp(ref_minus_policy) - ref_minus_policy - 1.0
    sample_loss = (
        ((policy_token_loss + kl_coefficient * kl_token) * mask).sum(dim=-1)
        / lengths
    )
    loss = sample_loss.mean()
    metrics = {
        "loss": loss.detach(),
        "policy_loss": ((policy_token_loss * mask).sum(dim=-1) / lengths).mean().detach(),
        "kl": ((kl_token * mask).sum(dim=-1) / lengths).mean().detach(),
        "clip_fraction": (
            (((ratio - 1.0).abs() > clip_epsilon).to(mask.dtype) * mask).sum()
            / mask.sum().clamp_min(1.0)
        ).detach(),
        "mean_completion_tokens": lengths.mean().detach(),
    }
    return loss, metrics
