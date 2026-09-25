"""Small compatibility helpers for supported Qwen multimodal families."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


SUPPORTED_QWEN_VLM_TYPES = frozenset({"qwen3_vl", "qwen3_5"})


def checkpoint_key_mapping(model_path: str | Path) -> dict[str, str] | None:
    """Return an exact legacy-to-current key map when a checkpoint needs it.

    Vision-OPD's released Qwen3.5 checkpoint stores the complete model using
    the pre-flattening Transformers names ``visual.*`` and
    ``language_model.model.*``. Current Qwen3.5 expects ``model.visual.*`` and
    ``model.language_model.*``. ``from_pretrained`` can rename keys while
    streaming, avoiding a second 10+ GB checkpoint just to change key names.
    """

    model_path = Path(model_path)
    single_weights = model_path / "model.safetensors"
    if not single_weights.is_file():
        return None

    from safetensors import safe_open

    with safe_open(single_weights, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    is_legacy_qwen35 = (
        bool(keys)
        and any(key.startswith("visual.") for key in keys)
        and any(key.startswith("language_model.model.") for key in keys)
        and not any(key.startswith("model.visual.") for key in keys)
    )
    if not is_legacy_qwen35:
        return None

    mapping: dict[str, str] = {}
    for key in keys:
        if key.startswith("visual."):
            mapping[key] = f"model.{key}"
        elif key.startswith("language_model.model."):
            suffix = key.removeprefix("language_model.model.")
            mapping[key] = f"model.language_model.{suffix}"
        elif key == "lm_head.weight":
            mapping[key] = key
    return mapping


def _qwen35_compatibility_base(model_path: Path) -> Path:
    """Locate the matching official Qwen3.5 skeleton for legacy checkpoints."""

    import os

    configured = os.environ.get("EVIVIT_QWEN35_BASE_MODEL")
    candidates = [
        Path(configured) if configured else None,
        model_path.parent / "qwen3_5_4b",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError(
        "legacy Qwen3.5 checkpoint needs a matching official base skeleton; "
        "set EVIVIT_QWEN35_BASE_MODEL or place qwen3_5_4b beside the checkpoint"
    )


def _load_legacy_qwen35(
    model_class: Any,
    model_path: Path,
    key_mapping: dict[str, str],
    *,
    device: str,
    dtype: torch.dtype,
    attn_implementation: str,
) -> Any:
    """Load legacy full weights into the current class without duplicating disk data."""

    from safetensors import safe_open
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    base_path = _qwen35_compatibility_base(model_path)
    model = model_class.from_pretrained(
        str(base_path),
        config=config,
        dtype=dtype,
        attn_implementation=attn_implementation,
        device_map={"": device},
        local_files_only=True,
    )
    target_state = model.state_dict()
    expected_targets = set(target_state)
    mapped_targets = set(key_mapping.values())
    missing = sorted(expected_targets - mapped_targets)
    unexpected = sorted(mapped_targets - expected_targets)
    if missing or unexpected:
        raise RuntimeError(
            "legacy checkpoint structural audit failed: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )

    weights_path = model_path / "model.safetensors"
    with torch.no_grad(), safe_open(weights_path, framework="pt", device="cpu") as handle:
        for source_name, target_name in key_mapping.items():
            source = handle.get_tensor(source_name)
            target = target_state[target_name]
            if tuple(source.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"legacy checkpoint shape mismatch for {source_name} -> {target_name}: "
                    f"{tuple(source.shape)} != {tuple(target.shape)}"
                )
            target.copy_(source.to(device=target.device, dtype=target.dtype))
    setattr(model, "_evivit_checkpoint_key_mapping_count", len(key_mapping))
    setattr(model, "_evivit_checkpoint_compatibility_base", str(base_path))
    return model


def load_qwen_model_class(
    model_class: Any,
    model_path: str | Path,
    *,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> Any:
    """Load a Qwen model and strictly audit compatibility key migration."""

    key_mapping = checkpoint_key_mapping(model_path)
    model_path = Path(model_path)
    if key_mapping is not None:
        return _load_legacy_qwen35(
            model_class,
            model_path,
            key_mapping,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
    return model_class.from_pretrained(
        str(model_path),
        dtype=dtype,
        attn_implementation=attn_implementation,
        device_map={"": device},
        local_files_only=True,
    )


def load_qwen_vlm(
    model_path: str | Path,
    *,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> tuple[Any, str]:
    """Load a known Qwen VLM using its exact Transformers model class."""

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    if config.model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration as model_class
    elif config.model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration as model_class
    else:
        raise ValueError(
            f"unsupported Qwen VLM family: {config.model_type!r}; "
            f"supported={sorted(SUPPORTED_QWEN_VLM_TYPES)}"
        )
    model = load_qwen_model_class(
        model_class,
        model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    return model, str(config.model_type)


def vision_probe_text(processor: Any, model_type: str) -> str:
    """Return a minimal valid image-bearing processor prompt."""

    if model_type == "qwen3_5":
        image_token = getattr(processor, "image_token", None)
        if not image_token:
            raise ValueError("Qwen3.5 processor does not expose image_token")
        return f"{image_token}\n."
    if model_type == "qwen3_vl":
        return "."
    raise ValueError(f"unsupported Qwen VLM family: {model_type!r}")


def language_model_visual_forward(
    model: Any,
    *,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    visual_mask: torch.Tensor,
    deepstack_visual_embeds: list[torch.Tensor],
    **kwargs: Any,
) -> Any:
    """Run the frozen text backbone with family-correct visual arguments."""

    call = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        **kwargs,
    }
    model_type = str(getattr(model.config, "model_type", ""))
    if model_type == "qwen3_vl":
        call["visual_pos_masks"] = visual_mask
        call["deepstack_visual_embeds"] = deepstack_visual_embeds
    elif model_type != "qwen3_5":
        raise ValueError(f"unsupported Qwen VLM family: {model_type!r}")
    return model.model.language_model(**call)
