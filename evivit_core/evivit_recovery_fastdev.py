"""Fail-closed contracts for the frozen Visual-CoT recovery fast-dev split."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
import uuid

from evivit_core.evivit_posttraining import LANGUAGE_LORA_TARGET_PATTERN
from evivit_core.evivit_recovery_router_sft import canonical_fingerprint, sha256_file


FASTDEV_ROWS = 600
FASTDEV_SELECTION_SHA256 = "f15418d5496eee9891d6d418154f1508d5f35d5a0a30b8ebada28f3f23cea430"
FASTDEV_MANIFEST_SHA256 = "078e1b7b7463eed2d7bb3fd12a86ade8119b2a4b1aca9a4b21b38f21b1fefc17"
FASTDEV_ROUTER_SHA256 = "517594e93376c6064d43af7cf50aa49ca402322de2ca77f9759bf290a1c810d5"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
FASTDEV_BUCKET_COUNTS = {"weak": 360, "protected": 120, "replay": 120}
FASTDEV_SOURCE_COUNTS = {
    "cub": 36,
    "docvqa": 49,
    "dude": 101,
    "flickr30k": 40,
    "gqa": 40,
    "infographicsvqa": 101,
    "openimages": 34,
    "sroie": 10,
    "textcap": 11,
    "textvqa": 137,
    "v7w": 13,
    "vsr": 28,
}
GROUP_DISPLAY = {"weak": "Weak", "protected": "Protect", "replay": "Replay"}
FASTDEV_MODE_TO_ANCHOR = {
    "B-L": "C0",
    "E-L": "C1",
    "E-B": "C1",
    "E-LB": "C1",
    "E-P": "C1",
    "E-PL": "C1",
    "E-PLB": "C1",
    "E-PLB-C": "C1",
    "E-PHLB": "C1",
}
ROUTER_RECOVERY_LANES = {"E-P", "E-PL", "E-PLB", "E-PLB-C", "E-PHLB"}
LANGUAGE_LORA_LANES = {
    "B-L",
    "E-L",
    "E-LB",
    "E-PL",
    "E-PLB",
    "E-PLB-C",
    "E-PHLB",
}
P3_LANES = {"C1", "E-L", "E-B", "E-LB", *ROUTER_RECOVERY_LANES}
TRAINED_BRIDGE_LANES = {"E-B", "E-LB", "E-PLB", "E-PLB-C", "E-PHLB"}
RECOVERY_ROW_FIELDS = (
    "selection_bucket",
    "recovery_group",
    "source_dataset",
    "source_release",
    "dev_role",
    "dev_seed",
    "selection_qa_sha256",
    "recovery_eval_protocol_fingerprint",
)

QWEN8B_CONFIG_SHA256 = "5cd452860dc1e9c29dd71cc3cef7f39b338b7a40793f7a260655c2d3568f3661"
QWEN8B_INDEX_SHA256 = "520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070"
QWEN8B_LANGUAGE_LAYERS = 36

P3_COMPONENT_SHA256 = {
    "selector": "d2c4896c7efb1c5cc1ae5bbef0da1d834eb521515f5143a8d5698e4d9de29814",
    "context_need": "363e55109f22af4d5c842458bcad5dbee634c3afa959043450d1e4e9ac21b659",
    "bridge": "7e0b42bff30132e961f6396bd30337bee79f527963177c85b4f292358316c744",
    "h_safe": "102bb49da4ede2c625c04cdb5b9267ea355562989aab119b2f581d6957fe6535",
}

P3_B16_PROTOCOL = {
    "host": "Qwen3-VL-8B-Instruct",
    "insertion_block": 18,
    "base_max_pixels": 16_777_216,
    "processor_min_pixels": 4096,
    "processor_max_pixels": 16_777_216,
    "processor_stream_contract": "shared_min_max_with_patch_exchange",
    "global_processor_max_pixels": None,
    "fine_processor_max_pixels": None,
    "global_token_budget": 2048,
    "fine_token_budget": 3072,
    "max_regions": 3,
    "minimum_region_tokens": 64,
    "evidence_policy": "mid_ptea",
    "evidence_decoder": "trace_split",
    "trace_split_budget_mode": "learned",
    "trace_split_min_context_fraction": 0.10,
    "trace_split_max_context_fraction": 0.25,
    "trace_split_context_expansion_factor": 1.0,
    "trace_split_context_decoder": "top_boxes",
    "adaptive_box_safety_mode": "replace",
    "patch_exchange": True,
    "patch_exchange_global_scale": 1.5,
    "patch_exchange_local_scale": 2.0,
    "patch_exchange_total_cap_ratio": 1.0,
    "patch_exchange_soft_floor_tokens": 2048,
    "patch_exchange_balanced_soft_floor": True,
    "patch_exchange_preserve_native_global": True,
    "patch_exchange_continuous_soft_floor": True,
    "patch_exchange_continuous_soft_floor_max_tokens": 4096,
    "patch_exchange_continuous_soft_floor_ramp_start_tokens": 4096,
    "patch_exchange_continuous_soft_floor_ramp_end_tokens": 12288,
    "patch_exchange_view_min_pixels": 4096,
    "fusion": "sparse_bridge",
    "bridge_mode": "bidirectional",
    "visual_output_mode": "append",
    "answer_protocol": "compact_json",
    "deterministic_inference": True,
    "deterministic_attention_backend": "flash",
    "max_new_tokens": 64,
}
RECOVERY_FORMAL_P3_KEYS = (
    "insertion_block",
    "global_token_budget",
    "fine_token_budget",
    "max_regions",
    "minimum_region_tokens",
    "evidence_decoder",
    "trace_split_budget_mode",
    "trace_split_min_context_fraction",
    "trace_split_max_context_fraction",
    "trace_split_context_expansion_factor",
    "trace_split_context_decoder",
    "patch_exchange",
    "patch_exchange_global_scale",
    "patch_exchange_local_scale",
    "patch_exchange_total_cap_ratio",
    "patch_exchange_soft_floor_tokens",
    "patch_exchange_balanced_soft_floor",
    "patch_exchange_preserve_native_global",
    "patch_exchange_continuous_soft_floor",
    "patch_exchange_continuous_soft_floor_max_tokens",
    "patch_exchange_continuous_soft_floor_ramp_start_tokens",
    "patch_exchange_continuous_soft_floor_ramp_end_tokens",
    "patch_exchange_view_min_pixels",
    "answer_protocol",
    "fusion",
    "bridge_mode",
    "visual_output_mode",
)
RECOVERY_FORMAL_P3_PROTOCOL = {
    key: P3_B16_PROTOCOL[key] for key in RECOVERY_FORMAL_P3_KEYS
}

FASTDEV_SELECTION_GATE = {
    "comparison": "candidate_minus_same_lane_untrained_anchor",
    "accuracy": "relaxed_correct",
    "aggregation": "source_macro_within_recovery_group",
    "eligible": {
        "Protect_delta_min": -0.005,
        "Replay_delta_min": -0.005,
        "json_valid_delta_min": -0.005,
    },
    "lexicographic": [
        "maximum Weak source-macro delta",
        "if Weak deltas differ by <=0.002: maximum Protect delta",
        "then maximum Replay delta",
        "then earlier checkpoint",
    ],
    "full_dev_policy": "top2_eligible_per_lane_only",
    "no_eligible_policy": "retain_untrained_anchor",
    "excluded_from_fastdev_gate": [
        "Fine-5 localization sentinel",
        "visual-token hard gate",
    ],
}


def read_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    if not rows and not allow_empty:
        raise ValueError(f"empty JSONL: {path}")
    return rows


@dataclass(frozen=True)
class LanguageAdapterAudit:
    path: Path
    config: dict[str, Any]
    config_sha256: str
    weights_file: Path
    weights_sha256: str
    bundle_fingerprint: str
    parameter_keys: int

    def protocol_fields(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "config_sha256": self.config_sha256,
            "weights_file": self.weights_file.name,
            "weights_sha256": self.weights_sha256,
            "bundle_fingerprint": self.bundle_fingerprint,
            "base_model_name_or_path": self.config["base_model_name_or_path"],
            "target_modules": self.config["target_modules"],
            "r": int(self.config["r"]),
            "lora_alpha": int(self.config["lora_alpha"]),
            "lora_dropout": float(self.config["lora_dropout"]),
            "parameter_keys": self.parameter_keys,
        }


@dataclass(frozen=True)
class RecoveryRouterCheckpointAudit:
    """CPU audit of the combined PTEA/H-Safe recovery checkpoint."""

    checkpoint: Path
    lane: str
    stage: str
    epoch: float
    micro_step: int
    optimizer_step: int
    training_state_path: Path
    training_state_sha256: str
    run_protocol_path: Path
    run_protocol_sha256: str
    protocol_fingerprint: str
    selector_tensors: int
    h_safe_tensors: int

    def component_fields(self, component: str, *, source: Path) -> dict[str, Any]:
        tensor_count = {
            "selector": self.selector_tensors,
            "h_safe": self.h_safe_tensors,
        }.get(component)
        if tensor_count is None:
            raise ValueError(f"unknown recovery router component: {component}")
        return {
            "path": str(self.training_state_path),
            "sha256": self.training_state_sha256,
            "state_key": f"{component}_state_dict",
            "parameter_tensors": tensor_count,
            "source_artifact": str(source.resolve()),
            "source_artifact_sha256": sha256_file(source.resolve()),
            "recovery_lane": self.lane,
            "recovery_stage": self.stage,
            "protocol_fingerprint": self.protocol_fingerprint,
        }


def resolve_recovery_adapter(checkpoint: Path, lane: str) -> Path:
    """Resolve the intentionally different Base and EviViT save layouts."""

    checkpoint = checkpoint.resolve()
    if lane == "B-L":
        candidate = checkpoint
    elif lane in {"E-L", "E-LB", "E-PL", "E-PLB", "E-PLB-C", "E-PHLB"}:
        candidate = checkpoint / "language_adapter"
    else:
        raise ValueError(f"lane has no language adapter: {lane}")
    if not (candidate / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"{lane} checkpoint lacks its exact adapter layout: {candidate}"
        )
    return candidate


def _validate_tensor_state(
    value: Any,
    *,
    expected: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    import torch

    if not isinstance(value, dict) or not value:
        raise ValueError(f"recovery checkpoint lacks {label} tensor state")
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        unexpected = sorted(set(value) - set(expected))
        raise ValueError(
            f"recovery {label} keys differ from source architecture: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    for name, tensor in value.items():
        source = expected[name]
        if not isinstance(tensor, torch.Tensor) or not isinstance(source, torch.Tensor):
            raise ValueError(f"recovery {label}.{name} is not a tensor")
        if tensor.shape != source.shape or tensor.dtype != source.dtype:
            raise ValueError(
                f"recovery {label}.{name} shape/dtype mismatch: "
                f"{tuple(tensor.shape)}/{tensor.dtype} versus "
                f"{tuple(source.shape)}/{source.dtype}"
            )
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError(f"recovery {label}.{name} contains non-finite values")
    return value


def validate_recovery_router_checkpoint(
    checkpoint: Path,
    *,
    lane: str,
    selector_source: Path,
    h_safe_source: Path,
    expected_model: Path,
    expected_bridge_source: Path,
) -> RecoveryRouterCheckpointAudit:
    """Fail closed before a trained PTEA/H-Safe checkpoint reaches CUDA."""

    if lane not in ROUTER_RECOVERY_LANES:
        raise ValueError(f"lane has no trained recovery router: {lane}")
    checkpoint = checkpoint.resolve()
    selector_source = selector_source.resolve()
    h_safe_source = h_safe_source.resolve()
    expected_model = expected_model.resolve()
    expected_bridge_source = expected_bridge_source.resolve()
    state_path = checkpoint / "training_state.pt"
    state_json_path = checkpoint / "training_state.json"
    protocol_path = checkpoint / "run_protocol.json"
    for path in (state_path, state_json_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    import torch

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    state_json = json.loads(state_json_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or not isinstance(protocol, dict):
        raise ValueError("recovery router state/protocol must be mappings")
    if state.get("format_version") != "evivit_visual_cot_recovery_sft_v1":
        raise ValueError("recovery router state format mismatch")
    if protocol.get("format_version") != "evivit_visual_cot_recovery_sft_v1":
        raise ValueError("recovery router protocol format mismatch")
    if state.get("lane") != lane or protocol.get("lane") != lane:
        raise ValueError("recovery router checkpoint lane mismatch")
    stage = str(protocol.get("recovery_stage", ""))
    expected_stage = "B" if lane in {"E-PLB-C", "E-PHLB"} else "A"
    if stage != expected_stage or state.get("recovery_stage") != expected_stage:
        raise ValueError("recovery router checkpoint stage mismatch")
    router = protocol.get("router")
    if not isinstance(router, dict) or router.get("stage") != expected_stage:
        raise ValueError("recovery router protocol stage is missing or inconsistent")

    unsigned = {key: value for key, value in protocol.items() if key != "protocol_fingerprint"}
    fingerprint = canonical_fingerprint(unsigned)
    # E-P/E-PL formal jobs launched before the canonical serializer was wired
    # into the trainer used the same sorted JSON payload with default
    # whitespace.  Accept that one narrowly defined legacy serialization so
    # their already-written checkpoints remain auditable; all newly launched
    # jobs use ``canonical_fingerprint`` above.
    legacy_encoded = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    legacy_fingerprint = hashlib.sha256(legacy_encoded).hexdigest()
    stored_fingerprint = protocol.get("protocol_fingerprint")
    if stored_fingerprint not in {fingerprint, legacy_fingerprint}:
        raise ValueError("recovery router protocol fingerprint is stale")
    if state.get("protocol_fingerprint") != stored_fingerprint:
        raise ValueError("recovery router state/protocol fingerprint mismatch")

    rows = int(protocol.get("rows", -1))
    accumulation = int(protocol.get("gradient_accumulation", -1))
    max_steps = int(protocol.get("max_steps", -1))
    if rows <= 0 or rows % 2 or accumulation != 20 or max_steps != 3 * rows:
        raise ValueError("recovery router checkpoint is not a formal three-epoch run")
    match = re.fullmatch(r"checkpoint_step_(\d+)", checkpoint.name)
    micro_step = int(state.get("micro_step", -1))
    optimizer_step = int(state.get("optimizer_step", -1))
    if (
        match is None
        or int(match.group(1)) != micro_step
        or micro_step <= 0
        or micro_step % accumulation
        or optimizer_step != micro_step // accumulation
    ):
        raise ValueError("recovery router checkpoint name/state/optimizer mismatch")
    epoch = micro_step / rows
    if epoch not in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}:
        raise ValueError(f"checkpoint is not a preregistered half epoch: {epoch}")
    if expected_stage == "B" and epoch <= 0.5:
        raise ValueError("Stage-B evaluation requires a post-fork checkpoint")
    if state_json != {
        "format_version": "evivit_visual_cot_recovery_sft_v1",
        "protocol_fingerprint": stored_fingerprint,
        "micro_step": micro_step,
        "optimizer_step": optimizer_step,
        "stage_optimizer_step": int(state.get("stage_optimizer_step", -1)),
    }:
        raise ValueError("training_state.json does not exactly mirror training_state.pt")

    if Path(str(protocol.get("model", ""))).resolve() != expected_model:
        raise ValueError("recovery router host model differs from fast-dev host")
    if protocol.get("selector_sha256") != sha256_file(selector_source):
        raise ValueError("recovery router source selector hash mismatch")
    if Path(str(protocol.get("bridge_initialization", ""))).resolve() != expected_bridge_source:
        raise ValueError("recovery router bridge initialization differs from frozen P3")
    formal = protocol.get("formal_p3")
    if formal != RECOVERY_FORMAL_P3_PROTOCOL:
        raise ValueError("recovery router formal P3/B16 contract mismatch")
    alignment = router.get("alignment")
    if not isinstance(alignment, dict) or int(alignment.get("rows", -1)) != rows:
        raise ValueError("recovery router alignment row count mismatch")
    if alignment.get("train_manifest_sha256") != protocol.get("manifest_sha256"):
        raise ValueError("recovery router train manifest hash mismatch")
    for key in ("train_manifest_sha256", "router_aux_sha256", "order_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(alignment.get(key, ""))):
            raise ValueError(f"recovery router alignment lacks valid {key}")
    if router.get("teacher_selector_sha256") != sha256_file(selector_source):
        raise ValueError("recovery router teacher selector hash mismatch")
    h_safe_contract = router.get("h_safe")
    if (
        not isinstance(h_safe_contract, dict)
        or h_safe_contract.get("artifact_sha256") != sha256_file(h_safe_source)
    ):
        raise ValueError("recovery router H-Safe source hash mismatch")
    expected_h_mode = "trainable_geometry_only" if lane == "E-PHLB" else "frozen"
    if h_safe_contract.get("mode") != expected_h_mode:
        raise ValueError("recovery router H-Safe mode mismatch")
    if expected_stage == "B":
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(protocol.get("parent_protocol_fingerprint", ""))
        ) or float(protocol.get("stage_start_micro_step", -1)) != rows / 2:
            raise ValueError("Stage-B parent binding is missing or not the 0.5-epoch fork")

    selector_payload = torch.load(selector_source, map_location="cpu", weights_only=False)
    h_safe_payload = torch.load(h_safe_source, map_location="cpu", weights_only=False)
    if selector_payload.get("selector_architecture") != "dual_residual_context_ptea":
        raise ValueError("recovery fast-dev requires the dual PTEA source architecture")
    if h_safe_payload.get("version") != "evivit_v6_adaptivebox_head_v1":
        raise ValueError("recovery fast-dev requires the frozen v6 H-Safe source")
    selector_state = _validate_tensor_state(
        state.get("selector_state_dict"),
        expected=selector_payload.get("model", {}),
        label="selector_state_dict",
    )
    h_safe_state = state.get("h_safe_state_dict")
    if lane == "E-PHLB":
        h_safe_state = _validate_tensor_state(
            h_safe_state,
            expected=h_safe_payload.get("state_dict", {}),
            label="h_safe_state_dict",
        )
    elif h_safe_state is not None:
        raise ValueError("non-E-PHLB checkpoint unexpectedly stores trainable H-Safe")
    return RecoveryRouterCheckpointAudit(
        checkpoint=checkpoint,
        lane=lane,
        stage=stage,
        epoch=epoch,
        micro_step=micro_step,
        optimizer_step=optimizer_step,
        training_state_path=state_path,
        training_state_sha256=sha256_file(state_path),
        run_protocol_path=protocol_path,
        run_protocol_sha256=sha256_file(protocol_path),
        protocol_fingerprint=fingerprint,
        selector_tensors=len(selector_state),
        h_safe_tensors=len(h_safe_state or {}),
    )


def materialize_recovery_router_eval_artifacts(
    audit: RecoveryRouterCheckpointAudit,
    *,
    selector_source: Path,
    h_safe_source: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Create evaluator-native PTEA/H-Safe files from one audited state."""

    import torch

    if sha256_file(audit.training_state_path) != audit.training_state_sha256:
        raise ValueError("recovery training state changed after CPU validation")
    state = torch.load(
        audit.training_state_path, map_location="cpu", weights_only=False
    )
    selector_payload = torch.load(
        selector_source, map_location="cpu", weights_only=False
    )
    _validate_tensor_state(
        state.get("selector_state_dict"),
        expected=selector_payload.get("model", {}),
        label="selector_state_dict",
    )
    selector_payload["model"] = state["selector_state_dict"]
    selector_payload["recovery_eval_provenance"] = {
        "lane": audit.lane,
        "stage": audit.stage,
        "epoch": audit.epoch,
        "training_state_sha256": audit.training_state_sha256,
        "protocol_fingerprint": audit.protocol_fingerprint,
    }
    outputs = {"selector": output_dir / "recovery_selector_checkpoint.pt"}
    if audit.lane == "E-PHLB":
        h_safe_payload = torch.load(
            h_safe_source, map_location="cpu", weights_only=False
        )
        _validate_tensor_state(
            state.get("h_safe_state_dict"),
            expected=h_safe_payload.get("state_dict", {}),
            label="h_safe_state_dict",
        )
        h_safe_payload["state_dict"] = state["h_safe_state_dict"]
        h_safe_payload["recovery_eval_provenance"] = dict(
            selector_payload["recovery_eval_provenance"]
        )
        outputs["h_safe"] = output_dir / "recovery_h_safe_checkpoint.pt"
        payloads = {"selector": selector_payload, "h_safe": h_safe_payload}
    else:
        payloads = {"selector": selector_payload}
    for name, destination in outputs.items():
        if destination.exists():
            raise FileExistsError(destination)
        temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
        try:
            torch.save(payloads[name], temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return outputs


def validate_recovery_bridge_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Audit a formal E-B/E-LB bridge bundle without initializing CUDA."""

    checkpoint = checkpoint.resolve()
    state_path = checkpoint / "training_state.json"
    bridge_path = checkpoint / "bridge_checkpoint.pt"
    parent_protocol_path = checkpoint / "run_protocol.json"
    if not parent_protocol_path.is_file():
        parent_protocol_path = checkpoint.parent / "run_protocol.json"
    for path in (state_path, bridge_path, parent_protocol_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    if state.get("format_version") != "evivit_visual_cot_recovery_sft_v1":
        raise ValueError("recovery bridge training-state format mismatch")
    fingerprint = str(state.get("protocol_fingerprint", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("recovery bridge protocol fingerprint is invalid")
    if parent_protocol.get("protocol_fingerprint") != fingerprint:
        raise ValueError("checkpoint is not bound to its parent training protocol")
    micro_step = int(state.get("micro_step", -1))
    optimizer_step = int(state.get("optimizer_step", -1))
    if micro_step <= 0 or optimizer_step != micro_step // 20:
        raise ValueError("recovery bridge micro/optimizer step mismatch")

    import torch

    payload = torch.load(bridge_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("recovery bridge checkpoint is not a mapping")
    if payload.get("version") != "evivit_v3_sparse_mid_bridge_v1":
        raise ValueError("unexpected recovery bridge checkpoint version")
    if int(payload.get("step", -1)) != optimizer_step:
        raise ValueError("bridge checkpoint and training-state steps differ")
    expected_config = {
        "insertion_block": 18,
        "bridge_dim": 256,
        "bridge_heads": 4,
        "neighborhood_radius": 1,
        "max_relative_residual": 0.2,
    }
    if payload.get("config") != expected_config:
        raise ValueError(
            f"recovery bridge architecture mismatch: {payload.get('config')}"
        )
    tensors = payload.get("bridge")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("recovery bridge checkpoint contains no bridge tensors")
    if any(not hasattr(value, "shape") for value in tensors.values()):
        raise ValueError("recovery bridge state contains a non-tensor value")
    return {
        "path": str(bridge_path),
        "sha256": sha256_file(bridge_path),
        "version": payload["version"],
        "parameter_tensors": len(tensors),
        "micro_step": micro_step,
        "optimizer_step": optimizer_step,
        "parent_protocol_fingerprint": fingerprint,
    }


def validate_language_lora_adapter(
    adapter: Path,
    *,
    expected_base_model: Path,
) -> LanguageAdapterAudit:
    """Validate PEFT identity before allowing model loading."""

    adapter = adapter.resolve()
    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required_exact = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "target_modules": LANGUAGE_LORA_TARGET_PATTERN,
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "modules_to_save": None,
    }
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in required_exact.items()
        if config.get(key) != expected
    }
    configured_base = config.get("base_model_name_or_path")
    if not configured_base:
        mismatches["base_model_name_or_path"] = (configured_base, str(expected_base_model))
    else:
        configured_path = Path(str(configured_base)).resolve()
        if configured_path != expected_base_model.resolve():
            mismatches["base_model_name_or_path"] = (
                str(configured_path),
                str(expected_base_model.resolve()),
            )
    if mismatches:
        raise ValueError(f"language LoRA adapter contract mismatch: {mismatches}")

    candidates = [
        path
        for path in (
            adapter / "adapter_model.safetensors",
            adapter / "adapter_model.bin",
        )
        if path.is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"adapter must contain exactly one supported weights file, found {candidates}"
        )
    weights = candidates[0]
    parameter_keys = 0
    if weights.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(weights), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
        key_pattern = re.compile(
            r"^base_model\.model\.model\.language_model\.layers\.(\d+)\."
            r"self_attn\.(q_proj|k_proj|v_proj|o_proj)\.lora_(A|B)\.weight$"
        )
        matches = [key_pattern.fullmatch(key) for key in keys]
        unexpected = [key for key, match in zip(keys, matches) if match is None]
        if unexpected:
            raise ValueError(f"adapter contains non-whitelisted tensors: {unexpected[:5]}")
        actual_slots = {
            (int(match.group(1)), match.group(2), match.group(3))
            for match in matches
            if match is not None
        }
        expected_slots = {
            (layer, projection, side)
            for layer in range(QWEN8B_LANGUAGE_LAYERS)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
            for side in ("A", "B")
        }
        if actual_slots != expected_slots or len(keys) != len(expected_slots):
            raise ValueError(
                "adapter does not contain the exact 36-layer q/k/v/o LoRA A/B set"
            )
        parameter_keys = len(keys)
    else:
        # Pickle weights cannot be key-audited without loading arbitrary code.
        # Recovery evaluation therefore accepts safetensors only.
        raise ValueError("recovery fast-dev requires adapter_model.safetensors")

    config_hash = sha256_file(config_path)
    weights_hash = sha256_file(weights)
    bundle = {
        "config_sha256": config_hash,
        "weights_name": weights.name,
        "weights_sha256": weights_hash,
        "parameter_keys": parameter_keys,
    }
    return LanguageAdapterAudit(
        path=adapter,
        config=config,
        config_sha256=config_hash,
        weights_file=weights,
        weights_sha256=weights_hash,
        bundle_fingerprint=canonical_fingerprint(bundle),
        parameter_keys=parameter_keys,
    )


@dataclass(frozen=True)
class FastDevBundle:
    root: Path
    merged_rows: tuple[dict[str, Any], ...]
    order_sha256: str

    def protocol_fields(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "rows": len(self.merged_rows),
            "selection_sha256": FASTDEV_SELECTION_SHA256,
            "manifest_sha256": FASTDEV_MANIFEST_SHA256,
            "router_sha256": FASTDEV_ROUTER_SHA256,
            "order_sha256": self.order_sha256,
            "bucket_counts": FASTDEV_BUCKET_COUNTS,
            "source_counts": FASTDEV_SOURCE_COUNTS,
            "eval_hash_overlap_rows": 0,
        }


def validate_fastdev_bundle(root: Path) -> FastDevBundle:
    """Validate the immutable 600-row split and recover its strata by ID."""

    root = root.resolve()
    selection_path = root / "fast_dev_600.selection.jsonl"
    materialized = root / "fast_dev_materialized"
    manifest_path = materialized / "train.qa.jsonl"
    router_path = materialized / "router_aux.jsonl"
    overlaps_path = materialized / "eval_hash_overlaps.jsonl"
    expected_hashes = {
        selection_path: FASTDEV_SELECTION_SHA256,
        manifest_path: FASTDEV_MANIFEST_SHA256,
        router_path: FASTDEV_ROUTER_SHA256,
        overlaps_path: EMPTY_SHA256,
    }
    for path, expected in expected_hashes.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen fast-dev hash mismatch for {path}: {actual}")
    if read_jsonl(overlaps_path, allow_empty=True):
        raise ValueError("fast-dev contains evaluation image-hash overlap")

    ready = json.loads((materialized / "READY").read_text(encoding="utf-8"))
    router_ready = json.loads(
        (materialized / "ROUTER_AUX_READY").read_text(encoding="utf-8")
    )
    if ready != {
        "rows": FASTDEV_ROWS,
        "selection_sha256": FASTDEV_SELECTION_SHA256,
        "train_manifest_sha256": FASTDEV_MANIFEST_SHA256,
    }:
        raise ValueError(f"fast-dev READY mismatch: {ready}")
    if router_ready != {
        "router_manifest_sha256": FASTDEV_ROUTER_SHA256,
        "rows": FASTDEV_ROWS,
        "selection_sha256": FASTDEV_SELECTION_SHA256,
    }:
        raise ValueError(f"fast-dev ROUTER_AUX_READY mismatch: {router_ready}")
    audit = json.loads(
        (materialized / "materialization_audit.json").read_text(encoding="utf-8")
    )
    audit_required = {
        "selected_rows": FASTDEV_ROWS,
        "resolved_rows_without_eval_overlap": FASTDEV_ROWS,
        "unique_training_image_hashes": FASTDEV_ROWS,
        "unresolved_rows": 0,
        "corrupt_rows": 0,
        "eval_hash_overlap_rows": 0,
        "valid_router_bbox_rows": FASTDEV_ROWS,
        "invalid_router_bbox_rows": 0,
        "training_ready": True,
        "router_aux_training_ready": True,
    }
    audit_mismatch = {
        key: (audit.get(key), value)
        for key, value in audit_required.items()
        if audit.get(key) != value
    }
    if audit_mismatch:
        raise ValueError(f"fast-dev materialization audit mismatch: {audit_mismatch}")

    selection = read_jsonl(selection_path)
    manifest = read_jsonl(manifest_path)
    router = read_jsonl(router_path)
    if not (len(selection) == len(manifest) == len(router) == FASTDEV_ROWS):
        raise ValueError("fast-dev sources do not all contain exactly 600 rows")
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    buckets: dict[str, int] = {}
    sources: dict[str, int] = {}
    for index, (selected, qa, aux) in enumerate(zip(selection, manifest, router)):
        identities = (
            str(selected.get("selection_id", "")),
            str(qa.get("id", "")),
            str(aux.get("id", "")),
        )
        if not identities[0] or len(set(identities)) != 1:
            raise ValueError(f"fast-dev ID/order mismatch at row {index}: {identities}")
        row_id = identities[0]
        if row_id in seen:
            raise ValueError(f"duplicate fast-dev ID: {row_id}")
        seen.add(row_id)
        for key in ("question", "answer", "source_dataset"):
            if selected.get(key) != qa.get(key) or qa.get(key) != aux.get(key):
                raise ValueError(f"fast-dev {key} mismatch for {row_id}")
        bucket = str(selected.get("selection_bucket", ""))
        source = str(qa.get("source_dataset", ""))
        if bucket not in GROUP_DISPLAY:
            raise ValueError(f"unknown recovery bucket for {row_id}: {bucket}")
        buckets[bucket] = buckets.get(bucket, 0) + 1
        sources[source] = sources.get(source, 0) + 1
        merged.append(
            {
                **qa,
                "eval_tier": "recovery_fastdev600",
                "selection_bucket": bucket,
                "recovery_group": GROUP_DISPLAY[bucket],
                "source_dataset": source,
                "source_release": selected.get("source_release"),
                "dev_role": "fast",
                "dev_seed": int(selected.get("dev_seed")),
                "selection_qa_sha256": selected.get("qa_sha256"),
            }
        )
    if buckets != FASTDEV_BUCKET_COUNTS:
        raise ValueError(f"fast-dev bucket counts drifted: {buckets}")
    if sources != FASTDEV_SOURCE_COUNTS:
        raise ValueError(f"fast-dev source counts drifted: {sources}")
    order_hash = hashlib.sha256(
        b"\0".join(str(row["id"]).encode("utf-8") for row in merged)
    ).hexdigest()
    return FastDevBundle(root=root, merged_rows=tuple(merged), order_sha256=order_hash)


def validate_qwen8b_model(model: Path) -> dict[str, Any]:
    model = model.resolve()
    config_path = model / "config.json"
    index_path = model / "model.safetensors.index.json"
    if sha256_file(config_path) != QWEN8B_CONFIG_SHA256:
        raise ValueError("Qwen3-VL-8B config hash mismatch")
    if sha256_file(index_path) != QWEN8B_INDEX_SHA256:
        raise ValueError("Qwen3-VL-8B weight-index hash mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl":
        raise ValueError(f"unexpected recovery host model_type: {config.get('model_type')}")
    return {
        "path": str(model),
        "config_sha256": QWEN8B_CONFIG_SHA256,
        "weight_index_sha256": QWEN8B_INDEX_SHA256,
        "model_type": config["model_type"],
    }


def validate_p3_components(paths: Mapping[str, Path]) -> dict[str, Any]:
    if set(paths) != set(P3_COMPONENT_SHA256):
        raise ValueError("P3 component set must be selector/context_need/bridge/h_safe")
    result = {}
    for name, expected in P3_COMPONENT_SHA256.items():
        path = paths[name].resolve()
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen P3 {name} hash mismatch: {actual}")
        result[name] = {"path": str(path), "sha256": actual}
    return result


def build_fastdev_protocol(
    *,
    lane: str,
    bundle: FastDevBundle,
    model: Mapping[str, Any],
    adapter: LanguageAdapterAudit | None,
    checkpoint: Path | None,
    checkpoint_epoch: float | None,
    checkpoint_step: int | None,
    p3_components: Mapping[str, Any] | None,
    evaluation_rows: int = FASTDEV_ROWS,
) -> dict[str, Any]:
    supported = {"C0", "C1", *FASTDEV_MODE_TO_ANCHOR}
    if lane not in supported:
        raise ValueError(f"unsupported recovery fast-dev lane: {lane}")
    is_candidate = lane in FASTDEV_MODE_TO_ANCHOR
    if is_candidate != (checkpoint is not None):
        raise ValueError("candidate modes require a checkpoint; anchors forbid it")
    expects_adapter = lane in LANGUAGE_LORA_LANES
    if expects_adapter != (adapter is not None):
        raise ValueError("language-adapter presence does not match the recovery lane")
    if is_candidate and (checkpoint_epoch is None or checkpoint_step is None):
        raise ValueError("candidate checkpoint position is required")
    if not is_candidate and (checkpoint_epoch is not None or checkpoint_step is not None):
        raise ValueError("untrained anchors cannot carry checkpoint position")
    if evaluation_rows not in {2, FASTDEV_ROWS}:
        raise ValueError("recovery evaluation is either isolated smoke2 or full fast-dev600")
    uses_p3 = lane in P3_LANES
    payload: dict[str, Any] = {
        "format_version": "visual_cot_recovery_fastdev_eval_v1",
        "lane": lane,
        "role": "candidate" if is_candidate else "same_path_untrained_anchor",
        "paired_anchor": FASTDEV_MODE_TO_ANCHOR.get(lane),
        "checkpoint": str(checkpoint.resolve()) if checkpoint is not None else None,
        "checkpoint_epoch": (
            float(checkpoint_epoch) if checkpoint_epoch is not None else None
        ),
        "checkpoint_step": int(checkpoint_step) if checkpoint_step is not None else None,
        "fastdev": bundle.protocol_fields(),
        "evaluation_scope": {
            "kind": "full_fastdev600" if evaluation_rows == FASTDEV_ROWS else "engineering_smoke2",
            "rows": evaluation_rows,
            "scorer_eligible": evaluation_rows == FASTDEV_ROWS,
        },
        "model": dict(model),
        "language_lora_adapter": adapter.protocol_fields() if adapter is not None else None,
        "actor": (
            {
                "kind": (
                    "base_original_plus_lora"
                    if lane == "B-L"
                    else "base_original_no_lora"
                ),
                "max_pixels": P3_B16_PROTOCOL["base_max_pixels"],
                "min_pixels": P3_B16_PROTOCOL["processor_min_pixels"],
                "prompt_variant": "evivit_compact_json",
                "answer_protocol": "compact_json",
                "max_new_tokens": 64,
            }
            if not uses_p3
            else {
                "kind": {
                    "C1": "frozen_p3_no_lora",
                    "E-L": "frozen_p3_plus_lora",
                    "E-B": "trained_p3_bridge_no_lora",
                    "E-LB": "trained_p3_bridge_plus_lora",
                    "E-P": "trained_p3_ptea_no_lora",
                    "E-PL": "trained_p3_ptea_plus_lora",
                    "E-PLB": "trained_p3_ptea_bridge_plus_lora",
                    "E-PLB-C": "trained_p3_ptea_bridge_frozen_hsafe_plus_lora",
                    "E-PHLB": "trained_p3_ptea_hsafe_bridge_plus_lora",
                }[lane],
                **P3_B16_PROTOCOL,
                "components": dict(p3_components or {}),
                "trained_components": (
                    ["sparse_bridge"]
                    if lane == "E-B"
                    else ["language_lora", "sparse_bridge"]
                    if lane == "E-LB"
                    else ["language_lora"]
                    if lane == "E-L"
                    else ["ptea"]
                    if lane == "E-P"
                    else ["ptea", "language_lora"]
                    if lane == "E-PL"
                    else ["ptea", "language_lora", "sparse_bridge"]
                    if lane in {"E-PLB", "E-PLB-C"}
                    else ["ptea", "h_safe", "language_lora", "sparse_bridge"]
                    if lane == "E-PHLB"
                    else []
                ),
                "inference_parameters_frozen": True,
            }
        ),
        "selection_gate": FASTDEV_SELECTION_GATE,
    }
    if uses_p3 and not p3_components:
        raise ValueError("C1/E-L fast-dev protocol requires frozen P3 components")
    if not uses_p3 and p3_components is not None:
        raise ValueError("C0/B-L protocol cannot carry P3 components")
    # Preserve the exact already-running C1/E-L protocol payload and hashes.
    if lane in {"C1", "E-L"}:
        actor = payload["actor"]
        actor.pop("trained_components", None)
        actor.pop("inference_parameters_frozen", None)
        actor["all_evivit_modules_frozen"] = True
    payload["protocol_fingerprint"] = canonical_fingerprint(payload)
    return payload


def annotate_fastdev_rows(
    bundle: FastDevBundle, protocol_fingerprint: str
) -> list[dict[str, Any]]:
    if not re.fullmatch(r"[0-9a-f]{64}", protocol_fingerprint):
        raise ValueError("invalid fast-dev protocol fingerprint")
    return [
        {**row, "recovery_eval_protocol_fingerprint": protocol_fingerprint}
        for row in bundle.merged_rows
    ]


def recovery_row_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy recovery-only fields without changing legacy evaluator rows."""

    return {key: row[key] for key in RECOVERY_ROW_FIELDS if key in row}


__all__ = [
    "FASTDEV_BUCKET_COUNTS",
    "FASTDEV_MANIFEST_SHA256",
    "FASTDEV_MODE_TO_ANCHOR",
    "FASTDEV_ROWS",
    "FASTDEV_ROUTER_SHA256",
    "FASTDEV_SELECTION_GATE",
    "FASTDEV_SELECTION_SHA256",
    "FASTDEV_SOURCE_COUNTS",
    "FastDevBundle",
    "LanguageAdapterAudit",
    "LANGUAGE_LORA_LANES",
    "P3_B16_PROTOCOL",
    "P3_COMPONENT_SHA256",
    "P3_LANES",
    "QWEN8B_LANGUAGE_LAYERS",
    "RECOVERY_ROW_FIELDS",
    "RECOVERY_FORMAL_P3_PROTOCOL",
    "ROUTER_RECOVERY_LANES",
    "RecoveryRouterCheckpointAudit",
    "TRAINED_BRIDGE_LANES",
    "annotate_fastdev_rows",
    "build_fastdev_protocol",
    "read_jsonl",
    "materialize_recovery_router_eval_artifacts",
    "recovery_row_metadata",
    "resolve_recovery_adapter",
    "validate_fastdev_bundle",
    "validate_language_lora_adapter",
    "validate_p3_components",
    "validate_qwen8b_model",
    "validate_recovery_bridge_checkpoint",
    "validate_recovery_router_checkpoint",
]
