"""GPU memory checks for model loading and evaluation."""

from __future__ import annotations

import csv
import io
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class GPUStatus:
    index: int
    name: str
    total_mib: int
    used_mib: int
    free_mib: int


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    gpu: GPUStatus
    requested_mib: int
    projected_total_used_mib: int
    projected_free_mib: int
    soft_total_used_mib: int
    hard_total_used_mib: int
    reasons: tuple[str, ...]

    def as_dict(self) -> dict:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


def parse_nvidia_smi_csv(payload: str) -> list[GPUStatus]:
    statuses = []
    for row in csv.reader(io.StringIO(payload)):
        if not row:
            continue
        values = [item.strip() for item in row]
        statuses.append(
            GPUStatus(
                index=int(values[0]),
                name=values[1],
                total_mib=int(values[2]),
                used_mib=int(values[3]),
                free_mib=int(values[4]),
            )
        )
    return statuses


def query_gpus() -> list[GPUStatus]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_nvidia_smi_csv(result.stdout)


def load_policy(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decide(
    gpu: GPUStatus,
    requested_mib: int,
    soft_total_used_mib: int,
    hard_total_used_mib: int,
    minimum_free_after_reservation_mib: int,
) -> SafetyDecision:
    projected_used = gpu.used_mib + requested_mib
    projected_free = gpu.free_mib - requested_mib
    reasons = []
    if gpu.used_mib >= hard_total_used_mib:
        reasons.append("GPU is already at or above the hard total-use limit")
    if projected_used > soft_total_used_mib:
        reasons.append("projected total use exceeds the soft limit")
    if projected_used > hard_total_used_mib:
        reasons.append("projected total use exceeds the hard limit")
    if projected_free < minimum_free_after_reservation_mib:
        reasons.append("projected free memory is below the required reserve")
    return SafetyDecision(
        allowed=not reasons,
        gpu=gpu,
        requested_mib=requested_mib,
        projected_total_used_mib=projected_used,
        projected_free_mib=projected_free,
        soft_total_used_mib=soft_total_used_mib,
        hard_total_used_mib=hard_total_used_mib,
        reasons=tuple(reasons),
    )


def check_gpu(gpu_index: int, requested_mib: int, policy_path: Path) -> SafetyDecision:
    policy = load_policy(policy_path)
    matches = [gpu for gpu in query_gpus() if gpu.index == gpu_index]
    if not matches:
        raise RuntimeError(f"GPU index {gpu_index} is not visible")
    return decide(
        matches[0],
        requested_mib=requested_mib,
        soft_total_used_mib=int(policy["soft_total_used_mib"]),
        hard_total_used_mib=int(policy["hard_total_used_mib"]),
        minimum_free_after_reservation_mib=int(
            policy["minimum_free_after_reservation_mib"]
        ),
    )
