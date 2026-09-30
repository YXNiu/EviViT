#!/usr/bin/env python3
"""Offline source/result checks for the public EviViT repository."""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ALLOWED_SUFFIXES = {".py", ".sh", ".md", ".json", ".csv", ".cff"}
PAGE_ASSET_SUFFIXES = {".png", ".svg", ".mjs"}
REQUIRED_FILES = {
    "README.md",
    "verify_release.py",
    "run_final_eval.sh",
    "run_final_baseline.sh",
    "run_final_judge.sh",
    "run_final_train_bridge.sh",
    "run_matched_sft.sh",
    "configs/gpu_safety.json",
    "scripts/eval_evivit_v3_mid_bridge.py",
    "scripts/eval_qwen_original_qa.py",
    "scripts/judge_qa_predictions_local_qwen.py",
    "scripts/train_evivit_v3_bridge.py",
    "scripts/train_evivit_answer_sft.py",
    "results/paper_table1_local_pair_averages.json",
    "results/fine_grained_open_pairs.csv",
    "results/fine_grained_gain_heatmap.csv",
    "results/budget_scaling_frontier.csv",
    "results/paper_efficiency_table.csv",
    "results/paper_sft_recovery.csv",
    "results/sft_training_curves.csv",
}
PRIVATE_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9])/(?:Users|home|Volumes|private|data_[A-Za-z0-9]+|mnt)/"
)
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
SECRET = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*['\"][^'\"]{12,}"
)


def audit_results(errors: list[str]) -> None:
    results = ROOT / "results"
    expected_rows = {
        "fine_grained_open_pairs.csv": 63,
        "fine_grained_gain_heatmap.csv": 63,
        "budget_scaling_frontier.csv": 28,
        "paper_efficiency_table.csv": 12,
        "paper_sft_recovery.csv": 4,
        "sft_training_curves.csv": 31500,
    }
    for name, count in expected_rows.items():
        path = results / name
        if not path.is_file():
            errors.append(f"missing result file: {name}")
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != count:
            errors.append(f"result row count {name}: {len(rows)} != {count}")
        if any(not all(row.values()) for row in rows):
            # Global controls have blank gains; early curve points have no moving mean.
            if name not in {
                "paper_efficiency_table.csv",
                "paper_sft_recovery.csv",
                "sft_training_curves.csv",
            }:
                errors.append(f"empty result cell: {name}")
        if name == "sft_training_curves.csv":
            arms = {row["model"] for row in rows}
            if arms != {"Base+SFT", "EviViT+SFT"}:
                errors.append(f"unexpected SFT curve arms: {sorted(arms)}")
            for arm in arms:
                if max(float(row["epoch"]) for row in rows if row["model"] == arm) < 2.99:
                    errors.append(f"incomplete SFT curve: {arm}")
        if name == "fine_grained_open_pairs.csv":
            keys = {(row["host"], row["benchmark"]) for row in rows}
            if len(keys) != 63:
                errors.append("duplicate or missing fine-grained host/benchmark pair")
            for row in rows:
                displayed_gain = float(row["evivit_accuracy_pct"]) - float(row["base_accuracy_pct"])
                if abs(displayed_gain - float(row["gain_pp_from_displayed_cells"])) > 0.011:
                    errors.append(f"fine-grained displayed gain mismatch: {row['host']} {row['benchmark']}")
    summary_path = results / "paper_table1_local_pair_averages.json"
    if not summary_path.is_file():
        errors.append("missing Table 1 summary")
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pairs = summary["paired_results"]
    if len(pairs) != 9 or len({pair["host"] for pair in pairs}) != 9:
        errors.append("expected nine unique Table 1 host pairs")
    for pair in pairs:
        if abs(pair["base_average"] + pair["gain"] - pair["evivit_average"]) > 0.011:
            errors.append(f"Table 1 average/gain mismatch: {pair['host']}")


def literal_project_path(node: ast.AST) -> Path | None:
    """Resolve literal ROOT-relative paths without importing training code."""
    if isinstance(node, ast.Name) and node.id == "ROOT":
        return ROOT
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, str)
    ):
        parent = literal_project_path(node.left)
        if parent is not None:
            return parent / node.right.value
    return None


def audit_python(tree: ast.AST, relative: Path, errors: list[str]) -> None:
    for node in ast.walk(tree):
        project_path = literal_project_path(node)
        if project_path is not None and project_path.suffix in {".py", ".sh", ".json"}:
            if not project_path.is_file():
                errors.append(f"missing project source/config path: {relative}:{node.lineno}")
        modules: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        for module in modules:
            if module.split(".")[0] not in {"evivit_core", "scripts"}:
                continue
            module_path = ROOT / module.replace(".", "/")
            if not (
                module_path.with_suffix(".py").is_file()
                or (module_path / "__init__.py").is_file()
            ):
                errors.append(f"missing local import {module}: {relative}")


def main() -> int:
    errors: list[str] = []
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    )
    actual = {str(path.relative_to(ROOT)) for path in files}
    missing = REQUIRED_FILES - actual
    if missing:
        errors.append(f"missing required files: {sorted(missing)}")
    total_bytes = 0
    python_count = 0
    for path in files:
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            errors.append(f"symlink: {relative}")
            continue
        page_asset = (
            relative.parts[:2] == ("docs", "assets")
            and path.suffix in PAGE_ASSET_SUFFIXES
        )
        if path.suffix not in ALLOWED_SUFFIXES and not page_asset:
            errors.append(f"unexpected asset: {relative}")
            continue
        raw = path.read_bytes()
        total_bytes += len(raw)
        if page_asset and path.suffix == ".png":
            if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                errors.append(f"invalid PNG page asset: {relative}")
            continue
        if b"\x00" in raw:
            errors.append(f"binary content: {relative}")
            continue
        source = raw.decode("utf-8")
        if PRIVATE_PATH.search(source) or EMAIL.search(source):
            errors.append(f"identifying or machine-specific text: {relative}")
        if SECRET.search(source) or "-----BEGIN " + "PRIVATE KEY-----" in source:
            errors.append(f"possible credential: {relative}")
        if path.suffix != ".py":
            continue
        python_count += 1
        try:
            tree = ast.parse(source, filename=str(relative))
        except SyntaxError as exc:
            errors.append(f"Python syntax: {relative}: {exc}")
            continue
        audit_python(tree, relative, errors)
    if total_bytes > 100 * 1024 * 1024:
        errors.append("source directory exceeds 100 MiB")
    audit_results(errors)
    print(f"files={len(files)} python_files={python_count} bytes={total_bytes}")
    if errors:
        for error in sorted(set(errors)):
            print(f"FAIL {error}")
        return 1
    print("PASS source syntax, local imports/paths, result structure, path/identity scan, and size")
    print("NOTE this does not replace a GPU run with external assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
