#!/usr/bin/env python3
"""Audit paired correctness changes between a baseline and a candidate.

The script is deliberately judge-backend agnostic.  It consumes two JSONL
files that share stable sample IDs and contain a boolean correctness field
(normally ``judge_correct``).  Besides net accuracy, it reports the quantities
needed by preservation-aware EviViT experiments: rescues, regressions,
preservation rate, rescue precision, and the harm-to-gain ratio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path, id_field: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row[id_field])
            if sample_id in rows:
                raise ValueError(f"duplicate {id_field}={sample_id!r} in {path}:{line_number}")
            rows[sample_id] = row
    return rows


def _as_bool(value: Any, *, field: str, sample_id: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"invalid boolean {field} for sample {sample_id!r}: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--correct-field", default="judge_correct")
    parser.add_argument("--strict-complete", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    baseline = _load(args.baseline, args.id_field)
    candidate = _load(args.candidate, args.id_field)
    baseline_ids, candidate_ids = set(baseline), set(candidate)
    shared = sorted(baseline_ids & candidate_ids)
    only_baseline = sorted(baseline_ids - candidate_ids)
    only_candidate = sorted(candidate_ids - baseline_ids)
    if args.strict_complete and (only_baseline or only_candidate):
        raise ValueError(
            f"ID mismatch: baseline_only={len(only_baseline)}, "
            f"candidate_only={len(only_candidate)}"
        )
    if not shared:
        raise ValueError("the two files contain no shared sample IDs")

    wins: list[str] = []
    losses: list[str] = []
    stable_correct: list[str] = []
    stable_wrong: list[str] = []
    for sample_id in shared:
        base_ok = _as_bool(
            baseline[sample_id].get(args.correct_field),
            field=args.correct_field,
            sample_id=sample_id,
        )
        cand_ok = _as_bool(
            candidate[sample_id].get(args.correct_field),
            field=args.correct_field,
            sample_id=sample_id,
        )
        if not base_ok and cand_ok:
            wins.append(sample_id)
        elif base_ok and not cand_ok:
            losses.append(sample_id)
        elif base_ok:
            stable_correct.append(sample_id)
        else:
            stable_wrong.append(sample_id)

    n = len(shared)
    base_correct = len(stable_correct) + len(losses)
    candidate_correct = len(stable_correct) + len(wins)
    changed = len(wins) + len(losses)
    base_wrong = n - base_correct
    result = {
        "format_version": "paired_preservation_audit_v1",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "id_field": args.id_field,
        "correct_field": args.correct_field,
        "shared_samples": n,
        "baseline_only": len(only_baseline),
        "candidate_only": len(only_candidate),
        "baseline_correct": base_correct,
        "candidate_correct": candidate_correct,
        "net_gain": candidate_correct - base_correct,
        "wins_rescued": len(wins),
        "losses_regressed": len(losses),
        "stable_correct": len(stable_correct),
        "stable_wrong": len(stable_wrong),
        "flip_rate": changed / n,
        "preservation_rate": len(stable_correct) / base_correct if base_correct else None,
        "rescue_rate": len(wins) / base_wrong if base_wrong else None,
        "rescue_precision_among_flips": len(wins) / changed if changed else None,
        "harm_to_gain_ratio": len(losses) / len(wins) if wins else None,
        "wins_ids": wins,
        "losses_ids": losses,
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if args.output_md:
        def pct(value: float | None) -> str:
            return "N/A" if value is None else f"{100.0 * value:.2f}%"

        markdown = "\n".join(
            [
                "# Paired preservation audit",
                "",
                f"- Shared samples: `{n}`",
                f"- Baseline / candidate correct: `{base_correct}` / `{candidate_correct}`",
                f"- Wins / losses / net: `{len(wins)}` / `{len(losses)}` / `{candidate_correct - base_correct:+d}`",
                f"- Preservation rate: `{pct(result['preservation_rate'])}`",
                f"- Rescue rate: `{pct(result['rescue_rate'])}`",
                f"- Rescue precision among flips: `{pct(result['rescue_precision_among_flips'])}`",
                f"- Harm-to-gain ratio: `{result['harm_to_gain_ratio'] if result['harm_to_gain_ratio'] is not None else 'N/A'}`",
                "",
            ]
        )
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
