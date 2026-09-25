#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 DATASET MODEL MANIFEST OUTPUT [GPU]" >&2
  echo "DATASET: visualprobe|vstar|perceptionbench|hr4k|hr8k|mmbench|mmstar|zoombench" >&2
  echo "Optional: set EVIVIT_BASE_ADAPTER to evaluate a language-LoRA baseline." >&2
}

if (( $# < 4 || $# > 5 )); then usage; exit 2; fi
dataset=$1
model=$2
manifest=$3
output=$4
gpu=${5:-0}

case "$dataset" in
  visualprobe|vstar|perceptionbench|hr4k|hr8k)
    min_pixels=4096
    max_new_tokens=64
    ;;
  mmbench|mmstar)
    min_pixels=4096
    max_new_tokens=32
    ;;
  zoombench)
    min_pixels=65536
    max_new_tokens=64
    ;;
  *) usage; exit 2 ;;
esac
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "GPU must be a nonnegative integer" >&2
  exit 2
fi
if [[ ! -d "$model" || ! -f "$manifest" ]]; then
  echo "Missing model directory or manifest" >&2
  exit 1
fi

absolute_path() {
  if [[ "$1" == /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$PWD" "$1"; fi
}
model=$(absolute_path "$model")
manifest=$(absolute_path "$manifest")
package_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
data_root=$(absolute_path "${EVIVIT_DATA_ROOT:-$PWD}")
python_bin=${EVIVIT_PYTHON:-python3}
export PYTHONPATH="$package_root${PYTHONPATH:+:$PYTHONPATH}"

command=(
  "$python_bin" "$package_root/scripts/eval_qwen_original_qa.py"
  --project-root "$data_root" --model "$model" --manifest "$manifest"
  --output "$output" --gpu "$gpu" --memory-budget-mib 0
  --min-pixels "$min_pixels" --max-pixels 16777216
  --max-new-tokens "$max_new_tokens" --prompt-variant canonical
  --deterministic-inference --deterministic-attention-backend flash
  --record-timing --resume
)
if [[ -n ${EVIVIT_BASE_ADAPTER:-} ]]; then
  if [[ ! -d "$EVIVIT_BASE_ADAPTER" ]]; then
    echo "Missing adapter directory: $EVIVIT_BASE_ADAPTER" >&2
    exit 1
  fi
  command+=(--adapter "$(absolute_path "$EVIVIT_BASE_ADAPTER")")
fi
if [[ -n ${EVIVIT_LIMIT:-} ]]; then
  if [[ ! "$EVIVIT_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVIVIT_LIMIT must be a positive integer" >&2
    exit 2
  fi
  command+=(--limit "$EVIVIT_LIMIT")
fi

if [[ ${EVIVIT_DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
