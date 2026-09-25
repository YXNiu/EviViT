#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 main|sft JUDGE_MODEL BASE_PRED EViViT_PRED BASE_OUT EViViT_OUT CACHE [GPU]" >&2
}

if (( $# < 7 || $# > 8 )); then usage; exit 2; fi
protocol=$1
model=$2
base_predictions=$3
evivit_predictions=$4
base_output=$5
evivit_output=$6
cache=$7
gpu=${8:-0}
python_bin=${EVIVIT_PYTHON:-python3}

canonical_output() {
  "$python_bin" -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"
}
base_output=$(canonical_output "$base_output")
evivit_output=$(canonical_output "$evivit_output")
cache=$(canonical_output "$cache")

case "$protocol" in
  main) max_input_tokens=1024; max_new_tokens=8 ;;
  sft) max_input_tokens=8192; max_new_tokens=16 ;;
  *) usage; exit 2 ;;
esac
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "GPU must be a nonnegative integer" >&2
  exit 2
fi
if [[ ! -d "$model" || ! -f "$base_predictions" || ! -f "$evivit_predictions" ]]; then
  echo "Missing judge model or prediction file" >&2
  exit 1
fi
if [[ "$base_output" == "$evivit_output" || "$base_output" == "$cache" || "$evivit_output" == "$cache" ]]; then
  echo "Judge outputs and cache must be three distinct paths" >&2
  exit 2
fi

absolute_path() {
  if [[ "$1" == /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$PWD" "$1"; fi
}
model=$(absolute_path "$model")
base_predictions=$(absolute_path "$base_predictions")
evivit_predictions=$(absolute_path "$evivit_predictions")
package_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$package_root${PYTHONPATH:+:$PYTHONPATH}"

command=(
  "$python_bin" "$package_root/scripts/judge_qa_predictions_local_qwen.py"
  --model "$model" --gpu "$gpu" --memory-budget-mib 0
  --predictions "$base_predictions" --output "$base_output"
  --predictions "$evivit_predictions" --output "$evivit_output"
  --cache "$cache" --max-input-tokens "$max_input_tokens"
  --max-new-tokens "$max_new_tokens"
  --deterministic-inference --deterministic-attention-backend flash --resume
)

if [[ ${EVIVIT_DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

# The shared judge's cache key and resume check do not encode token limits.
# A small sidecar prevents accidental reuse across the main and SFT protocols.
protocol_stamp="${protocol}:${max_input_tokens}:${max_new_tokens}"
check_protocol_stamp() {
  local target=$1 marker="${1}.evivit-judge-protocol"
  if [[ -e "$target" && ! -f "$marker" ]]; then
    echo "Existing judge file has no protocol marker: $target; choose a fresh path" >&2
    exit 2
  fi
  if [[ -f "$marker" && "$(<"$marker")" != "$protocol_stamp" ]]; then
    echo "Judge protocol differs from existing file: $target; choose a fresh path" >&2
    exit 2
  fi
}
for target in "$base_output" "$evivit_output" "$cache"; do
  check_protocol_stamp "$target"
done
for target in "$base_output" "$evivit_output" "$cache"; do
  marker="${target}.evivit-judge-protocol"
  if [[ ! -f "$marker" ]]; then
    mkdir -p "$(dirname "$marker")"
    printf '%s\n' "$protocol_stamp" > "$marker"
  fi
done
exec "${command[@]}"
