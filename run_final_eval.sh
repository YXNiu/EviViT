#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 4b|8b DATASET MODEL SELECTOR CONTEXT_NEED BRIDGE H_SAFE MANIFEST OUTPUT [GPU]" >&2
  echo "DATASET: visualprobe|vstar|perceptionbench|hr4k|hr8k|mmbench|mmstar|zoombench" >&2
}

if (( $# < 9 || $# > 10 )); then
  usage
  exit 2
fi

scale=$1
dataset=$2
model=$3
selector=$4
context_need=$5
bridge=$6
h_safe=$7
manifest=$8
output=$9
gpu=${10:-0}

case "$scale" in
  4b) insertion_block=16 ;;
  8b) insertion_block=18 ;;
  *) usage; exit 2 ;;
esac

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
if [[ ! -d "$model" ]]; then
  echo "Missing model directory: $model" >&2
  exit 1
fi
for required in "$selector" "$context_need" "$bridge" "$h_safe" "$manifest"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing input file: $required" >&2
    exit 1
  fi
done

absolute_path() {
  if [[ "$1" == /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$PWD" "$1"; fi
}
model=$(absolute_path "$model")
selector=$(absolute_path "$selector")
context_need=$(absolute_path "$context_need")
bridge=$(absolute_path "$bridge")
h_safe=$(absolute_path "$h_safe")
manifest=$(absolute_path "$manifest")

package_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
data_root=$(absolute_path "${EVIVIT_DATA_ROOT:-$PWD}")
python_bin=${EVIVIT_PYTHON:-python3}
export PYTHONPATH="$package_root${PYTHONPATH:+:$PYTHONPATH}"

command=(
  "$python_bin" "$package_root/scripts/eval_evivit_v3_mid_bridge.py"
  --project-root "$data_root" --model "$model" --manifest "$manifest"
  --output "$output" --gpu "$gpu" --insertion-block "$insertion_block"
  --selector-checkpoint "$selector" --bridge-checkpoint "$bridge"
  --adaptive-box-head "$h_safe" --adaptive-box-safety-mode replace
  --evidence-policy mid_ptea --evidence-decoder trace_split
  --trace-split-budget-mode learned
  --trace-split-context-need-checkpoint "$context_need"
  --trace-split-min-context-fraction 0.10
  --trace-split-max-context-fraction 0.25
  --trace-split-context-expansion-factor 1.0
  --trace-split-context-decoder top_boxes
  --global-token-budget 2048 --fine-token-budget 3072
  --minimum-region-tokens 64 --max-regions 3
  --processor-min-pixels "$min_pixels"
  --processor-max-pixels 16777216
  --patch-exchange --patch-exchange-global-scale 1.5
  --patch-exchange-local-scale 2.0 --patch-exchange-total-cap-ratio 1.0
  --patch-exchange-soft-floor-tokens 2048
  --patch-exchange-balanced-soft-floor
  --patch-exchange-preserve-native-global
  --patch-exchange-continuous-soft-floor
  --patch-exchange-continuous-soft-floor-max-tokens 4096
  --patch-exchange-continuous-soft-floor-ramp-start-tokens 4096
  --patch-exchange-continuous-soft-floor-ramp-end-tokens 12288
  --patch-exchange-view-min-pixels 4096
  --answer-protocol compact_json --memory-budget-mib 0
  --max-new-tokens "$max_new_tokens"
  --deterministic-inference --deterministic-attention-backend flash
  --record-timing --resume
)

if [[ -n ${EVIVIT_LANGUAGE_ADAPTER:-} ]]; then
  if [[ ! -d "$EVIVIT_LANGUAGE_ADAPTER" ]]; then
    echo "Missing language-LoRA adapter directory: $EVIVIT_LANGUAGE_ADAPTER" >&2
    exit 1
  fi
  command+=(--language-lora-adapter "$(absolute_path "$EVIVIT_LANGUAGE_ADAPTER")")
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
