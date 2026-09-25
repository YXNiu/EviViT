#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 4b|8b MODEL SELECTOR CONTEXT_NEED H_SAFE TRAIN_MANIFEST OUTPUT_DIR [GPU]" >&2
  echo "Both scales train the final Sparse Bridge from its identity initialization." >&2
}

if (( $# < 7 || $# > 8 )); then
  usage
  exit 2
fi
scale=$1
model=$2
selector=$3
context_need=$4
h_safe=$5
manifest=$6
output_dir=$7
gpu=${8:-0}

case "$scale" in
  4b) insertion_block=16 ;;
  8b) insertion_block=18 ;;
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
for required in "$selector" "$context_need" "$h_safe" "$manifest"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing input file: $required" >&2
    exit 1
  fi
done
if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing output: $output_dir" >&2
  exit 1
fi

absolute_path() {
  if [[ "$1" == /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$PWD" "$1"; fi
}
model=$(absolute_path "$model")
selector=$(absolute_path "$selector")
context_need=$(absolute_path "$context_need")
h_safe=$(absolute_path "$h_safe")
manifest=$(absolute_path "$manifest")

package_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
data_root=$(absolute_path "${EVIVIT_DATA_ROOT:-$PWD}")
python_bin=${EVIVIT_PYTHON:-python3}
export PYTHONPATH="$package_root${PYTHONPATH:+:$PYTHONPATH}"

command=(
  "$python_bin" "$package_root/scripts/train_evivit_v3_bridge.py"
  --project-root "$data_root" --train-manifest "$manifest"
  --model "$model" --selector-checkpoint "$selector"
  --trace-split-context-need-checkpoint "$context_need"
  --adaptive-box-head "$h_safe" --output-dir "$output_dir"
  --insertion-block "$insertion_block" --fusion sparse_bridge
  --evidence-decoder trace_split --trace-split-budget-mode learned
  --trace-split-context-decoder top_boxes
  --trace-split-min-context-fraction 0.10
  --trace-split-max-context-fraction 0.25
  --trace-split-context-expansion-factor 1.0
  --global-token-budget 2048 --fine-token-budget 3072
  --minimum-region-tokens 64 --max-regions 3
  --processor-min-pixels 4096 --processor-max-pixels 16777216
  --patch-exchange --patch-exchange-global-scale 1.5
  --patch-exchange-local-scale 2.0 --patch-exchange-total-cap-ratio 1.0
  --patch-exchange-soft-floor-tokens 2048
  --patch-exchange-balanced-soft-floor --patch-exchange-preserve-native-global
  --patch-exchange-continuous-soft-floor
  --patch-exchange-continuous-soft-floor-max-tokens 4096
  --patch-exchange-continuous-soft-floor-ramp-start-tokens 4096
  --patch-exchange-continuous-soft-floor-ramp-end-tokens 12288
  --patch-exchange-view-min-pixels 4096
  --answer-protocol compact_json --limit 1144
  --max-steps 3432 --save-steps 1144
  --gradient-accumulation 4 --gradient-checkpointing
  --learning-rate 1e-4 --weight-decay 0.01
  --identity-weight 0.01 --max-grad-norm 1.0
  --seed 20260819
  --gpu "$gpu" --memory-budget-mib 0
)

if [[ ${EVIVIT_DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
