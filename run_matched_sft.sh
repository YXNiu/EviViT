#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 base MODEL TRAIN_10K OUTPUT_DIR [GPU]" >&2
  echo "   or: $0 evivit MODEL TRAIN_10K SELECTOR CONTEXT_NEED BRIDGE H_SAFE OUTPUT_DIR [GPU]" >&2
}

if (( $# < 4 )); then usage; exit 2; fi
lane=$1
model=$2
manifest=$3
case "$lane" in
  base)
    if (( $# < 4 || $# > 5 )); then usage; exit 2; fi
    output_dir=$4
    gpu=${5:-0}
    ;;
  evivit)
    if (( $# < 8 || $# > 9 )); then usage; exit 2; fi
    selector=$4
    context_need=$5
    bridge=$6
    h_safe=$7
    output_dir=$8
    gpu=${9:-0}
    ;;
  *) usage; exit 2 ;;
esac

if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "GPU must be a nonnegative integer" >&2
  exit 2
fi
if [[ ! -d "$model" || ! -f "$manifest" ]]; then
  echo "Missing model directory or 10K training manifest" >&2
  exit 1
fi
if [[ $(wc -l < "$manifest") -ne 10000 ]]; then
  echo "The matched SFT manifest must have exactly 10,000 rows" >&2
  exit 2
fi
if [[ "$lane" == evivit ]]; then
  for required in "$selector" "$context_need" "$bridge" "$h_safe"; do
    if [[ ! -f "$required" ]]; then echo "Missing attachment checkpoint: $required" >&2; exit 1; fi
  done
fi
if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing output: $output_dir" >&2
  exit 1
fi

absolute_path() {
  if [[ "$1" == /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$PWD" "$1"; fi
}
model=$(absolute_path "$model")
manifest=$(absolute_path "$manifest")
if [[ "$lane" == evivit ]]; then
  selector=$(absolute_path "$selector")
  context_need=$(absolute_path "$context_need")
  bridge=$(absolute_path "$bridge")
  h_safe=$(absolute_path "$h_safe")
fi

package_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
data_root=$(absolute_path "${EVIVIT_DATA_ROOT:-$PWD}")
python_bin=${EVIVIT_PYTHON:-python3}
memory_budget_mib=${EVIVIT_SFT_MEMORY_BUDGET_MIB:-30000}
if [[ ! "$memory_budget_mib" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVIVIT_SFT_MEMORY_BUDGET_MIB must be a positive integer" >&2
  exit 2
fi
export PYTHONPATH="$package_root${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$lane" == base ]]; then
  command=(
    "$python_bin" "$package_root/scripts/train_evivit_answer_sft.py"
    --variant base_q16m --recovery-lane B-L --answer-protocol compact_json
    --train-manifest "$manifest" --output-dir "$output_dir"
    --project-root "$data_root" --model "$model" --gpu "$gpu"
    --memory-budget-mib "$memory_budget_mib" --num-epochs 3 --max-train-rows 10000
    --seed 20260911 --learning-rate 1e-6 --weight-decay 0
    --warmup-ratio 0.03 --scheduler-type cosine
    --gradient-accumulation-steps 20 --max-grad-norm 1.0
    --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
    --gradient-checkpointing --save-resume-steps 250
  )
else
  command=(
    "$python_bin" "$package_root/scripts/train_evivit_v3_bridge.py"
    --recovery-lane E-L --train-manifest "$manifest" --model "$model"
    --project-root "$data_root" --selector-checkpoint "$selector"
    --warm-start-bridge-checkpoint "$bridge" --adaptive-box-head "$h_safe"
    --output-dir "$output_dir" --insertion-block 18
    --processor-min-pixels 4096 --processor-max-pixels 16777216
    --global-processor-max-pixels 4194304
    --fine-processor-max-pixels 16777216
    --global-token-budget 2048 --fine-token-budget 3072
    --visual-output-mode append --bridge-mode bidirectional
    --answer-protocol compact_json --max-regions 3 --minimum-region-tokens 64
    --patch-exchange --patch-exchange-global-scale 1.5
    --patch-exchange-local-scale 2.0 --patch-exchange-total-cap-ratio 1.0
    --patch-exchange-soft-floor-tokens 2048
    --patch-exchange-balanced-soft-floor --patch-exchange-preserve-native-global
    --patch-exchange-continuous-soft-floor
    --patch-exchange-continuous-soft-floor-max-tokens 4096
    --patch-exchange-continuous-soft-floor-ramp-start-tokens 4096
    --patch-exchange-continuous-soft-floor-ramp-end-tokens 12288
    --patch-exchange-view-min-pixels 4096 --evidence-decoder trace_split
    --trace-split-budget-mode learned
    --trace-split-context-need-checkpoint "$context_need"
    --trace-split-min-context-fraction 0.10
    --trace-split-max-context-fraction 0.25
    --trace-split-context-expansion-factor 1.0
    --trace-split-context-decoder top_boxes --fusion sparse_bridge
    --bridge-dim 256 --bridge-heads 4 --neighborhood-radius 1
    --max-relative-residual 0.2 --limit 10000 --max-steps 30000
    --gradient-accumulation 20 --gradient-checkpointing
    --language-learning-rate 1e-6 --bridge-learning-rate 5e-6
    --lora-r 8 --lora-alpha 16 --lora-dropout 0.05
    --warmup-ratio 0.03 --weight-decay 0 --identity-weight 0.01
    --max-grad-norm 1.0 --save-steps 5000
    --seed 20260911 --gpu "$gpu" --memory-budget-mib "$memory_budget_mib"
  )
fi

if [[ ${EVIVIT_DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
