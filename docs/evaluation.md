# Evaluation Guide

[← Project overview](../README.md)

The public launchers support Qwen3-VL-4B and Qwen3-VL-8B. They require external model weights, scale-matched EviViT attachment checkpoints, and benchmark assets. Attachment checkpoints are not included in this release.

## Environment and assets

Use Python 3 with CUDA-enabled PyTorch, NumPy, Pillow, Qwen3-VL-compatible Transformers, Safetensors, and PEFT. The launchers select PyTorch's **flash SDPA** backend (`--deterministic-attention-backend flash`) under deterministic algorithms; your GPU/PyTorch environment must support it. This flag selects the PyTorch backend, not a separate `flash-attn` installation command. Run `python3 verify_release.py` for the offline source and result checks.

The examples use this layout; filenames are illustrative paths to assets you supply, not download endpoints:

```text
EviViT/                 # this repository
assets/
  model4/               # Qwen3-VL-4B host
  model8/               # Qwen3-VL-8B host, if needed
  judge8/               # local Qwen3-VL-8B answer judge
  selector4.pt          # trained PTEA selector
  context4.pt           # trained ContextNeed
  bridge4.pt            # trained Sparse Bridge
  h_safe4.pt            # trained H-Safe
  visualprobe.jsonl     # evaluation manifest
  ...                   # images and scale-matched 8B checkpoints
work/                   # output predictions and judge caches
```

Each evaluation row must include `id`, `image`, `question`, and `answer`. `EVIVIT_DATA_ROOT` resolves relative image paths in these manifests. Benchmark-specific preparation is separate from loading the raw human-search annotations.

## EviViT, baseline, and local judging

Run from the repository root:

```bash
export EVIVIT_PYTHON=python3
export EVIVIT_DATA_ROOT=../assets

bash run_final_eval.sh 4b visualprobe \
  ../assets/model4 ../assets/selector4.pt ../assets/context4.pt \
  ../assets/bridge4.pt ../assets/h_safe4.pt \
  ../assets/visualprobe.jsonl ../work/visualprobe_evivit.jsonl 0

bash run_final_baseline.sh visualprobe \
  ../assets/model4 ../assets/visualprobe.jsonl ../work/visualprobe_base.jsonl 0

bash run_final_judge.sh main ../assets/judge8 \
  ../work/visualprobe_base.jsonl ../work/visualprobe_evivit.jsonl \
  ../work/visualprobe_base_judged.jsonl ../work/visualprobe_evivit_judged.jsonl \
  ../work/judge_cache_main.jsonl 0
```

The final positional argument selects the GPU. For 8B, use `8b` and the corresponding host, selector, ContextNeed, Bridge, and H-Safe checkpoints.

| Dataset key | Evaluation target |
| --- | --- |
| `visualprobe` | VisualProbe |
| `vstar` | V*Bench |
| `perceptionbench` | PerceptionBench |
| `hr4k`, `hr8k` | HR-Bench 4K / 8K |
| `mmbench`, `mmstar` | MMBench / MMStar |
| `zoombench` | ZoomBench using full images |

Use judge mode `main` for the base/EviViT comparison and `sft` for SFT predictions. Use separate prediction outputs and judge caches for different experiments.

## Small checks before a full run

- `EVIVIT_DRY_RUN=1` prints the assembled command **after checking required input paths**. It still requires the supplied model directory, checkpoints, and manifest to exist.
- `EVIVIT_LIMIT=1` limits inference to one question; it still loads the real models and checkpoints.
- `python3 verify_release.py` is an asset-free source/result check, not a GPU or accuracy test.

## Evaluating SFT adapters

Set `EVIVIT_BASE_ADAPTER` to a `base_sft/checkpoint-epoch-*` directory for the baseline, or `EVIVIT_LANGUAGE_ADAPTER` to an `evivit_sft/checkpoint_step_*/language_adapter` directory for EviViT. Then rerun the respective launchers with new output paths and use judge mode `sft` with separate caches.

## Reported results

[`results/`](../results/) includes paired fine-grained scores, seven-benchmark means, budget frontiers, efficiency measurements, SFT comparisons, and training curves. These summaries do not replace per-question benchmark manifests or checkpoints. The paper specifies the evaluation protocol and distinguishes local pairs from author-reported/API references.
