# EviViT

Code for the EviViT training and evaluation experiments with Qwen3-VL-4B/8B. This package includes the implementation and reported result summaries. Model weights, benchmark images, human-search traces, and QA training data are not included; the data will be released separately.

## Setup

Use Python 3 with CUDA-enabled PyTorch, NumPy, Pillow, Qwen3-VL-compatible Transformers, Safetensors, and PEFT. Supply the Qwen3-VL host and local judge models, scale-matched EviViT checkpoints, and benchmark JSONL files. Each evaluation row needs `id`, `image`, `question`, and `answer`. The examples below keep inputs in `../assets/` and outputs in `../work/`; `EVIVIT_DATA_ROOT` resolves relative image paths inside the JSONL files.

## Evaluate

From the unpacked code directory:

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

For 8B, use `8b` and the corresponding 8B checkpoints. Supported dataset keys are `visualprobe`, `vstar`, `perceptionbench`, `hr4k`, `hr8k`, `mmbench`, `mmstar`, and `zoombench`; ZoomBench uses full images. Set `EVIVIT_DRY_RUN=1` to inspect a command or `EVIVIT_LIMIT=1` for a one-question test. Use judge mode `main` for the base/EviViT comparison and `sft` for SFT predictions, with separate outputs and caches.

## Train

Prepare evidence maps and frozen Qwen features with the programs in `scripts/`. Fit PTEA with `train_patch_text_evidence_map.py` and `train_tracesplit_context_ptea.py`, ContextNeed with `train_evisplit_context_need_full1144.py`, and H-Safe with `train_evivit_v6_adaptivebox.py`, then train the Sparse Bridge. Each component script accepts `--help`. The Bridge launcher takes the trained selector, ContextNeed, and H-Safe checkpoints:

```bash
bash run_final_train_bridge.sh 4b \
  ../assets/model4 ../assets/selector4.pt ../assets/context4.pt \
  ../assets/h_safe4.pt ../assets/train1144.jsonl ../work/bridge4 0
```

For the matched 10K SFT comparison:

```bash
bash run_matched_sft.sh base ../assets/model8 ../assets/train10k.jsonl ../work/base_sft 0
bash run_matched_sft.sh evivit ../assets/model8 ../assets/train10k.jsonl \
  ../assets/selector8.pt ../assets/context8.pt ../assets/bridge8.pt ../assets/h_safe8.pt \
  ../work/evivit_sft 1
```

For SFT evaluation, set `EVIVIT_BASE_ADAPTER` to a `base_sft/checkpoint-epoch-*` directory or `EVIVIT_LANGUAGE_ADAPTER` to an `evivit_sft/checkpoint_step_*/language_adapter` directory, then rerun evaluation with new output paths. `results/` contains the reported table and figure data and the SFT training curves.
