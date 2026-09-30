# Training Guide

[← Project overview](../README.md)

The public training workflow targets the Qwen3-VL-4B/8B attachments. It requires host weights, original training images, trace-derived targets, prepared features, and the appropriate component-specific manifests. The [raw human-search dataset](https://huggingface.co/datasets/YXNiu/Human-Search-Traces) supplies event-level annotations, not all precomputed training inputs.

## Component training

Prepare evidence maps and frozen Qwen features with the programs in [`scripts/`](../scripts/), then train the components in order:

| Component | Entry points |
| --- | --- |
| PTEA | `scripts/train_patch_text_evidence_map.py`, `scripts/train_tracesplit_context_ptea.py` |
| ContextNeed | `scripts/train_evisplit_context_need_full1144.py` |
| H-Safe | `scripts/train_evivit_v6_adaptivebox.py` |
| Sparse Bridge | `run_final_train_bridge.sh` |

Each component program accepts `--help` for its input arguments. Refer to the paper's reproducibility appendix for the training schedule and supervision definitions; a complete end-to-end preparation recipe is not implied by the launcher alone.

## Sparse Bridge

Supply trained selector, ContextNeed, and H-Safe checkpoints:

```bash
export EVIVIT_PYTHON=python3
export EVIVIT_DATA_ROOT=../assets
bash run_final_train_bridge.sh 4b \
  ../assets/model4 ../assets/selector4.pt ../assets/context4.pt \
  ../assets/h_safe4.pt ../assets/train1144.jsonl ../work/bridge4 0
```

For Qwen3-VL-8B, use `8b` and the corresponding scale-matched assets. The output directory must not already exist.

## Matched language-side SFT

The comparison uses the same 10,000-example QA manifest and language-LoRA schedule. This QA manifest is not included in the release and is distinct from the 1,144 raw human-search sessions.

```bash
bash run_matched_sft.sh base ../assets/model8 ../assets/train10k.jsonl ../work/base_sft 0
bash run_matched_sft.sh evivit ../assets/model8 ../assets/train10k.jsonl \
  ../assets/selector8.pt ../assets/context8.pt ../assets/bridge8.pt ../assets/h_safe8.pt \
  ../work/evivit_sft 1
```

The launchers validate the 10K row count and refuse to overwrite an existing output directory. For evaluation of the resulting language adapters, see [the evaluation guide](evaluation.md#evaluating-sft-adapters).
