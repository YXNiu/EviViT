# EviViT

本项目提供 EviViT 在 Qwen3-VL-4B/8B 上的训练与评测代码，以及论文报告的结果汇总。包内不含模型权重、基准图像、人工搜索轨迹和 QA 训练数据；数据后续单独开源。

## 环境与输入

使用 Python 3、支持 CUDA 的 PyTorch、NumPy、Pillow、兼容 Qwen3-VL 的 Transformers、Safetensors 和 PEFT。自行提供 Qwen3-VL 宿主模型和本地评判模型、与模型规模匹配的 EviViT 检查点，以及基准测试 JSONL 文件。每条评测记录需要 `id`、`image`、`question` 和 `answer`。下方示例将输入放在 `../assets/`、输出放在 `../work/`；`EVIVIT_DATA_ROOT` 用于解析 JSONL 中的相对图像路径。

## 评测

在解压后的代码目录中运行：

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

评测 8B 时，将 `4b` 换成 `8b`，并使用对应的 8B 检查点。支持的数据集参数为 `visualprobe`、`vstar`、`perceptionbench`、`hr4k`、`hr8k`、`mmbench`、`mmstar` 和 `zoombench`；ZoomBench 使用完整图像。设置 `EVIVIT_DRY_RUN=1` 可预览命令，设置 `EVIVIT_LIMIT=1` 可进行单题测试。Base/EviViT 对比使用评判模式 `main`，SFT 预测结果使用 `sft`；两种模式分别使用不同的输出与缓存路径。

## 训练

先用 `scripts/` 中的程序准备证据图和冻结的 Qwen 特征。使用 `train_patch_text_evidence_map.py` 和 `train_tracesplit_context_ptea.py` 训练 PTEA，使用 `train_evisplit_context_need_full1144.py` 训练 ContextNeed，使用 `train_evivit_v6_adaptivebox.py` 训练 H-Safe，最后训练 Sparse Bridge。各组件脚本均可通过 `--help` 查看参数。Bridge 训练入口接收已训练的选择器、ContextNeed 和 H-Safe 检查点：

```bash
bash run_final_train_bridge.sh 4b \
  ../assets/model4 ../assets/selector4.pt ../assets/context4.pt \
  ../assets/h_safe4.pt ../assets/train1144.jsonl ../work/bridge4 0
```

配对 10K SFT 对比的运行方式：

```bash
bash run_matched_sft.sh base ../assets/model8 ../assets/train10k.jsonl ../work/base_sft 0
bash run_matched_sft.sh evivit ../assets/model8 ../assets/train10k.jsonl \
  ../assets/selector8.pt ../assets/context8.pt ../assets/bridge8.pt ../assets/h_safe8.pt \
  ../work/evivit_sft 1
```

评测 SFT 模型时，将 `EVIVIT_BASE_ADAPTER` 指向 `base_sft/checkpoint-epoch-*` 目录，或将 `EVIVIT_LANGUAGE_ADAPTER` 指向 `evivit_sft/checkpoint_step_*/language_adapter` 目录，再用新的输出路径运行评测。`results/` 包含论文中的表格、图表数据和 SFT 训练曲线。
