# cad-finetune

基于 **Hugging Face Transformers**、**PEFT（LoRA / QLoRA）** 与 **DeepSpeed** 的 **CAD / 医疗文本二分类** 微调脚手架：支持加权损失、数据集过采样、命令行覆盖超参，以及离线评估与预测落盘。

---

## 功能概览

| 能力 | 说明 |
|------|------|
| **QLoRA / LoRA** | 4bit 量化 + LoRA，分类头与 `modules_to_save` 可配置 |
| **全量微调** | 独立实验配置与脚本（无量化、无 LoRA） |
| **DeepSpeed ZeRO2** | 默认 `configs/deepspeed/zero2.json`，可与 HF `TrainingArguments` 对齐 |
| **加权 CE** | `WeightedTrainer` + 数据集 YAML 中的 `class_weights` |
| **评估** | `cli.eval`：加载 checkpoint，对测试集 `predict`，输出 `metrics.json` / `predictions.jsonl` |
| **Flash Attention（可选）** | `pip install -e ".[flash]"`，训练脚本中加 `--attn-implementation flash_attention_2` |

---

## 环境要求

- **Python ≥ 3.10**
- **Linux + NVIDIA GPU**（推荐；QLoRA / DeepSpeed 依赖 CUDA）
- **PyTorch** 需与显卡驱动、CUDA 版本匹配（由你本机安装）

---

## 安装

```bash
git clone <你的仓库 URL>.git
cd <仓库目录>

# 可编辑安装（推荐开发）
pip install -e .

# 可选：FlashAttention-2（与 torch/CUDA 版本强相关）
pip install -e ".[flash]"
```

依赖声明见 [`pyproject.toml`](pyproject.toml)。**不再维护根目录 `requirements.txt`**，请以 `pyproject.toml` 为准。

---

## 数据准备

默认数据集配置见 [`configs/datasets/medical_binary.yaml`](configs/datasets/medical_binary.yaml)：

- **训练**：`data/raw/medical_train.json`
- **测试**：`data/raw/medical_test.json`
- 未指定独立验证集时，从训练集按 **`validation_split`** 划分验证集

JSON 字段由 [`configs/tasks/binary_classification.yaml`](configs/tasks/binary_classification.yaml) 定义（默认文本列 **`input`**，标签列 **`output`**，值为 `0` / `1`）。

---

## 快速开始

在项目根目录执行（脚本内已 `cd` 到仓库根并设置 `PYTHONPATH`）。

### QLoRA 训练

```bash
bash scripts/train/finetune_lora.sh
```

多卡时请在 `finetune_lora.sh` 里把 `deepspeed --num_gpus=1` 改成所需卡数。附加 CLI 示例：

```bash
bash scripts/train/finetune_lora.sh --skip-test
```

超参写在 `finetune_lora.sh` 的 `deepspeed ... cad_finetune.cli.train \` 一段中；与 [`src/cad_finetune/cli/overrides.py`](src/cad_finetune/cli/overrides.py) 中的 CLI 参数对应，会覆盖 experiment YAML 合并结果。

### 全量微调

```bash
bash scripts/train/finetune_full.sh
```

### 离线评估

```bash
bash scripts/eval/eval_binary_cls.sh
```

请编辑脚本中的 **`--checkpoint`**（指向含 `adapter_config.json` 的 LoRA 目录，或全量权重的 HF 格式目录）。`eval` 与 `train` 共用同一套 **override** 参数（如 `--model-name-or-path`、`--test-file`、`--per-device-eval-batch-size` 等）。

### 仅调用 Python 模块

```bash
export PYTHONPATH=src

python -m cad_finetune.cli.train --config configs/experiments/qwen2_medical_lora.yaml ...
python -m cad_finetune.cli.eval --config configs/experiments/qwen2_medical_lora.yaml --checkpoint <路径> ...
```

---

## 配置说明

| 路径 | 作用 |
|------|------|
| `configs/experiments/*.yaml` | 实验入口：`paths` 引用 model / task / dataset / deepspeed；顶层可写 `output_dir` 等 |
| `configs/models/*.yaml` | 模型 ID、量化、LoRA 结构、`target_modules` 等（**LoRA 开启时必须配置 `target_modules`**） |
| `configs/datasets/*.yaml` | 数据路径、划分、过采样、`class_weights` |
| `configs/tasks/*.yaml` | 列名、`max_length`、类别数 |
| `configs/deepspeed/*.json` | DeepSpeed 策略 |

**训练超参（学习率、batch、scheduler、logging、LoRA 开关等）** 默认通过 **Shell + CLI** 传入，不再使用 `configs/training/`。

---

## 仓库结构（精简）

```text
configs/
  datasets/          # 数据与采样、类别权重
  deepspeed/         # ZeRO 等
  experiments/       # 实验组合（paths）
  models/            # 单模型配置
  tasks/             # 任务字段与长度
scripts/
  train/             # finetune_lora.sh, finetune_full.sh
  eval/              # eval_binary_cls.sh
src/cad_finetune/
  cli/               # train / eval / overrides
  models/            # factory, 可选自定义序列分类封装
  tasks/classification/
  train/             # runner, WeightedTrainer
  data/              # collator
```

更细的目录说明可参考 [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md)（部分内容可能滞后于当前仓库，以实际文件为准）。

---

## 输出产物

| 类型 | 位置（默认，可改脚本参数） |
|------|----------------------------|
| 检查点 / 适配器 | `--output-dir` |
| 训练后测试集预测 | `--prediction-output-dir` 下的 `metrics.json`、`predictions.jsonl` |
| 日志 | `--logging-dir`；若 `--report-to wandb` 需先 `wandb login` |

---

## 常见问题

**模型从哪里下载？**  
`model_name_or_path` 为 Hub ID（如 `Qwen/Qwen2-7B-Instruct`）时，首次运行会缓存到本机 Hugging Face 默认目录（或你设置的 `HF_HOME` / `cache_dir`）。

**QLoRA 与 Flash Attention 同时报错？**  
可先去掉 `--attn-implementation flash_attention_2`，或改用 `--attn-implementation sdpa`，再查 torch / flash-attn / CUDA 版本兼容性。

---

## 许可证

若准备开源，请在本仓库根目录添加 **`LICENSE`** 文件（如 MIT、Apache-2.0），并在本段替换为实际许可证名称与全文链接。

---

## 致谢

构建于 [Transformers](https://github.com/huggingface/transformers)、[PEFT](https://github.com/huggingface/peft)、[DeepSpeed](https://github.com/microsoft/DeepSpeed) 等开源项目之上。
