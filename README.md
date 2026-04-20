# 🩺 MedLLM-Finetuning

基于 **Hugging Face Transformers**、**PEFT** 与 **DeepSpeed** 的医疗文本二分类微调框架。

本项目提供了一套开箱即用的微调流程，旨在降低医疗大语言模型（LLM）的训练门槛，提供高效、灵活的训练体验。

### ✨ 核心特性
* **多模式支持：** 兼容 QLoRA（4bit + LoRA）、LoRA（bf16 全精度基座）与全量微调。
* **硬件友好：** 深度集成 DeepSpeed，支持多卡 ZeRO 优化，缓解 OOM 焦虑。
* **配置驱动：** 将繁琐的超参抽离为 YAML 配置，代码与参数解耦，易于复现。


---

## 📂 数据准备

默认数据集配置见 [`configs/datasets/medical_binary.yaml`](configs/datasets/medical_binary.yaml)：

- **训练集路径**：`data/raw/medical_train.json`
- **测试集路径**：`data/raw/medical_test.json`
- **验证集路径**：未指定独立验证集时，从训练集按 **`validation_split`** 划分验证集

> **数据格式要求：** JSON 字段由 `configs/tasks/binary_classification.yaml` 定义。默认将文本列设为 `input`，标签列设为 `output`（值为 `0` 或 `1`）。

**数据集示例**（每行一个 JSON 对象；`input` 为完整提示与患者描述文本，`output` 为二分类标签字符串）：

```json
{
  "input": "You are a cardiovascular specialist and predict whether the patient will develop coronary heart disease within the next 10 years. The output should be a binary value (1 = Yes, 0 = No). The patient is a...",
  "output": "0"
}
```

---
## 🤖 支持的大模型示例

在训练脚本中，通过 `--model-name-or-path` 传入 Hugging Face Hub ID 或本地路径。部分医学模型或自定义权重需追加 `--trust-remote-code true`。

| 模型类型 | 模型名称 | Hub ID |
| :--- | :--- | :--- |
| **医用大模型** | MedAlpaca (7B) | `medalpaca/medalpaca-7b` |
| | Meditron (7B) | `epfl-llm/meditron-7b` |
| | Med42 (8B) | `m42-health/Llama3-Med42-8B` |
| **通用大模型** | Qwen2 (7B) | `Qwen/Qwen2-7B-Instruct` |
| | Mistral (7B) | `mistralai/Mistral-7B-Instruct-v0.3` |
| | Llama 2 (7B) | `meta-llama/Llama-2-7b-chat-hf` |
| | Llama 3 (8B) | `meta-llama/Meta-Llama-3-8B-Instruct` |
| | Gemma 2 (9B) | `google/gemma-2-9b-it` |
| **小参数量编码器**| BioBERT (约1B) | `dmis-lab/biobert-v1.1` |

*注：对于受保护的模型（Gated Models），请确保已在 Hugging Face 接受许可，并在环境中配置了 `HF_TOKEN`。BioBERT 等编码器架构只支持lora和全量微调。*

---

## 🛠️ 环境要求与安装

**前置依赖：** Python ≥ 3.10，Linux 操作系统，NVIDIA GPU（驱动与 CUDA 需与 PyTorch 匹配）。

```bash
git clone https://github.com/Jackohhhh/MedLLM-Finetuning.git
cd MedLLM-Finetuning

# 使用可编辑模式安装（推荐开发环境使用）
pip install -e .

# 可选：安装 FlashAttention-2 以加速训练（需严格匹配 torch/CUDA 版本）
pip install -e ".[flash]"
```

> 依赖声明见 [`pyproject.toml`](pyproject.toml)。

---
## 🚀 训练指南

在项目根目录执行以下脚本。若需**多卡训练**，请在脚本中将 `deepspeed --num_gpus=1` 修改为实际的卡数 `N`。

* **QLoRA（4bit 量化）** 
  显存占用最低。默认使用 `--load-in-4bit true`、`--bf16` 及 `paged_adamw_32bit` 优化器。
  ```bash
  bash scripts/train/finetune_qlora.sh
  ```

* **LoRA（bf16 全精度基座）**
  无量化损耗，但显存需求显著高于 QLoRA。
  ```bash
  bash scripts/train/finetune_lora.sh
  ```

* **全量微调 (Full Fine-tuning)**
  更新模型全部参数，需要充足的算力资源。
  ```bash
  bash scripts/train/finetune_full.sh
  ```

---

## 📊 模型测试

评估微调后的模型表现。请务必将 `--checkpoint` 参数替换为实际完整的检查点路径。

```bash
bash scripts/eval/eval_binary_cls.sh --checkpoint outputs/checkpoints/Qwen_Qwen2-7B-Instruct_qlora/checkpoint-1000
```

---

## ⚙️ 配置系统说明

本项目采用结构化的 YAML / JSON 配置文件管理超参。如果需要修改 LoRA 目标层、序列长度等，可直接编辑对应文件：

| 配置文件类型 | 路径与作用 |
| :--- | :--- |
| **实验入口** | `configs/experiments/medical.yaml`：统合各个模块的路径（task、dataset、deepspeed），为训练主干。 |
| **模型结构** | `configs/models/seq_cls.yaml`：定义骨架、LoRA 结构与默认目标层。可通过 CLI 覆盖。 |
| **数据与权重** | `configs/datasets/*.yaml`：定义数据路径、划分比例、正负样本权重 (`class_weights`)。 |
| **任务属性** | `configs/tasks/*.yaml`：定义文本列名、`max_length`、类别总数。 |
| **加速策略** | `configs/deepspeed/*.json`：配置 DeepSpeed 的 ZeRO 阶段等参数。 |

---

## 📁 输出文件结构

* **模型权重：** `outputs/checkpoints/<model_slug>_<模式>/` 下（模式包含 full、lora、qlora 等）。
* **预测结果：** 训练后测试集的预测结果位于 `outputs/predictions/<model_slug>_<模式>/` 下的 `metrics.json` 和 `predictions.jsonl`。
* **训练日志：** 若启用了 `--report-to wandb`，请在 W&B 控制台查看曲线；本地日志保存在 `wandb/` 目录下。

---
## 💻 硬件需求（实测）

在集成 **DeepSpeed** 与 **QLoRA** 的前提下，可在**消费级 NVIDIA GPU** 上对常见 7B / 8B 级模型做微调实验。下表为**经验范围**，实际占用随序列长度、批量、ZeRO 阶段等变化；若 OOM，见下文「常见问题」。

| 模型规模 | 精度 / 策略 | 显存（约） | 适合 GPU（参考） |
| :--- | :--- | :--- | :--- |
| 7B / 8B（如 Llama-3） | QLoRA（4bit） | ~10 GB 起 | RTX 3060、4060 等 |
| 7B / 8B（如 Qwen2） | LoRA（bf16 基座） | ~32 GB 起 | RTX 4080、3090 等 |
| 7B / 8B | 全量微调 | ~100 GB+ | A100 80GB、多卡 ZeRO 等 |
---

## ❓ 常见问题 (FAQ)

**Q：环境 OOM（显存不足）怎么办？**
优先尝试减小 `--per-device-train-batch-size` 并增大 `--gradient-accumulation-steps`；或者切换到 QLoRA 模式、使用多卡 ZeRO-3、确保已开启 `--gradient-checkpointing`。

**Q：未设置 `HF_TOKEN` 导致下载很慢或报错 429？**
请前往 Hugging Face Settings 创建 Token，并在终端执行 `export HF_TOKEN="你的token"`，或使用 `huggingface-cli login` 进行授权。

**Q：Transformers 升级大版本后参数报错？**
本仓库已对 `eval_strategy` 等 v5 差异做了兼容。若仍有 `TrainingArguments` 报错，请对照 Transformers 最新发行说明，或通过 `pyproject.toml` 锁定版本。

**Q：`load_best_model_at_end` 保存的是最后一步的权重吗？**
不一定。开启后，系统会保存验证集上指定指标（如 F1）最优步骤的权重。若需强制使用某一步骤，测试时请明确指向 `checkpoint-<step>` 目录。

---

## 📝 更新日志

| 版本日期 | 核心更新说明 |
| :--- | :--- |
| 2026-04-17 | 基于 DeepSpeed 完成 Qwen2 的 LoRA、QLoRA 与全量微调的工程实现及测试。 |
| 2026-04-18 | 已对齐 Hugging Face 常见用法，适配绝大多数通用与医疗大模型（Hub ID 或本地权重）。 |


---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

致谢：构建于 [Transformers](https://github.com/huggingface/transformers)、[PEFT](https://github.com/huggingface/peft) 与 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 等优秀开源项目之上。