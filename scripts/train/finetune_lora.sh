#!/usr/bin/env bash
#
# 在下面这一条 deepspeed 命令里直接改参数即可（与 LLaVA 脚本一样「摊开」写）。
# 仍需要 --config 指向 experiment YAML，作为 oversampling / class_weights / LoRA target_modules 等骨架。
# 本命令里出现的 CLI 参数会覆盖对应 YAML。
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

################## 按需改行内参数 ##################
# 多卡：把下一行改成 --num_gpus=2（或 4、8）
# 验证集：不设 --validation-file 时，按 configs/datasets/*.yaml 里 validation_split 从训练集划分（当前为 0.2）
# 优化器：例如 --optim paged_adamw_32bit（QLoRA 常用）或 adamw_torch
# tokenizer：不写 --tokenizer-name-or-path 时用模型 YAML；未配置 tokenizer 时自动用 model_name_or_path
# Flash Attention：pip install -e ".[flash]"；与 4bit 冲突时可去掉 --attn-implementation 或改成 sdpa
# 精度：--fp16 / --bf16 二选一（与 train.py 一致）
# 换模型：改 --model-name-or-path（必要时再加 --tokenizer-name-or-path）
# --logging-steps：只影响 HF/Trainer 与 wandb 记录间隔，不必同步改 zero2.json；改 DeepSpeed 终端刷屏频率请改 zero2 里 steps_per_print
# 训练落盘：--output-dir 下有 checkpoint、trainer_state、all_results.json 等；wandb 看网页；--logging-dir 为 TensorBoard 类日志目录
# 训完测集：--prediction-output-dir 下写 metrics.json / predictions.jsonl
################## 按需改行内参数 ##################

deepspeed --num_gpus=1 --module cad_finetune.cli.train \
  --config configs/experiments/qwen2_medical_lora.yaml \
  --experiment-name qwen2_medical_lora \
  --model-name-or-path Qwen/Qwen2-7B-Instruct \
  --train-file data/raw/medical_train.json \
  --test-file data/raw/medical_test.json \
  --output-dir outputs/checkpoints/qwen2_medical_lora \
  --prediction-output-dir outputs/predictions/qwen2_medical_lora \
  --logging-dir outputs/logs/qwen2_medical_lora \
  --launcher deepspeed \
  --seed 42 \
  --gradient-checkpointing true \
  --dataloader-num-workers 4 \
  --report-to wandb \
  --deepspeed configs/deepspeed/zero2.json \
  --num-train-epochs 1 \
  --per-device-train-batch-size 8 \
  --per-device-eval-batch-size 16 \
  --gradient-accumulation-steps 4 \
  --learning-rate 3e-5 \
  --weight-decay 0.0 \
  --warmup-ratio 0.1 \
  --lr-scheduler-type cosine \
  --logging-steps 10 \
  --logging-first-step true \
  --log-level info \
  --optim paged_adamw_32bit \
  --save-strategy steps \
  --save-steps 200 \
  --evaluation-strategy steps \
  --eval-steps 200 \
  --save-total-limit 1 \
  --max-steps -1 \
  --max-length 1024 \
  --input-column input \
  --label-column output \
  --num-labels 2 \
  --load-best-model-at-end true \
  --metric-for-best-model f1 \
  --greater-is-better true \
  --lora-enable true \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --load-in-4bit true \
  --load-in-8bit false \
  --attn-implementation flash_attention_2 \
  --bf16 \
  "$@"
