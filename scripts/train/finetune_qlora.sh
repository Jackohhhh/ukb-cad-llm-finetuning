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
# 精度：--fp16 / --bf16 二选一；无需改 configs/deepspeed/*.json（Trainer 会按此处同步 DeepSpeed）
# 换模型：改 --model-name-or-path（必要时再加 --tokenizer-name-or-path）
################## 按需改行内参数 ##################

deepspeed --num_gpus=1 --module cad_finetune.cli.train \
  --deepspeed configs/deepspeed/zero2.json \
  --config configs/experiments/qwen2_medical_lora.yaml \
  --experiment-name qwen2_medical_lora \
  --model-name-or-path Qwen/Qwen2-7B-Instruct \
  --train-file data/raw/medical_train.json \
  --test-file data/raw/medical_test.json \
  --output-dir outputs/checkpoints/qwen2_medical_lora \
  --prediction-output-dir outputs/predictions/qwen2_medical_lora \
  --launcher deepspeed \
  --bf16 \
  --tf32 true \
  --load-in-4bit true \
  --load-in-8bit false \
  --lora-r 16 \
  --lora-alpha 32 \
  --gradient-checkpointing true \
  --dataloader-num-workers 4 \
  --num-train-epochs 5 \
  --per-device-train-batch-size 64 \
  --per-device-eval-batch-size 16 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-4 \
  --weight-decay 0.0 \
  --warmup-ratio 0.1 \
  --lr-scheduler-type cosine \
  --logging-steps 10 \
  --logging-first-step true \
  --log-level info \
  --optim paged_adamw_32bit \
  --save-strategy epoch \
  --evaluation-strategy epoch \
  --save-total-limit 5 \
  --max-steps -1 \
  --max-length 1024 \
  --input-column input \
  --label-column output \
  --num-labels 2 \
  --load-best-model-at-end true \
  --metric-for-best-model auc \
  --greater-is-better true \
  --lora-enable true \
  --lora-dropout 0.05 \
  --attn-implementation flash_attention_2 \
  --report-to wandb \
  --seed 42 \
  "$@"
