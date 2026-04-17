#!/usr/bin/env bash
#
# LoRA 微调（bf16 全精度基座，无量化）。显存显著高于 QLoRA（finetune_qlora.sh）；单卡请按需降低
# per-device-train-batch-size / 增大 gradient-accumulation-steps。
# 优化器：adamw_torch（与全量微调一致；QLoRA 用的 paged_adamw_32bit 主要配合 4bit 省显存）。
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

deepspeed --num_gpus=1 --module cad_finetune.cli.train \
  --deepspeed configs/deepspeed/zero2.json \
  --config configs/experiments/qwen2_medical_lora.yaml \
  --experiment-name qwen2_medical_lora_bf16 \
  --model-name-or-path Qwen/Qwen2-7B-Instruct \
  --train-file data/raw/medical_train.json \
  --test-file data/raw/medical_test.json \
  --output-dir outputs/checkpoints/qwen2_medical_lora_bf16 \
  --prediction-output-dir outputs/predictions/qwen2_medical_lora_bf16 \
  --launcher deepspeed \
  --bf16 \
  --tf32 true \
  --load-in-4bit false \
  --load-in-8bit false \
  --lora-r 16 \
  --lora-alpha 32 \
  --gradient-checkpointing true \
  --dataloader-num-workers 4 \
  --num-train-epochs 5 \
  --per-device-train-batch-size 32 \
  --per-device-eval-batch-size 16 \
  --gradient-accumulation-steps 1 \
  --learning-rate 3e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --lr-scheduler-type cosine \
  --logging-steps 10 \
  --logging-first-step true \
  --log-level info \
  --optim adamw_torch \
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
