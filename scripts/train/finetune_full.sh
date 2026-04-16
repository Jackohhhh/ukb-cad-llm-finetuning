#!/usr/bin/env bash
#
# 全量微调（无 LoRA、无量化）。显存需求远高于 QLoRA；多卡 + DeepSpeed ZeRO 仍可能吃紧，请按卡显存调 batch / accum。
# Flash Attention：pip install -e ".[flash]" 后在 deepspeed 命令中加 --attn-implementation flash_attention_2
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

deepspeed --num_gpus=1 --module cad_finetune.cli.train \
  --config configs/experiments/qwen2_medical_full.yaml \
  --experiment-name qwen2_medical_full \
  --model-name-or-path Qwen/Qwen2-7B-Instruct \
  --train-file data/raw/medical_train.json \
  --test-file data/raw/medical_test.json \
  --output-dir outputs/checkpoints/qwen2_medical_full \
  --prediction-output-dir outputs/predictions/qwen2_medical_full \
  --logging-dir outputs/logs/qwen2_medical_full \
  --launcher deepspeed \
  --seed 42 \
  --gradient-checkpointing true \
  --dataloader-num-workers 4 \
  --report-to none \
  --deepspeed configs/deepspeed/zero2.json \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --lr-scheduler-type cosine \
  --logging-steps 50 \
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
  --bf16 \
  --optim adamw_torch \
  "$@"
