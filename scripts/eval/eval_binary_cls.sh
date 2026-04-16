#!/usr/bin/env bash
#
# 离线评估：加载 checkpoint，对 configs 里 test_file 做 predict，结果写入 prediction_output_dir。
# --checkpoint：LoRA 填含 adapter_config.json 的目录。训练结束若 save_model 写在 output 根目录则用根路径；
#   若只有按步保存的子目录则用例如 outputs/.../checkpoint-200。全量微调填完整权重目录（config.json + 权重）。
#
# 输出文件：
#   --prediction-output-dir/metrics.json      汇总指标（accuracy / f1 / auc 等）
#   --prediction-output-dir/predictions.jsonl 逐条预测与正类概率
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python3 -m cad_finetune.cli.eval \
  --config configs/experiments/qwen2_medical_lora.yaml \
  --checkpoint outputs/checkpoints/qwen2_medical_lora \
  --model-name-or-path Qwen/Qwen2-7B-Instruct \
  --prediction-output-dir outputs/predictions/qwen2_medical_lora_eval \
  --per-device-eval-batch-size 8 \
  --dataloader-num-workers 2 \
  "$@"
