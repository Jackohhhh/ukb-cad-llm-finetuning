#!/usr/bin/env bash
#
# 离线评估：加载 checkpoint，对 configs 里 test_file 做 predict，结果写入 prediction_output_dir。
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
  --config configs/experiments/medical.yaml \
  --checkpoint outputs/checkpoints/Qwen_Qwen2-7B-Instruct_qlora \
  --prediction-output-dir outputs/predictions/qwen2_medical_eval_qlora \
  --per-device-eval-batch-size 8 \
  --dataloader-num-workers 2 \
  "$@"
