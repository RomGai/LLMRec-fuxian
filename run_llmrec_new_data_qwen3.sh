#!/usr/bin/env bash
set -euo pipefail

# One-shot entry:
# 1) augmentation
# 2) training
# 3) evaluation (test split only)
# Uses existing:
# - *_user_items_negs_train.csv
# - *_user_items_negs_test.csv
python llmrec_qwen3_pipeline.py \
  --data_dir LLMRec-new-data \
  --dataset Baby_Products \
  --output_dir outputs_qwen3_pipeline \
  --stage all \
  --eval_user_source test \
  --eval_negatives 1000 \
  --max_aug_users 300 \
  --max_aug_items 1000 \
  --epochs 10 \
  --batch_size 1024 \
  --emb_dim 64
