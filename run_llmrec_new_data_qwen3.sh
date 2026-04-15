#!/usr/bin/env bash
set -euo pipefail

# 1) Data augmentation with Qwen3-8B (thinking mode disabled in code)
python llmrec_qwen3_pipeline.py \
  --data_dir LLMRec-new-data \
  --dataset Baby_Products \
  --output_dir outputs_qwen3_pipeline \
  --stage augment \
  --max_aug_users 300 \
  --max_aug_items 1000

# 2) Train simplified LLMRec-style BPR model
python llmrec_qwen3_pipeline.py \
  --data_dir LLMRec-new-data \
  --dataset Baby_Products \
  --output_dir outputs_qwen3_pipeline \
  --stage train \
  --epochs 10 \
  --batch_size 1024 \
  --emb_dim 64

# 3) Evaluate with 1 target + 1000 random sampled negatives
python llmrec_qwen3_pipeline.py \
  --data_dir LLMRec-new-data \
  --dataset Baby_Products \
  --output_dir outputs_qwen3_pipeline \
  --stage eval \
  --eval_user_source test \
  --eval_negatives 1000
