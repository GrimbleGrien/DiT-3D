#!/usr/bin/env bash
# Wrapper to train the legacy DiT backbone on the precomputed embedding dataset.

set -euo pipefail

python train.py \
  --target_embeddings \
  --embedding_backbone dit \
  --embedding_data_path data/mae_embeddings/chair/train.pt \
  --embedding_dim 384 \
  --experiment_name embedding_dit_legacy \
  --model_type DiT-S/4 \
  --bs 32 \
  --lr 1e-4 \
  --niter 1000 \
  --saveIter 10 \
  --vizIter 10 \
  --use_tb
