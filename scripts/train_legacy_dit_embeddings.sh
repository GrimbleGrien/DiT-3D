#!/usr/bin/env bash
# Wrapper to train the legacy DiT backbone on the precomputed embedding dataset.

set -euo pipefail

python train.py \
  --dataroot ../../PSF/data/ShapeNetCore.v2.PC15k/ \
  --category chair \
  --target_embeddings \
  --embedding_backbone dit \
  --embedding_data_path data/mae_embeddings/chair/train.pt \
  --embedding_dim 384 \
  --num_classes 1 \
  --experiment_name embedding_dit_legacy \
  --model_type DiT-S/4 \
  --bs 32 \
  --lr 1e-4 \
  --niter 10000 \
  --saveIter 10 \
  --vizIter 1000 \
  --viz_nc 3 \
  --viz_points 2048 \
  --use_tb \
  --embedding_viz_checkpoint checkpoints/mae1000/best.pth \
  --model checkpoints/embedding_dit_legacy/latest.pth
