#!/usr/bin/env bash
# Generate MaskedEmbedder embeddings for ShapeNet chair split.

# set -euo pipefail

python compute_mae_embeddings.py \
  --split train \
  --category chair \
  --batch_size 64 \
  --num_workers 4 \
  --max_samples 10000 \
  --output data/mae_embeddings/chair/train.pt \
  --checkpoint checkpoints/mae1000/best.pth \
  --device cuda
