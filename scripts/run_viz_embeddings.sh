#!/usr/bin/env bash
# Visualize embeddings by decoding them with a pretrained DiT3D checkpoint.
# Adjust the paths below to your checkpoints before running.

set -euo pipefail

EMBED_CKPT=${EMBED_CKPT:-checkpoints/embedding_dit_legacy/best.pth}
DIT3D_CKPT=${DIT3D_CKPT:-checkpoints/mae1000/best.pth}
OUT_PATH=${OUT_PATH:-assets/embedding_viz.png}

python scripts/viz_embeddings_inference.py \
  --embedding_checkpoint "${EMBED_CKPT}" \
  --dit3d_checkpoint "${DIT3D_CKPT}" \
  --output "${OUT_PATH}" \
  --num_samples 16 \
  --class_label 0 \
  --num_classes 1 \
  --model_type DiT-S/4 \
  --embedding_dim 384 \
  --viz_points 2048 \
  --viz_nc 3 \
  --use_mae \
  --mae_config_path configs/pretrainMAE.yaml \
  --device cuda

echo "Saved visualization to ${OUT_PATH}"
