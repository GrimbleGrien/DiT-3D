#!/usr/bin/env bash
# Visualize stored MAE embeddings (no embedding sampling) by decoding them with a pretrained DiT3D checkpoint.
# Override paths via environment variables below.

set -euo pipefail

EMBED_DATA=${EMBED_DATA:-data/mae_embeddings/chair/train.pt}
DIT3D_CKPT=${DIT3D_CKPT:-checkpoints/mae1000/best.pth}
OUT_PATH=${OUT_PATH:-assets/embedding_dataset_viz.png}

python scripts/viz_embeddings_from_dataset.py \
  --embedding_data_path "${EMBED_DATA}" \
  --dit3d_checkpoint "${DIT3D_CKPT}" \
  --output "${OUT_PATH}" \
  --num_samples 16 \
  --num_classes 1 \
  --model_type DiT-S/4 \
  --viz_points 2048 \
  --viz_nc 3 \
  --use_mae \
  --mae_config_path configs/pretrainMAE.yaml \
  --device cuda

echo "Saved visualization to ${OUT_PATH}"
