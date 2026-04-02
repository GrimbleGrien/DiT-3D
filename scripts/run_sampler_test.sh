python scripts/infer_from_embeddings.py \
  --checkpoint checkpoints/jmae500/best.pth \
  --embeddings data/mae_embeddings/embeddings.npy \
  --output_dir outputs/pc_from_real_embeds \
  --device cuda:0 \
  --batch_size 8 \
  --use_ema \
  --class_idx 0
