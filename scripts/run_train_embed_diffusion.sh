python scripts/train_embed_diffusion.py \
    --emb_path data/mae_embeddings/embeddings.npy \
    --epochs 200 \
    --time_num 200 \
    --batch_size 256 \
    --lr 1e-4 \
    --output_dir checkpoints/embed_diffusion \
    --seed 42 \
    --device cuda:0
