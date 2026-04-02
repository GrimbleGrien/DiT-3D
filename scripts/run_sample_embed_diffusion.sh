python scripts/sample_embed_diffusion.py \
    --checkpoint checkpoints/embed_diffusion/best.pth \
    --num_samples 50 \
    --output_path outputs/embed_diffusion/samples.npy \
    --device cuda:0 \
    --seed 42 \
    --real_embeddings data/mae_embeddings/embeddings.npy
