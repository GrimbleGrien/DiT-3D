python scripts/sample_near_embeddings.py \
    --input_embeddings data/mae_embeddings/embeddings.npy \
    --index 0 \
    --num_samples 25 \
    --noise_sigma 0.5 \
    --output_path outputs/embed_diffusion/near_samples.npy \
    --real_embeddings data/mae_embeddings/embeddings.npy \
    --match_stats \
    --seed 42
