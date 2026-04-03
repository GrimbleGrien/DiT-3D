python scripts/sample_embed_diffusion.py \
    --checkpoint checkpoints/embed_diffusion/best_ema.pth \
    --num_samples 500 \
    --output_path outputs/embed_diffusion/samples.npy \
    --device cuda:0 \
    --seed 42 \
    --real_embeddings data/mae_embeddings/embeddings.npy \
    --model_type transformer \
    --depth 8 \
    --num_heads 8 \
    --token_dim 64 \
    --dropout 0.1 \
    --match_stats

python scripts/cluster_sample_embeddings.py \
    --input outputs/embed_diffusion/samples.npy \
    --num_clusters 25 \
    --num_select 25 \
    --max_iters 50 \
    --seed 42 \
    --output_path outputs/embed_diffusion/samples_k25.npy \
    --save_indices
