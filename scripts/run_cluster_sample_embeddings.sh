python scripts/cluster_sample_embeddings.py \
    --input outputs/embed_diffusion/samples.npy \
    --num_clusters 25 \
    --num_select 25 \
    --max_iters 50 \
    --seed 42 \
    --output_path outputs/embed_diffusion/samples_k25.npy \
    --save_indices
