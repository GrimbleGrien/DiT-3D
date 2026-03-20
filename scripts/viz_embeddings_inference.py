#!/usr/bin/env python
"""
Generate point cloud visualizations by sampling the pretrained embedding diffusion
model and decoding the embeddings with a pretrained DiT3D checkpoint.
"""
import os
import sys
import argparse
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(REPO_ROOT)

from train import Model, get_betas, GaussianDiffusion  # noqa: E402
from models.dit3d import DiT3D_models  # noqa: E402
from utils.visualize import visualize_pointcloud_batch  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Inference-only embedding -> DiT3D visualization.")
    parser.add_argument("--embedding_checkpoint", type=str, required=True,
                        help="Path to the trained embedding diffusion checkpoint (best/ema).")
    parser.add_argument("--dit3d_checkpoint", type=str, required=True,
                        help="Path to the pretrained DiT3D checkpoint used for decoding embeddings.")
    parser.add_argument("--output", type=str, default="assets/embedding_viz.png",
                        help="Where to save the generated point cloud grid.")

    parser.add_argument("--num_samples", type=int, default=16, help="Number of embeddings/point clouds to generate.")
    parser.add_argument("--embedding_dim", type=int, default=384, help="Dimension of the MaskedEmbedder vectors.")
    parser.add_argument("--embedding_backbone", type=str, default="dit", choices=["dit", "dit3d"],
                        help="Embedding backbone used during training.")
    parser.add_argument("--model_type", type=str, default="DiT-S/4", choices=list(DiT3D_models.keys()))
    parser.add_argument("--voxel_size", type=int, default=32, choices=[16, 32, 64, 128, 256])
    parser.add_argument("--viz_points", type=int, default=2048, help="Number of points to generate per sample.")
    parser.add_argument("--viz_nc", type=int, default=3, help="Channels for viz output (keep at 3 for XYZ).")
    parser.add_argument("--class_label", type=int, default=0, help="Class index to condition both models on.")
    parser.add_argument("--num_classes", type=int, default=1, help="Total classes expected by label embedder.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Diffusion hyperparameters (should match training)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--time_num", type=int, default=1000)
    parser.add_argument("--schedule_type", type=str, default="linear")
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--model_mean_type", type=str, default="eps")
    parser.add_argument("--model_var_type", type=str, default="fixedsmall")

    parser.add_argument("--use_mae", action="store_true", help="Enable MAE embedder inside models if trained that way.")
    parser.add_argument("--mae_config_path", type=str, default="configs/pretrainMAE.yaml")
    parser.add_argument("--class_dropout_prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_embedding_model(args, betas, device):
    wrapper_args = SimpleNamespace(
        target_embeddings=True,
        embedding_backbone=args.embedding_backbone,
        embedding_dim=args.embedding_dim,
        class_dropout_prob=args.class_dropout_prob,
        num_classes=args.num_classes,
        use_mae=args.use_mae,
        mae_config_path=args.mae_config_path,
        window_size=0,
        window_block_indexes="0,3,6,9",
        model_type=args.model_type,
        use_pretrained=False,
        voxel_size=args.voxel_size,
        attention=True,
        dropout=0.1,
        embed_dim=64,
    )

    model = Model(wrapper_args, betas, args.loss_type, args.model_mean_type, args.model_var_type).to(device)
    ckpt = torch.load(args.embedding_checkpoint, map_location="cpu")
    state = ckpt.get("ema", ckpt.get("model_state", ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def build_viz_decoder(args, betas, device):
    viz_model = DiT3D_models[args.model_type](
        pretrained=False,
        input_size=args.voxel_size,
        num_classes=args.num_classes,
        use_mae=True,
        mae_config_path=args.mae_config_path,
    ).to(device)
    ckpt = torch.load(args.dit3d_checkpoint, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    viz_model.load_state_dict(state, strict=False)
    viz_model.eval()

    viz_diffusion = GaussianDiffusion(betas, args.loss_type, args.model_mean_type, args.model_var_type)
    return viz_model, viz_diffusion


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    betas = get_betas(args.schedule_type, args.beta_start, args.beta_end, args.time_num)
    device = torch.device(args.device)

    embedding_model = build_embedding_model(args, betas, device)
    viz_model, viz_diffusion = build_viz_decoder(args, betas, device)

    # Sample embeddings
    embedding_shape = (args.num_samples, args.embedding_dim, 1)
    y_labels = torch.full((args.num_samples,), args.class_label, device=device, dtype=torch.long)
    embeddings = embedding_model.gen_samples(embedding_shape, device, y_labels, clip_denoised=False)
    embeddings_condition = embeddings.squeeze(-1)

    def viz_denoise(data, t, y_inner):
        return viz_model(data, t, y_inner, mae_embed=embeddings_condition)

    x_gen = viz_diffusion.p_sample_loop(
        viz_denoise,
        shape=(args.num_samples, args.viz_nc, args.viz_points),
        device=device,
        y=y_labels,
        clip_denoised=False,
    )

    visualize_pointcloud_batch(args.output, x_gen.transpose(1, 2), None, None, None)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
