#!/usr/bin/env python
"""
Sanity-check visualization: decode stored MaskedEmbedder embeddings into point clouds
using a pretrained DiT3D checkpoint (no embedding diffusion sampling).
"""
import os
import sys
import argparse
import random

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(REPO_ROOT)

from train import get_betas, GaussianDiffusion  # noqa: E402
from models.dit3d import DiT3D_models  # noqa: E402
from utils.visualize import visualize_pointcloud_batch  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Decode stored MAE embeddings to point clouds for sanity check.")
    parser.add_argument("--embedding_data_path", type=str, default="data/mae_embeddings/chair/train.pt",
                        help="Torch file produced by compute_mae_embeddings.py")
    parser.add_argument("--dit3d_checkpoint", type=str, required=True,
                        help="Pretrained DiT3D checkpoint used for decoding embeddings.")
    parser.add_argument("--output", type=str, default="assets/embedding_dataset_viz.png",
                        help="Where to save the generated point cloud grid.")

    parser.add_argument("--num_samples", type=int, default=16, help="Number of embeddings/point clouds to visualize.")
    parser.add_argument("--model_type", type=str, default="DiT-S/4", choices=list(DiT3D_models.keys()))
    parser.add_argument("--voxel_size", type=int, default=32, choices=[16, 32, 64, 128, 256])
    parser.add_argument("--viz_points", type=int, default=2048, help="Number of points to generate per sample.")
    parser.add_argument("--viz_nc", type=int, default=3, help="Channels for viz output (keep at 3 for XYZ).")
    parser.add_argument("--class_label", type=int, default=None,
                        help="Override class index; if unset, use cate_idx from the embedding file (or 0 fallback).")
    parser.add_argument("--num_classes", type=int, default=1, help="Total classes expected by label embedder.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Diffusion hyperparameters (must match DiT3D training)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--time_num", type=int, default=1000)
    parser.add_argument("--schedule_type", type=str, default="linear")
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--model_mean_type", type=str, default="eps")
    parser.add_argument("--model_var_type", type=str, default="fixedsmall")

    parser.add_argument("--use_mae", action="store_true", help="Enable MAE embedder inside DiT3D if trained that way.")
    parser.add_argument("--mae_config_path", type=str, default="configs/pretrainMAE.yaml")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_embeddings(path, num_samples, seed):
    random.seed(seed)
    payload = torch.load(path, map_location="cpu")
    embeddings = payload["embeddings"]  # (N, dim)
    cate_idx = payload.get("cate_idx", None)

    total = embeddings.shape[0]
    idxs = random.sample(range(total), k=min(num_samples, total))
    emb_sel = embeddings[idxs]
    if cate_idx is not None:
        cate_sel = cate_idx[idxs]
    else:
        cate_sel = None
    return emb_sel, cate_sel


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

    embeddings_cpu, cate_idx_cpu = load_embeddings(args.embedding_data_path, args.num_samples, args.seed)
    embeddings = embeddings_cpu.to(device)
    y_labels = None
    if args.class_label is not None:
        y_labels = torch.full((embeddings.shape[0],), args.class_label, device=device, dtype=torch.long)
    elif cate_idx_cpu is not None:
        y_labels = cate_idx_cpu.to(device)
    else:
        y_labels = torch.zeros((embeddings.shape[0],), device=device, dtype=torch.long)

    viz_model, viz_diffusion = build_viz_decoder(args, betas, device)

    embeddings_condition = embeddings  # (B, dim)

    def viz_denoise(data, t, y_inner):
        return viz_model(data, t, y_inner, mae_embed=embeddings_condition)

    x_gen = viz_diffusion.p_sample_loop(
        viz_denoise,
        shape=(embeddings.shape[0], args.viz_nc, args.viz_points),
        device=device,
        y=y_labels,
        clip_denoised=False,
    )

    visualize_pointcloud_batch(args.output, x_gen.transpose(1, 2), None, None, None)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
