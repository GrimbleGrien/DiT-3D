import argparse
import os
import numpy as np
import torch

from train import (
    Model,
    get_betas,
    get_dataset,
    get_default_mae_mask_ratio,
)


def build_args(cli):
    # Minimal namespace compatible with Model constructor
    return argparse.Namespace(
        model_dir="",
        experiment_name="",
        window_size=0,
        window_block_indexes="0,3,6,9",
        attention=True,
        dropout=0.1,
        embed_dim=64,
        loss_type="mse",
        model_mean_type="eps",
        model_var_type="fixedsmall",
        model_type=cli.model_type,
        use_pretrained=False,
        use_mae=True,
        mae_config_path=cli.mae_config_path,
        voxel_size=cli.voxel_size,
        num_classes=cli.num_classes,
        schedule_type=cli.schedule_type,
        beta_start=cli.beta_start,
        beta_end=cli.beta_end,
        time_num=cli.time_num,
    )


def normalize_state_dict(state_dict):
    if any(k.startswith("model.module.") for k in state_dict.keys()):
        return {k.replace("model.module.", "model."): v for k, v in state_dict.items()}
    return state_dict


def mask_points(pts, mae_points, mae_mask_ratio):
    """
    pts: [B, N, 3]
    Returns masked_pts: [B, mae_points, 3]
    """
    bsz, n_pts, _ = pts.shape
    target = min(mae_points, n_pts)
    if target < n_pts:
        idx = torch.randperm(n_pts, device=pts.device)[:target]
        idx = idx.unsqueeze(0).expand(bsz, -1)
        pts = torch.gather(pts, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    if mae_mask_ratio > 0:
        keep = torch.rand(pts.shape[:2], device=pts.device) > mae_mask_ratio
        pts = pts * keep.unsqueeze(-1)
    return pts


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="MAE-conditioned DiT3D inference")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--dataroot", required=True, type=str)
    parser.add_argument("--category", default="chair", type=str)
    parser.add_argument("--voxel_size", type=int, default=32)
    parser.add_argument("--npoints", type=int, default=2048)
    parser.add_argument("--mae_config_path", type=str, default="configs/pretrainMAE.yaml")
    parser.add_argument("--mae_points", type=int, default=1024)
    parser.add_argument("--mae_mask_ratio", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_pc", type=str, default="generated.npy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--model_type", type=str, default="DiT-S/4")
    parser.add_argument("--num_classes", type=int, default=55)
    parser.add_argument("--schedule_type", type=str, default="linear")
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--time_num", type=int, default=1000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mae_mask_ratio is None:
        args.mae_mask_ratio = get_default_mae_mask_ratio(args.mae_config_path)

    device = torch.device(args.device)

    # dataset for conditioning
    dataset, _ = get_dataset(args.dataroot, args.npoints, args.category)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(loader))
    x_cond = batch["train_points"].to(device)  # [B, npoints, 3]
    y = batch["cate_idx"].to(device)

    # prepare MAE conditioning
    mae_pts = mask_points(x_cond, args.mae_points, args.mae_mask_ratio)  # [B, m, 3]

    # noise for diffusion
    noise_shape = (args.batch_size, 3, args.npoints)
    # model setup
    betas = get_betas(args.schedule_type, args.beta_start, args.beta_end, args.time_num)
    opt_for_model = build_args(args)
    model = Model(opt_for_model, betas, opt_for_model.loss_type, opt_for_model.model_mean_type, opt_for_model.model_var_type)
    model = model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_key = "ema" if (args.use_ema and "ema" in ckpt) else "model_state"
    state = normalize_state_dict(ckpt[state_key])
    model.load_state_dict(state)
    model.eval()

    # run conditioned sampling
    gen = model.gen_samples(shape=noise_shape, device=device, y=y, mae=mae_pts, clip_denoised=True)

    # optional unconditioned comparison
    gen_uncond = model.gen_samples(shape=noise_shape, device=device, y=y, mae=None, clip_denoised=True)

    # stats and save
    gen_np = gen.cpu().numpy()
    out_dir = os.path.dirname(args.output_pc)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.output_pc, gen_np)

    l2_diff = torch.norm(gen - gen_uncond, dim=1).mean().item()
    stats = {
        "cond_min": float(gen.min().item()),
        "cond_max": float(gen.max().item()),
        "cond_mean": float(gen.mean().item()),
        "cond_std": float(gen.std().item()),
        "l2_diff_vs_uncond": l2_diff,
    }
    print(stats)
    print(f"Saved conditioned samples to {args.output_pc}")


if __name__ == "__main__":
    main()
