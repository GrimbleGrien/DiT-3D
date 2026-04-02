import argparse
import os
import sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train import (
    Model,
    get_betas,
    get_dataset,
    get_default_mae_mask_ratio,
)
from utils.visualize import visualize_pointcloud_batch


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


def interp_t(t_vals, mode="linear", power_alpha=2.0):
    if mode == "linear":
        return t_vals
    if mode == "cosine":
        return 0.5 - 0.5 * torch.cos(np.pi * t_vals)
    if mode == "power_in":
        return torch.pow(t_vals, power_alpha)
    if mode == "power_out":
        return 1.0 - torch.pow(1.0 - t_vals, power_alpha)
    raise ValueError(f"Unknown t interpolation mode: {mode}")


def slerp(e0, e1, t_vals, eps=1e-8):
    # e0, e1: [1, hidden], t_vals: [steps, 1]
    e0_n = e0 / (e0.norm(dim=1, keepdim=True) + eps)
    e1_n = e1 / (e1.norm(dim=1, keepdim=True) + eps)
    dot = (e0_n * e1_n).sum(dim=1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(dot)  # [1,1]
    so = torch.sin(omega)
    if torch.any(so < eps):
        return (1 - t_vals) * e0 + t_vals * e1
    return (torch.sin((1 - t_vals) * omega) / so) * e0 + (torch.sin(t_vals * omega) / so) * e1


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="MAE embedding interpolation inference")
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
    parser.add_argument("--steps", type=int, default=5, help="Number of interpolation steps including endpoints")
    parser.add_argument("--output_dir", type=str, default="outputs/mae_interp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--clip_denoised", action="store_true", help="Enable clip_denoised during sampling")
    parser.add_argument("--use_point_mae", action="store_true", help="Deprecated: embeddings are always interpolated; point-clouds are not interpolated")
    parser.add_argument("--idx_a", type=int, default=-1, help="Dataset index for endpoint A (-1 = random)")
    parser.add_argument("--idx_b", type=int, default=-1, help="Dataset index for endpoint B (-1 = random)")
    parser.add_argument("--anchor_parent", type=str, default="closest", choices=["closest", "a", "b"],
                        help="Point-cloud conditioning source for all steps: closest (default), a, or b")
    parser.add_argument("--interp_mode", type=str, default="lerp",
                        choices=["lerp", "nlerp", "slerp", "cosine", "power_in", "power_out"],
                        help="Interpolation mode for MAE embeddings")
    parser.add_argument("--interp_modes", type=str, default="",
                        help="Comma-separated list of interpolation modes; overrides --interp_mode")
    parser.add_argument("--power_alpha", type=float, default=2.0, help="Exponent for power_in/power_out t-warping")
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
    idx0 = args.idx_a if args.idx_a >= 0 else np.random.randint(0, len(dataset))
    idx1 = args.idx_b if args.idx_b >= 0 else np.random.randint(0, len(dataset))
    sample0 = dataset[idx0]
    sample1 = dataset[idx1]

    x0 = sample0["train_points"].unsqueeze(0).to(device)  # [1, N, 3]
    x1 = sample1["train_points"].unsqueeze(0).to(device)
    y0 = torch.tensor([sample0["cate_idx"]], device=device)

    # model setup
    betas = get_betas(args.schedule_type, args.beta_start, args.beta_end, args.time_num)
    opt_for_model = build_args(args)
    model = Model(opt_for_model, betas, opt_for_model.loss_type, opt_for_model.model_mean_type, opt_for_model.model_var_type)
    model = model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_key = "ema" if (args.use_ema and "ema" in ckpt) else "model_state"
    state = normalize_state_dict(ckpt[state_key])
    model.load_state_dict(state, strict=False)
    model.eval()

    mae_embedder = model._get_mae_embedder()
    min_keep = getattr(mae_embedder, "num_group", None)

    # build masked MAE inputs and embeddings
    steps = max(2, args.steps)
    t_vals = torch.linspace(0, 1, steps, device=device).view(-1, 1)
    if args.use_point_mae:
        print("[warn] --use_point_mae is deprecated; embeddings are always interpolated and point-clouds are not.")

    mae0 = model.build_mae_input(
        x0.transpose(1, 2),
        mae_points=args.mae_points,
        mae_mask_ratio=args.mae_mask_ratio,
        min_keep=min_keep
    )
    mae1 = model.build_mae_input(
        x1.transpose(1, 2),
        mae_points=args.mae_points,
        mae_mask_ratio=args.mae_mask_ratio,
        min_keep=min_keep
    )
    e0 = model.get_mae_embed(mae0)  # [1, hidden]
    e1 = model.get_mae_embed(mae1)  # [1, hidden]
    if args.interp_modes.strip():
        modes = [m.strip() for m in args.interp_modes.split(",") if m.strip()]
    else:
        modes = [args.interp_mode]

    outputs_by_mode = {}
    for mode in modes:
        if mode == "lerp":
            embeds = (1 - t_vals) * e0 + t_vals * e1  # [steps, hidden]
        elif mode == "nlerp":
            mixed = (1 - t_vals) * e0 + t_vals * e1
            embeds = mixed / (mixed.norm(dim=1, keepdim=True) + 1e-8)
        elif mode == "slerp":
            embeds = slerp(e0, e1, t_vals)
        elif mode in ("cosine", "power_in", "power_out"):
            t_warp = interp_t(t_vals, mode=mode, power_alpha=args.power_alpha)
            embeds = (1 - t_warp) * e0 + t_warp * e1
        else:
            raise ValueError(f"Unknown interp_mode: {mode}")

        # batched generation for all embeddings at once (faster)
        total_bs = args.batch_size * steps
        emb_batch = embeds.repeat_interleave(args.batch_size, dim=0)  # [steps*bs, hidden]
        y_batch = y0.repeat(total_bs)  # [steps*bs]
        noise_shape = (total_bs, 3, args.npoints)
        outputs = model.gen_samples(
            shape=noise_shape,
            device=device,
            y=y_batch,
            mae_embed=emb_batch,
            clip_denoised=args.clip_denoised,
        ).detach().cpu()
        outputs_by_mode[mode] = outputs

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "cond_a.npy"), x0.detach().cpu().numpy())
    np.save(os.path.join(args.output_dir, "cond_b.npy"), x1.detach().cpu().numpy())
    if args.anchor_parent == "a":
        parent_ids = np.zeros(steps, dtype=np.int64)
    elif args.anchor_parent == "b":
        parent_ids = np.ones(steps, dtype=np.int64)
    else:
        parent_ids = (t_vals.squeeze(1) > 0.5).long().cpu().numpy()
    np.save(os.path.join(args.output_dir, "parent_ids.npy"), parent_ids)
    for mode, outputs in outputs_by_mode.items():
        interp_npy = f"interp_samples_{mode}.npy"
        np.save(os.path.join(args.output_dir, interp_npy), outputs.numpy())
    np.save(os.path.join(args.output_dir, "embed_a.npy"), e0.detach().cpu().numpy())
    np.save(os.path.join(args.output_dir, "embed_b.npy"), e1.detach().cpu().numpy())

    # visualization grid
    for mode, outputs in outputs_by_mode.items():
        interp_png = f"interp_samples_{mode}.png"
        visualize_pointcloud_batch(
            os.path.join(args.output_dir, interp_png),
            outputs.transpose(1, 2),
            None, None, None
        )
    visualize_pointcloud_batch(
        os.path.join(args.output_dir, "parents_ab.png"),
        torch.cat([x0.cpu(), x1.cpu()], dim=0),
        None, None, None
    )
    visualize_pointcloud_batch(
        os.path.join(args.output_dir, "cond_a.png"),
        x0,
        None, None, None
    )
    visualize_pointcloud_batch(
        os.path.join(args.output_dir, "cond_b.png"),
        x1,
        None, None, None
    )

    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
