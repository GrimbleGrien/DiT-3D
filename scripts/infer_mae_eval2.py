import argparse
import os
import sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train import Model, get_betas, get_dataset, get_default_mae_mask_ratio
from utils.visualize import visualize_pointcloud_batch


def new_x_chain(x, num_chain):
    return torch.randn(num_chain, *x.shape[1:], device=x.device)


def sample_eval_cond(x, y, num_samples):
    """
    Sample a conditioning batch of size num_samples from the current batch.
    If num_samples > batch size, sample with replacement.
    """
    if x.shape[0] >= num_samples:
        idx = torch.randperm(x.shape[0], device=x.device)[:num_samples]
    else:
        idx = torch.randint(0, x.shape[0], (num_samples,), device=x.device)
    return x[idx], y[idx]


def build_args(cli):
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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Train-eval-matched MAE inference for two samples")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--dataroot", required=True, type=str)
    parser.add_argument("--category", default="chair", type=str)
    parser.add_argument("--voxel_size", type=int, default=32)
    parser.add_argument("--npoints", type=int, default=2048)
    parser.add_argument("--mae_config_path", type=str, default="configs/pretrainMAE.yaml")
    parser.add_argument("--mae_points", type=int, default=1024)
    parser.add_argument("--mae_mask_ratio", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="outputs/mae_eval2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--eval_bs", type=int, default=25)
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

    dataset, _ = get_dataset(args.dataroot, args.npoints, args.category)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_bs, shuffle=True, drop_last=True)
    batch = next(iter(loader))
    x = batch["train_points"].transpose(1, 2).to(device)  # [B, 3, N]
    y = batch["cate_idx"].to(device)

    betas = get_betas(args.schedule_type, args.beta_start, args.beta_end, args.time_num)
    opt_for_model = build_args(args)
    model = Model(opt_for_model, betas, opt_for_model.loss_type, opt_for_model.model_mean_type, opt_for_model.model_var_type)
    model = model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_key = "ema" if (args.use_ema and "ema" in ckpt) else "model_state"
    model.load_state_dict(ckpt[state_key], strict=False)
    model.eval()

    x_cond, y_eval = sample_eval_cond(x, y, args.eval_bs)

    mae_embedder = model._get_mae_embedder()
    min_keep = getattr(mae_embedder, "num_group", None)

    mae_eval = model.build_mae_input(
        x_cond,
        mae_points=args.mae_points,
        mae_mask_ratio=args.mae_mask_ratio,
        min_keep=min_keep
    )

    gen_eval = model.gen_samples(
        shape=new_x_chain(x_cond, args.eval_bs).shape,
        device=device,
        y=y_eval,
        mae=mae_eval,
        clip_denoised=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "samples_eval.npy"), gen_eval.cpu().numpy())
    np.save(os.path.join(args.output_dir, "samples_cond.npy"), x_cond.cpu().numpy())

    visualize_pointcloud_batch(
        os.path.join(args.output_dir, "samples_eval.png"),
        gen_eval.transpose(1, 2),
        None, None, None
    )
    visualize_pointcloud_batch(
        os.path.join(args.output_dir, "samples_cond.png"),
        x_cond.transpose(1, 2),
        None, None, None
    )

    stats = [gen_eval.mean().item(), gen_eval.std().item()]
    gen_eval_range = [gen_eval.min().item(), gen_eval.max().item()]
    print(
        f"eval_gen_range: [{gen_eval_range[0]:.4f}, {gen_eval_range[1]:.4f}] "
        f"eval_gen_stats: [mean={stats[0]:.4f}, std={stats[1]:.4f}]"
    )
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
