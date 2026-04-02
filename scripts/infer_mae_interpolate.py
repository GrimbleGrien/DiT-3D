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
    idx0 = np.random.randint(0, len(dataset))
    idx1 = np.random.randint(0, len(dataset))
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
    model.load_state_dict(ckpt[state_key], strict=False)
    model.eval()

    mae_embedder = model._get_mae_embedder()
    min_keep = getattr(mae_embedder, "num_group", None)

    # build masked MAE inputs and embeddings
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

    steps = max(2, args.steps)
    t_vals = torch.linspace(0, 1, steps, device=device).view(-1, 1)
    embeds = (1 - t_vals) * e0 + t_vals * e1  # [steps, hidden]

    # generate samples for each embedding
    outputs = []
    for i in range(steps):
        emb = embeds[i:i+1]
        noise_shape = (args.batch_size, 3, args.npoints)
        gen = model.gen_samples(shape=noise_shape, device=device, y=y0, mae_embed=emb, clip_denoised=True)
        outputs.append(gen.detach().cpu())

    outputs = torch.cat(outputs, dim=0)  # [steps, 3, N]

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "interp_samples.npy"), outputs.numpy())
    np.save(os.path.join(args.output_dir, "embed_a.npy"), e0.detach().cpu().numpy())
    np.save(os.path.join(args.output_dir, "embed_b.npy"), e1.detach().cpu().numpy())

    # visualization grid
    visualize_pointcloud_batch(
        os.path.join(args.output_dir, "interp_samples.png"),
        outputs.transpose(1, 2),
        None, None, None
    )

    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
