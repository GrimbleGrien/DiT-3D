import argparse
import os
import sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train import Model, get_betas
from utils.visualize import visualize_pointcloud_batch


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


def normalize_state_dict(state_dict):
    if any(k.startswith("model.module.") for k in state_dict.keys()):
        return {k.replace("model.module.", "model."): v for k, v in state_dict.items()}
    return state_dict


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Generate point clouds from precomputed MAE embeddings")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--embeddings", required=True, type=str)
    parser.add_argument("--output_dir", type=str, default="outputs/pc_from_embeds")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--npoints", type=int, default=2048)
    parser.add_argument("--voxel_size", type=int, default=32)
    parser.add_argument("--mae_config_path", type=str, default="configs/pretrainMAE.yaml")
    parser.add_argument("--model_type", type=str, default="DiT-S/4")
    parser.add_argument("--num_classes", type=int, default=55)
    parser.add_argument("--schedule_type", type=str, default="linear")
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--time_num", type=int, default=1000)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--clip_denoised", action="store_true")
    parser.add_argument("--class_idx", type=int, default=0, help="Class index used for conditioning")
    parser.add_argument("--debug_stats", action="store_true", help="Print embedding/output diversity stats")
    parser.add_argument("--dataroot", type=str, default="", help="Optional: use masked GT input for MAE conditioning")
    parser.add_argument("--category", type=str, default="chair")
    parser.add_argument("--mae_points", type=int, default=1024)
    parser.add_argument("--mae_mask_ratio", type=float, default=0.6)
    parser.add_argument("--gt_index", type=int, default=-1, help="GT index for masked MAE conditioning (required if dataroot set)")
    parser.add_argument("--mae_only", action="store_true", help="Ignore mae_embed and condition only on MAE input")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    embeds = np.load(args.embeddings)
    emb_tensor = torch.from_numpy(embeds).float().to(device)

    betas = get_betas(args.schedule_type, args.beta_start, args.beta_end, args.time_num)
    opt_for_model = build_args(args)
    model = Model(opt_for_model, betas, opt_for_model.loss_type, opt_for_model.model_mean_type, opt_for_model.model_var_type)
    model = model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_key = "ema" if (args.use_ema and "ema" in ckpt) else "model_state"
    state = normalize_state_dict(ckpt[state_key])
    model.load_state_dict(state, strict=False)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    mae_in = None
    if args.dataroot:
        if args.gt_index < 0:
            raise ValueError("gt_index must be set when using dataroot for MAE conditioning")
        from train import get_dataset
        dataset, _ = get_dataset(args.dataroot, args.npoints, args.category)
        if args.gt_index >= len(dataset):
            raise ValueError("gt_index out of range for dataset")
        sample = dataset[args.gt_index]
        x0 = sample["train_points"].unsqueeze(0).to(device)  # [1, N, 3]
        mae_embedder = model._get_mae_embedder()
        min_keep = getattr(mae_embedder, "num_group", None)
        mae_in = model.build_mae_input(
            x0.transpose(1, 2),
            mae_points=args.mae_points,
            mae_mask_ratio=args.mae_mask_ratio,
            min_keep=min_keep
        )

    outputs = []
    grid_idx = 0
    for i in range(0, emb_tensor.shape[0], args.batch_size):
        emb_batch = emb_tensor[i:i + args.batch_size]
        y = torch.full((emb_batch.shape[0],), args.class_idx, dtype=torch.long, device=device)
        noise_shape = (emb_batch.shape[0], 3, args.npoints)
        gen = model.gen_samples(
            shape=noise_shape,
            device=device,
            y=y,
            mae=mae_in,
            mae_embed=None if args.mae_only else emb_batch,
            clip_denoised=args.clip_denoised,
        )
        gen_cpu = gen.detach().cpu()
        if args.debug_stats and i == 0:
            with torch.no_grad():
                emb_np = emb_batch.detach().cpu().numpy()
                if emb_np.shape[0] >= 2:
                    d01 = float(np.linalg.norm(emb_np[0] - emb_np[1]))
                    d12 = float(np.linalg.norm(emb_np[1] - emb_np[2])) if emb_np.shape[0] >= 3 else None
                else:
                    d01, d12 = None, None
                out_np = gen_cpu.numpy()
                out_d01 = float(np.linalg.norm(out_np[0] - out_np[1])) if out_np.shape[0] >= 2 else None
                print(f"[debug] emb_l2_0_1={d01} emb_l2_1_2={d12} out_l2_0_1={out_d01}")
        outputs.append(gen_cpu)
        grid_path = os.path.join(args.output_dir, f"pc_from_embeddings_batch_{grid_idx:04d}.png")
        visualize_pointcloud_batch(
            grid_path,
            gen_cpu.transpose(1, 2),
            None, None, None
        )
        grid_idx += 1

    outputs = torch.cat(outputs, dim=0)
    np.save(os.path.join(args.output_dir, "pc_from_embeddings.npy"), outputs.numpy())
    visualize_pointcloud_batch(
        os.path.join(args.output_dir, "pc_from_embeddings.png"),
        outputs.transpose(1, 2),
        None, None, None
    )
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
