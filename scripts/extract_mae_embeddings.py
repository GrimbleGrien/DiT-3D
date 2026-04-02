import argparse
import os
import sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train import Model, get_betas, get_dataset, get_default_mae_mask_ratio


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
    parser = argparse.ArgumentParser(description="Extract MAE embeddings for a dataset")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--dataroot", required=True, type=str)
    parser.add_argument("--category", default="chair", type=str)
    parser.add_argument("--npoints", type=int, default=2048)
    parser.add_argument("--voxel_size", type=int, default=32)
    parser.add_argument("--mae_config_path", type=str, default="configs/pretrainMAE.yaml")
    parser.add_argument("--mae_points", type=int, default=1024)
    parser.add_argument("--mae_mask_ratio", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/mae_embeddings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=0)
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
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    if args.max_samples and args.max_samples > 0:
        indices = indices[: args.max_samples]

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

    emb_list = []
    label_list = []
    idx_list = []

    for idx in indices:
        sample = dataset[int(idx)]
        x = sample["train_points"].unsqueeze(0).to(device)  # [1, N, 3]
        y = sample["cate_idx"]

        mae_pts = model.build_mae_input(
            x.transpose(1, 2),
            mae_points=args.mae_points,
            mae_mask_ratio=args.mae_mask_ratio,
            min_keep=min_keep
        )
        emb = model.get_mae_embed(mae_pts)  # [1, D]
        emb_list.append(emb.squeeze(0).cpu().numpy())
        label_list.append(int(y))
        idx_list.append(int(idx))

    embeddings = np.stack(emb_list, axis=0)
    labels = np.asarray(label_list, dtype=np.int64)
    indices_out = np.asarray(idx_list, dtype=np.int64)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(args.output_dir, "labels.npy"), labels)
    np.save(os.path.join(args.output_dir, "indices.npy"), indices_out)

    print(f"Saved embeddings to {args.output_dir}")


if __name__ == "__main__":
    main()
