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


def make_window_weights(length, start, size, shape, device):
    w = torch.zeros(length, device=device)
    end = start + size
    if shape == "flat":
        w[start:end] = 1.0
    elif shape == "gaussian":
        idx = torch.arange(start, end, device=device)
        center = (start + end - 1) / 2.0
        sigma = max(size / 4.0, 1e-6)
        vals = torch.exp(-0.5 * ((idx - center) / sigma) ** 2)
        if vals.max() > 0:
            vals = vals / vals.max()
        w[start:end] = vals
    elif shape == "triangular":
        idx = torch.arange(start, end, device=device)
        center = (start + end - 1) / 2.0
        span = max((end - start) / 2.0, 1e-6)
        vals = 1.0 - torch.abs(idx - center) / span
        vals = torch.clamp(vals, min=0.0, max=1.0)
        w[start:end] = vals
    else:
        raise ValueError(f"Unknown window_shape: {shape}")
    return w[start:end]


def apply_boost(x, w, boost_mode, boost_scale):
    if boost_mode == "multiply":
        return x * (1.0 + (boost_scale - 1.0) * w)
    if boost_mode == "add":
        return x + boost_scale * w
    raise ValueError(f"Unknown boost_mode: {boost_mode}")


def window_starts(length, window_size, window_step):
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if window_step <= 0:
        raise ValueError("window_step must be > 0")
    starts = list(range(0, max(1, length - window_size + 1), window_step))
    last = length - window_size
    if last > 0 and (len(starts) == 0 or starts[-1] != last):
        starts.append(last)
    return starts


def compute_window_params(length, num_windows, overlap_pct):
    if num_windows <= 0:
        raise ValueError("num_windows must be > 0")
    overlap = max(0.0, min(0.99, overlap_pct / 100.0))
    denom = 1.0 + (num_windows - 1) * (1.0 - overlap)
    window_size = max(1, int(round(length / denom)))
    window_step = max(1, int(round(window_size * (1.0 - overlap))))
    return window_size, window_step


def gen_samples_in_chunks(model, device, y_batch, emb_batch, npoints, clip_denoised, max_total_batch):
    outputs = []
    total = emb_batch.shape[0]
    for i in range(0, total, max_total_batch):
        emb_chunk = emb_batch[i:i + max_total_batch]
        y_chunk = y_batch[i:i + max_total_batch]
        noise_shape = (emb_chunk.shape[0], 3, npoints)
        gen = model.gen_samples(
            shape=noise_shape,
            device=device,
            y=y_chunk,
            mae_embed=emb_chunk,
            clip_denoised=clip_denoised,
        )
        outputs.append(gen.detach().cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Window-boost probing for MAE embeddings/tokens")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--dataroot", required=True, type=str)
    parser.add_argument("--category", default="chair", type=str)
    parser.add_argument("--idx", type=int, default=-1, help="Dataset index for parent sample (-1 = random)")
    parser.add_argument("--voxel_size", type=int, default=32)
    parser.add_argument("--npoints", type=int, default=2048)
    parser.add_argument("--mae_config_path", type=str, default="configs/pretrainMAE.yaml")
    parser.add_argument("--mae_points", type=int, default=1024)
    parser.add_argument("--mae_mask_ratio", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="outputs/mae_window_boost")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--clip_denoised", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_total_batch", type=int, default=64)
    parser.add_argument("--window_mode", type=str, default="embedding", choices=["embedding", "token", "both"])
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--window_step", type=int, default=16)
    parser.add_argument("--num_windows", type=int, default=0,
                        help="If >0, compute window_size/step from this count and --overlap_pct")
    parser.add_argument("--overlap_pct", type=float, default=0.0,
                        help="Window overlap percentage (0-99). Used with --num_windows")
    parser.add_argument("--boost_mode", type=str, default="multiply", choices=["multiply", "add"])
    parser.add_argument("--boost_scale", type=float, default=1.5)
    parser.add_argument("--window_shape", type=str, default="flat", choices=["flat", "gaussian", "triangular"])
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
    idx = args.idx if args.idx >= 0 else np.random.randint(0, len(dataset))
    sample = dataset[idx]

    x0 = sample["train_points"].unsqueeze(0).to(device)  # [1, N, 3]
    y0 = torch.tensor([sample["cate_idx"]], device=device)

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

    mae_pts = model.build_mae_input(
        x0.transpose(1, 2),
        mae_points=args.mae_points,
        mae_mask_ratio=args.mae_mask_ratio,
        min_keep=min_keep
    )

    e0 = model.get_mae_embed(mae_pts)  # [1, hidden]

    os.makedirs(args.output_dir, exist_ok=True)
    visualize_pointcloud_batch(
        os.path.join(args.output_dir, "parent.png"),
        x0.detach().cpu(),
        None, None, None
    )

    outputs_by_mode = {}
    window_indices_by_mode = {}

    if args.window_mode in ("embedding", "both"):
        hidden = e0.shape[1]
        if args.num_windows > 0:
            w_size, w_step = compute_window_params(hidden, args.num_windows, args.overlap_pct)
        else:
            w_size, w_step = args.window_size, args.window_step
        starts = window_starts(hidden, w_size, w_step)
        emb_list = []
        win_indices = []
        for s in starts:
            e_win = e0.clone()
            w = make_window_weights(hidden, s, w_size, args.window_shape, device).view(1, -1)
            e_win[:, s:s + w_size] = apply_boost(
                e_win[:, s:s + w_size], w, args.boost_mode, args.boost_scale
            )
            emb_list.append(e_win)
            win_indices.append([s, s + w_size])
        emb_batch = torch.cat(emb_list, dim=0)
        emb_batch = emb_batch.repeat_interleave(args.batch_size, dim=0)
        y_batch = y0.repeat(emb_batch.shape[0])
        outputs = gen_samples_in_chunks(
            model, device, y_batch, emb_batch, args.npoints, args.clip_denoised, args.max_total_batch
        )
        outputs_by_mode["embedding"] = outputs
        window_indices_by_mode["embedding"] = np.array(win_indices, dtype=np.int32)

    if args.window_mode in ("token", "both"):
        x_tokens, _ = mae_embedder.encode_tokens(mae_pts)  # [1, seq, trans_dim]
        seq_len = x_tokens.shape[1]
        if args.num_windows > 0:
            w_size, w_step = compute_window_params(seq_len, args.num_windows, args.overlap_pct)
        else:
            w_size, w_step = args.window_size, args.window_step
        starts = window_starts(seq_len, w_size, w_step)
        emb_list = []
        win_indices = []
        for s in starts:
            x_win = x_tokens.clone()
            w = make_window_weights(seq_len, s, w_size, args.window_shape, device).view(1, -1, 1)
            x_win[:, s:s + w_size, :] = apply_boost(
                x_win[:, s:s + w_size, :], w, args.boost_mode, args.boost_scale
            )
            x_feat = x_win.max(1)[0]
            e_win = mae_embedder.final_proj(x_feat)
            emb_list.append(e_win)
            win_indices.append([s, s + w_size])
        emb_batch = torch.cat(emb_list, dim=0)
        emb_batch = emb_batch.repeat_interleave(args.batch_size, dim=0)
        y_batch = y0.repeat(emb_batch.shape[0])
        outputs = gen_samples_in_chunks(
            model, device, y_batch, emb_batch, args.npoints, args.clip_denoised, args.max_total_batch
        )
        outputs_by_mode["token"] = outputs
        window_indices_by_mode["token"] = np.array(win_indices, dtype=np.int32)

    for mode, outputs in outputs_by_mode.items():
        np.save(os.path.join(args.output_dir, f"samples_{mode}.npy"), outputs.numpy())
        np.save(os.path.join(args.output_dir, f"window_indices_{mode}.npy"), window_indices_by_mode[mode])

        # For visualization: take the first sample from each window
        num_windows = window_indices_by_mode[mode].shape[0]
        vis = outputs[::args.batch_size][:num_windows].transpose(1, 2)
        visualize_pointcloud_batch(
            os.path.join(args.output_dir, f"samples_{mode}.png"),
            vis,
            None, None, None
        )

    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
