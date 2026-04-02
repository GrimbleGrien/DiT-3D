import argparse
import os
import sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.embed_diffusion import DenoiserMLP, DenoiserTransformer, GaussianDiffusion1D, get_betas


def main():
    parser = argparse.ArgumentParser(description="Sample MAE embeddings from diffusion model")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_path", type=str, default="outputs/embed_diffusion/samples.npy")
    parser.add_argument("--save_pt", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time_num", type=int, default=200)
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--schedule_type", type=str, default="linear")
    parser.add_argument("--embed_dim", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--time_dim", type=int, default=256)
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "transformer"])
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--token_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--real_embeddings", type=str, default="",
                        help="Optional path to real embeddings.npy for stats comparison")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})

    embed_dim = cfg.get("embed_dim", args.embed_dim)
    hidden_dim = cfg.get("hidden_dim", args.hidden_dim)
    time_dim = cfg.get("time_dim", args.time_dim)
    model_type = cfg.get("model_type", args.model_type)
    depth = cfg.get("depth", args.depth)
    num_heads = cfg.get("num_heads", args.num_heads)
    token_dim = cfg.get("token_dim", args.token_dim)
    dropout = cfg.get("dropout", args.dropout)
    time_num = cfg.get("time_num", args.time_num)
    beta_start = cfg.get("beta_start", args.beta_start)
    beta_end = cfg.get("beta_end", args.beta_end)
    schedule_type = cfg.get("schedule_type", args.schedule_type)

    if embed_dim == 0:
        raise ValueError("embed_dim must be provided if not stored in checkpoint")

    if model_type == "mlp":
        model = DenoiserMLP(embed_dim=embed_dim, hidden_dim=hidden_dim, time_dim=time_dim).to(device)
    else:
        model = DenoiserTransformer(
            embed_dim=embed_dim,
            token_dim=token_dim,
            depth=depth,
            num_heads=num_heads,
            time_dim=time_dim,
            dropout=dropout,
        ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    betas = get_betas(schedule_type, beta_start, beta_end, time_num)
    diffusion = GaussianDiffusion1D(betas)

    with torch.no_grad():
        samples = diffusion.p_sample_loop(model, shape=(args.num_samples, embed_dim), device=device)

    samples_np = samples.cpu().numpy()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, samples_np)
    if args.save_pt:
        torch.save(samples.cpu(), os.path.splitext(args.output_path)[0] + ".pt")

    def stats(arr):
        arr = arr.astype(np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "l2_mean": float(np.linalg.norm(arr, axis=1).mean()),
            "l2_std": float(np.linalg.norm(arr, axis=1).std()),
        }

    print(f"Saved samples to {args.output_path}")
    s_stats = stats(samples_np)
    print(f"sample_stats: {s_stats}")
    if args.real_embeddings:
        real = np.load(args.real_embeddings)
        r_stats = stats(real)
        print(f"real_stats: {r_stats}")


if __name__ == "__main__":
    main()
