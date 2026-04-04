import argparse
import os
import sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train import get_dataset
from utils.visualize import visualize_pointcloud_batch


def main():
    parser = argparse.ArgumentParser(description="Sample nearby embeddings by local perturbation")
    parser.add_argument("--input_embeddings", required=True, type=str,
                        help="Path to embeddings.npy (GT or other embeddings)")
    parser.add_argument("--index", type=int, default=0, help="Index of GT embedding to perturb")
    parser.add_argument("--num_samples", type=int, default=25)
    parser.add_argument("--noise_sigma", type=float, default=0.5, help="Gaussian noise std for local perturbation")
    parser.add_argument("--near_mode", type=str, default="noise", choices=["noise", "slerp"],
                        help="How to generate nearby embeddings")
    parser.add_argument("--knn", type=int, default=20, help="KNN pool size for slerp mode")
    parser.add_argument("--t_min", type=float, default=0.05, help="Min interpolation t for slerp")
    parser.add_argument("--t_max", type=float, default=0.25, help="Max interpolation t for slerp")
    parser.add_argument("--preserve_norm", action="store_true", help="Rescale output to parent L2 norm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="outputs/embed_diffusion/near_samples.npy")
    parser.add_argument("--match_stats", action="store_true", help="Match stats to dataset embeddings")
    parser.add_argument("--real_embeddings", type=str, default="",
                        help="Path to real embeddings.npy for stats matching")
    parser.add_argument("--dataroot", type=str, default="", help="Optional dataset root to save GT reference image")
    parser.add_argument("--category", type=str, default="chair")
    parser.add_argument("--npoints", type=int, default=2048)
    parser.add_argument("--ref_output_path", type=str, default="",
                        help="If set, save GT reference point cloud image here")
    parser.add_argument("--debug_stats", action="store_true", help="Print sample diversity stats")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    embeds = np.load(args.input_embeddings)
    if args.index < 0 or args.index >= embeds.shape[0]:
        raise ValueError("index out of range for input embeddings")
    base = embeds[args.index]

    if args.near_mode == "noise":
        noise = rng.randn(args.num_samples, base.shape[0]) * args.noise_sigma
        samples = base[None, :] + noise
    else:
        # slerp between base and a nearby real embedding
        diffs = embeds - base[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        nn_idx = np.argsort(dists)[1: max(2, args.knn + 1)]
        samples = []
        base_norm = np.linalg.norm(base) + 1e-8
        for _ in range(args.num_samples):
            j = rng.choice(nn_idx)
            other = embeds[j]
            t = rng.uniform(args.t_min, args.t_max)
            # slerp
            b0 = base / (np.linalg.norm(base) + 1e-8)
            b1 = other / (np.linalg.norm(other) + 1e-8)
            dot = np.clip(np.dot(b0, b1), -1.0, 1.0)
            omega = np.arccos(dot)
            if omega < 1e-6:
                out = (1 - t) * base + t * other
            else:
                out = (np.sin((1 - t) * omega) / np.sin(omega)) * base + (np.sin(t * omega) / np.sin(omega)) * other
            if args.preserve_norm:
                out = out / (np.linalg.norm(out) + 1e-8) * base_norm
            samples.append(out)
        samples = np.stack(samples, axis=0)

    if args.debug_stats:
        l2_01 = float(np.linalg.norm(samples[0] - samples[1])) if samples.shape[0] >= 2 else None
        print(f"[debug] sample_l2_0_1={l2_01} std_mean={samples.std(axis=0).mean():.6f}")

    if args.match_stats and args.real_embeddings:
        real = np.load(args.real_embeddings)
        real_mean = real.mean(axis=0, keepdims=True)
        real_std = real.std(axis=0, keepdims=True) + 1e-6
        samp_mean = samples.mean(axis=0, keepdims=True)
        samp_std = samples.std(axis=0, keepdims=True) + 1e-6
        samples = (samples - samp_mean) / samp_std * real_std + real_mean

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, samples)
    print(f"Saved {samples.shape[0]} nearby embeddings to {args.output_path}")

    if args.dataroot and args.ref_output_path:
        dataset, _ = get_dataset(args.dataroot, args.npoints, args.category)
        if args.index < 0 or args.index >= len(dataset):
            raise ValueError("index out of range for dataset")
        sample = dataset[args.index]
        x0 = sample["train_points"].unsqueeze(0)  # [1, N, 3]
        os.makedirs(os.path.dirname(args.ref_output_path), exist_ok=True)
        visualize_pointcloud_batch(args.ref_output_path, x0, None, None, None)
        print(f"Saved reference image to {args.ref_output_path}")


if __name__ == "__main__":
    main()
