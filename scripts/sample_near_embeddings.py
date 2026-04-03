import argparse
import os
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Sample nearby embeddings by local perturbation")
    parser.add_argument("--input_embeddings", required=True, type=str,
                        help="Path to embeddings.npy (GT or other embeddings)")
    parser.add_argument("--index", type=int, default=0, help="Index of GT embedding to perturb")
    parser.add_argument("--num_samples", type=int, default=25)
    parser.add_argument("--noise_sigma", type=float, default=0.5, help="Gaussian noise std for local perturbation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="outputs/embed_diffusion/near_samples.npy")
    parser.add_argument("--match_stats", action="store_true", help="Match stats to dataset embeddings")
    parser.add_argument("--real_embeddings", type=str, default="",
                        help="Path to real embeddings.npy for stats matching")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    embeds = np.load(args.input_embeddings)
    if args.index < 0 or args.index >= embeds.shape[0]:
        raise ValueError("index out of range for input embeddings")
    base = embeds[args.index]

    noise = rng.randn(args.num_samples, base.shape[0]) * args.noise_sigma
    samples = base[None, :] + noise

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


if __name__ == "__main__":
    main()
