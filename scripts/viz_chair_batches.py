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


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Visualize GT chair dataset in batches")
    parser.add_argument("--dataroot", required=True, type=str)
    parser.add_argument("--category", default="chair", type=str)
    parser.add_argument("--npoints", type=int, default=2048)
    parser.add_argument("--start_idx", type=int, default=1)
    parser.add_argument("--end_idx", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--output_dir", type=str, default="outputs/chair_gt_batches")
    args = parser.parse_args()

    dataset, _ = get_dataset(args.dataroot, args.npoints, args.category)
    os.makedirs(args.output_dir, exist_ok=True)

    start = max(0, args.start_idx)
    end = min(args.end_idx, len(dataset) - 1)

    for batch_start in range(start, end + 1, args.batch_size):
        batch_end = min(batch_start + args.batch_size - 1, end)
        samples = []
        for idx in range(batch_start, batch_end + 1):
            sample = dataset[idx]
            samples.append(sample["train_points"].unsqueeze(0))  # [1, N, 3]
        x = torch.cat(samples, dim=0)  # [B, N, 3]
        out_path = os.path.join(args.output_dir, f"chair_{batch_start:04d}_{batch_end:04d}.png")
        visualize_pointcloud_batch(out_path, x, None, None, None)

    print(f"Saved batches to {args.output_dir}")


if __name__ == "__main__":
    main()
