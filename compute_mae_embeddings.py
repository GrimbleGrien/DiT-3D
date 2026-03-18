import argparse
import os

import torch
from torch.utils.data import DataLoader

from datasets.shapenet_data_pc import ShapeNet15kPointClouds
from models.dit3d import DiT_S_4


def parse_args():
    parser = argparse.ArgumentParser(description="Dump MaskedEmbedder embeddings for ShapeNet chairs.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/mae1000/best.pth")
    parser.add_argument("--dataroot", type=str, default="../../PSF/data/ShapeNetCore.v2.PC15k/")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    parser.add_argument("--category", type=str, default="chair")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="data/mae_embeddings/chair/train.pt")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--npoints", type=int, default=1024, help="Number of points to sample per model")
    return parser.parse_args()


def load_model(checkpoint, device):
    model = DiT_S_4(use_mae=True, mae_config_path="configs/pretrainMAE.yaml")
    ckpt = torch.load(checkpoint, map_location="cpu")
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading model from {args.checkpoint} on {args.device}")
    model = load_model(args.checkpoint, args.device)
    embedder = model.mae_embedder

    dataset = ShapeNet15kPointClouds(
        root_dir=args.dataroot,
        categories=[args.category],
        split=args.split,
        tr_sample_size=args.npoints,
        te_sample_size=args.npoints,
        normalize_per_shape=True,
        random_subsample=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True,
    )

    all_embeddings = []
    idxs = []
    cate_idxs = []
    sids = []
    mids = []
    seen = 0

    with torch.no_grad():
        for batch in loader:
            if args.max_samples > 0 and seen >= args.max_samples:
                break
            pts = batch["train_points"][:, : args.npoints].to(args.device)
            embedding = embedder(pts)
            batch_size = embedding.shape[0]
            if args.max_samples > 0:
                remaining = args.max_samples - seen
                if remaining <= 0:
                    break
                if remaining < batch_size:
                    embedding = embedding[:remaining]
                    batch_size = embedding.shape[0]
            all_embeddings.append(embedding.cpu())
            idxs.append(batch["idx"][:batch_size])
            cate_idxs.append(batch["cate_idx"][:batch_size])
            sids.extend(batch["sid"][:batch_size])
            mids.extend(batch["mid"][:batch_size])
            seen += batch_size
            print(f"\rCollected {seen} embeddings", end="", flush=True)

    embeddings = torch.cat(all_embeddings, dim=0)
    idxs = torch.cat(idxs, dim=0)
    cate_idxs = torch.cat(cate_idxs, dim=0)

    payload = {
        "embeddings": embeddings,
        "idx": idxs,
        "cate_idx": cate_idxs,
        "sid": sids,
        "mid": mids,
        "split": args.split,
        "category": args.category,
    }
    torch.save(payload, args.output)
    print(f"\nSaved {embeddings.shape[0]} embeddings to {args.output}")


if __name__ == "__main__":
    main()
