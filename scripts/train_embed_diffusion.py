import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.embed_diffusion import DenoiserMLP, GaussianDiffusion1D, get_betas


def save_checkpoint(path, model, optimizer, epoch, loss, config):
    save_dict = {
        "epoch": epoch,
        "loss": loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
    }
    torch.save(save_dict, path)


def main():
    parser = argparse.ArgumentParser(description="Train diffusion model on MAE embeddings")
    parser.add_argument("--emb_path", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--time_num", type=int, default=200)
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--schedule_type", type=str, default="linear")
    parser.add_argument("--output_dir", type=str, default="outputs/embed_diffusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--time_dim", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    embeddings = np.load(args.emb_path)
    emb_tensor = torch.from_numpy(embeddings).float()
    dataset = TensorDataset(emb_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    embed_dim = emb_tensor.shape[1]
    model = DenoiserMLP(embed_dim=embed_dim, hidden_dim=args.hidden_dim, time_dim=args.time_dim).to(device)

    betas = get_betas(args.schedule_type, args.beta_start, args.beta_end, args.time_num)
    diffusion = GaussianDiffusion1D(betas)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = None

    config = {
        "embed_dim": embed_dim,
        "hidden_dim": args.hidden_dim,
        "time_dim": args.time_dim,
        "time_num": args.time_num,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "schedule_type": args.schedule_type,
    }

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            t = torch.randint(0, diffusion.num_timesteps, size=(x_batch.shape[0],), device=device)
            loss = diffusion.p_losses(model, x_batch, t).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else 0.0
        save_checkpoint(os.path.join(args.output_dir, "latest.pth"), model, optimizer, epoch, avg_loss, config)
        if best_loss is None or avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(os.path.join(args.output_dir, "best.pth"), model, optimizer, epoch, avg_loss, config)
        print(f"[{epoch+1:03d}/{args.epochs:03d}] loss={avg_loss:.6f}")


if __name__ == "__main__":
    main()
