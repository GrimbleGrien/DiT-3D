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

from utils.embed_diffusion import DenoiserMLP, DenoiserTransformer, GaussianDiffusion1D, get_betas


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
    parser.add_argument("--beta_end", type=float, default=0.01)
    parser.add_argument("--schedule_type", type=str, default="linear")
    parser.add_argument("--output_dir", type=str, default="outputs/embed_diffusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--time_dim", type=int, default=256)
    parser.add_argument("--model_type", type=str, default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--token_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["none", "cosine", "step"])
    parser.add_argument("--lr_step_size", type=int, default=100)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="If >0, train on a random subset of this many embeddings (overfit test)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    embeddings = np.load(args.emb_path)
    if args.max_samples and args.max_samples > 0:
        perm = np.random.permutation(len(embeddings))
        embeddings = embeddings[perm[: args.max_samples]]
    emb_tensor = torch.from_numpy(embeddings).float()
    dataset = TensorDataset(emb_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    embed_dim = emb_tensor.shape[1]
    if args.model_type == "mlp":
        model = DenoiserMLP(embed_dim=embed_dim, hidden_dim=args.hidden_dim, time_dim=args.time_dim).to(device)
    else:
        model = DenoiserTransformer(
            embed_dim=embed_dim,
            token_dim=args.token_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            time_dim=args.time_dim,
            dropout=args.dropout,
        ).to(device)

    betas = get_betas(args.schedule_type, args.beta_start, args.beta_end, args.time_num)
    diffusion = GaussianDiffusion1D(betas)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    if args.use_ema:
        if args.model_type == "mlp":
            ema = DenoiserMLP(embed_dim=embed_dim, hidden_dim=args.hidden_dim, time_dim=args.time_dim).to(device)
        else:
            ema = DenoiserTransformer(
                embed_dim=embed_dim,
                token_dim=args.token_dim,
                depth=args.depth,
                num_heads=args.num_heads,
                time_dim=args.time_dim,
                dropout=args.dropout,
            ).to(device)
        ema.load_state_dict(model.state_dict())
        for p in ema.parameters():
            p.requires_grad = False
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    else:
        scheduler = None

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = None

    config = {
        "embed_dim": embed_dim,
        "hidden_dim": args.hidden_dim,
        "time_dim": args.time_dim,
        "model_type": args.model_type,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "token_dim": args.token_dim,
        "dropout": args.dropout,
        "time_num": args.time_num,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "schedule_type": args.schedule_type,
        "use_ema": args.use_ema,
        "ema_decay": args.ema_decay,
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
            if args.use_ema:
                with torch.no_grad():
                    for ema_p, p in zip(ema.parameters(), model.parameters()):
                        ema_p.mul_(args.ema_decay).add_(p, alpha=1.0 - args.ema_decay)
            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else 0.0
        save_checkpoint(os.path.join(args.output_dir, "latest.pth"), model, optimizer, epoch, avg_loss, config)
        if args.use_ema:
            save_checkpoint(os.path.join(args.output_dir, "latest_ema.pth"), ema, optimizer, epoch, avg_loss, config)
        if best_loss is None or avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(os.path.join(args.output_dir, "best.pth"), model, optimizer, epoch, avg_loss, config)
            if args.use_ema:
                save_checkpoint(os.path.join(args.output_dir, "best_ema.pth"), ema, optimizer, epoch, avg_loss, config)
        if scheduler is not None:
            scheduler.step()
        print(f"[{epoch+1:03d}/{args.epochs:03d}] loss={avg_loss:.6f}")


if __name__ == "__main__":
    main()
