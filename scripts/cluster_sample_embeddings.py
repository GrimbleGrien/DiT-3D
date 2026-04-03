import argparse
import os
import numpy as np


def kmeans(x, k, max_iters=50, seed=42):
    rng = np.random.RandomState(seed)
    n = x.shape[0]
    # init centers
    init_idx = rng.choice(n, size=k, replace=False)
    centers = x[init_idx].copy()
    labels = np.zeros(n, dtype=np.int64)

    for _ in range(max_iters):
        # assign
        dists = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # update
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centers[i] = x[mask].mean(axis=0)
            else:
                # reinit empty cluster
                centers[i] = x[rng.randint(0, n)]
    return centers, labels


def select_balanced(x, centers, labels, num_select, seed=42):
    rng = np.random.RandomState(seed)
    k = centers.shape[0]
    # compute per-sample distance to its center
    dists = ((x - centers[labels]) ** 2).sum(axis=1)
    per_cluster = {i: np.where(labels == i)[0] for i in range(k)}

    selected = []
    # one per cluster if possible
    for i in range(k):
        idxs = per_cluster[i]
        if idxs.size == 0:
            continue
        best = idxs[np.argmin(dists[idxs])]
        selected.append(best)

    if len(selected) >= num_select:
        return np.array(selected[:num_select], dtype=np.int64)

    # fill remaining by round-robin from clusters (closest first)
    cluster_lists = []
    for i in range(k):
        idxs = per_cluster[i]
        if idxs.size == 0:
            continue
        order = idxs[np.argsort(dists[idxs])]
        cluster_lists.append(list(order))

    ptr = 0
    while len(selected) < num_select and cluster_lists:
        lst = cluster_lists[ptr % len(cluster_lists)]
        if lst:
            cand = lst.pop(0)
            if cand not in selected:
                selected.append(cand)
        else:
            cluster_lists.pop(ptr % len(cluster_lists))
            ptr -= 1
        ptr += 1
        if ptr > 100000:
            break
    # if still short, random fill
    if len(selected) < num_select:
        remaining = [i for i in range(x.shape[0]) if i not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: (num_select - len(selected))])
    return np.array(selected, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Cluster generated embeddings and select balanced subset")
    parser.add_argument("--input", required=True, type=str, help="Path to samples.npy")
    parser.add_argument("--num_clusters", type=int, default=25)
    parser.add_argument("--num_select", type=int, default=25)
    parser.add_argument("--max_iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="outputs/embed_diffusion/samples_kmeans.npy")
    parser.add_argument("--save_indices", action="store_true")
    args = parser.parse_args()

    x = np.load(args.input)
    if x.shape[0] < args.num_clusters:
        raise ValueError("num_clusters cannot exceed number of samples")

    centers, labels = kmeans(x, args.num_clusters, max_iters=args.max_iters, seed=args.seed)
    sel_idx = select_balanced(x, centers, labels, args.num_select, seed=args.seed)
    sel = x[sel_idx]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, sel)
    if args.save_indices:
        np.save(os.path.splitext(args.output_path)[0] + "_indices.npy", sel_idx)
    print(f"Saved {sel.shape[0]} embeddings to {args.output_path}")


if __name__ == "__main__":
    main()
