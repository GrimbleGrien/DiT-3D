import argparse
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def load_pointclouds(path):
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    # Support [K,3,N] and [K,N,3]
    if arr.shape[1] == 3:
        pcs = np.transpose(arr, (0, 2, 1))
    elif arr.shape[2] == 3:
        pcs = arr
    else:
        raise ValueError(f"Unknown point-cloud shape {arr.shape}; expected [K,3,N] or [K,N,3].")
    return pcs


def render_frame(points, elev, azim, point_size, color, dpi=100, figsize=(5, 5)):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=color, s=point_size)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout(pad=0)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return Image.fromarray(buf)


def blend_transition(img_a, img_b, n):
    # n transition frames between two images (excluding endpoints)
    if n <= 0:
        return []
    frames = []
    for i in range(1, n + 1):
        alpha = i / float(n + 1)
        frames.append(Image.blend(img_a, img_b, alpha))
    return frames


def main():
    parser = argparse.ArgumentParser(description="Create rotating interpolation GIF from point-cloud .npy")
    parser.add_argument("--input", required=True, type=str, help="Path to interpolation .npy")
    parser.add_argument("--output", required=True, type=str, help="Output .gif path")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--rotate_frames", type=int, default=24, help="Frames per sample rotation")
    parser.add_argument("--transition_frames", type=int, default=8, help="Crossfade frames between samples")
    parser.add_argument("--max_samples", type=int, default=0, help="If >0, use only first N samples")
    parser.add_argument("--elev", type=float, default=30.0)
    parser.add_argument("--azim_start", type=float, default=225.0)
    parser.add_argument("--point_size", type=float, default=3.0)
    parser.add_argument("--color", type=str, default="limegreen")
    args = parser.parse_args()

    pcs = load_pointclouds(args.input)
    if args.max_samples and args.max_samples > 0:
        pcs = pcs[: args.max_samples]

    if len(pcs) == 0:
        raise ValueError("No point clouds found to render.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    all_frames = []
    start_views = []

    # Pre-render start-view frames for clean transitions
    for pts in pcs:
        start_views.append(
            render_frame(
                pts,
                elev=args.elev,
                azim=args.azim_start,
                point_size=args.point_size,
                color=args.color,
            )
        )

    for i, pts in enumerate(pcs):
        # Rotate current sample
        azims = np.linspace(args.azim_start, args.azim_start + 360.0, args.rotate_frames, endpoint=False)
        for az in azims:
            all_frames.append(
                render_frame(
                    pts,
                    elev=args.elev,
                    azim=float(az),
                    point_size=args.point_size,
                    color=args.color,
                )
            )

        # Return to canonical angle before transition
        all_frames.append(start_views[i])

        # Crossfade to next sample at canonical angle
        if i < len(pcs) - 1:
            all_frames.extend(blend_transition(start_views[i], start_views[i + 1], args.transition_frames))

    duration_ms = int(round(1000.0 / max(1, args.fps)))
    all_frames[0].save(
        args.output,
        save_all=True,
        append_images=all_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

    print(f"Saved GIF: {args.output}")
    print(f"Frames: {len(all_frames)} | FPS: {args.fps}")


if __name__ == "__main__":
    main()
