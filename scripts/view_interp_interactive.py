import argparse
import os
import numpy as np


def load_pointclouds(path):
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {arr.shape}")

    # Support both [K, 3, N] and [K, N, 3]
    if arr.shape[1] == 3:
        pcs = np.transpose(arr, (0, 2, 1))
    elif arr.shape[2] == 3:
        pcs = arr
    else:
        raise ValueError(
            f"Could not infer point-cloud format from shape {arr.shape}. "
            "Expected [K,3,N] or [K,N,3]."
        )
    return pcs


def main():
    parser = argparse.ArgumentParser(description="Interactive viewer for interpolation point clouds (.npy)")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to .npy file (e.g., outputs/interpol_21/interp_samples_lerp.npy)",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index for viewing")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError(
            "open3d is required for interactive visualization. "
            "Install it in your environment (e.g., pip install open3d)."
        ) from exc

    pcs = load_pointclouds(args.input)
    start = max(0, min(args.start, len(pcs) - 1))

    print(f"Loaded {len(pcs)} samples from: {args.input}")
    print("Close the viewer window to continue to the next sample. Press Ctrl+C to stop.")

    for idx in range(start, len(pcs)):
        pts = pcs[idx].astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        print(f"Showing sample {idx + 1}/{len(pcs)}")
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=f"Interpolation sample {idx + 1}/{len(pcs)}",
        )


if __name__ == "__main__":
    main()
