"""Decide which axis is "up" in these clouds, by looking at the renders.

`src/yaw.py` has to rotate about the object's vertical axis. If the wrong axis
is chosen, the sweep measures objects tipping over rather than turning on the
spot, and every later claim is about the wrong thing. The HDF5 files carry no
axis convention, the upstream projection rotates about the first Euler axis, and
PointNet's own augmentation rotates about Y - so this is settled by looking, not
by assuming.

Output: results/figures/up_axis_check.png. One row per candidate axis, one
column per angle. The correct axis is the one where the object turns while
staying upright.

Usage:
    python scripts/check_up_axis.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import pcv2  # noqa: E402
from src.data import load_modelnet40  # noqa: E402
from src.yaw import rotate_yaw  # noqa: E402

ANGLES = [0, 45, 90, 135]
AXES = {0: "x", 1: "y", 2: "z"}
# Classes whose upright pose is unmistakable in a silhouette.
WANTED = ["chair", "person", "airplane", "bottle"]


def main() -> int:
    root = "data/modelnet40_ply_hdf5_2048"
    points, labels, classnames = load_modelnet40(root, "test")
    points = points[:, :1024]

    chosen = []
    for name in WANTED:
        if name not in classnames:
            continue
        idx = int(np.flatnonzero(labels == classnames.index(name))[0])
        chosen.append((name, points[idx]))
    print("objects:", ", ".join(n for n, _ in chosen))

    pipeline = pcv2.build_pipeline("modelnet40", classnames, protocol=pcv2.PROTOCOL_PLAIN)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(AXES) * len(chosen)
    fig, axes = plt.subplots(n_rows, len(ANGLES), figsize=(2.1 * len(ANGLES), 2.1 * n_rows))

    row = 0
    for name, cloud in chosen:
        for axis, axis_name in AXES.items():
            for col, angle in enumerate(ANGLES):
                rotated = rotate_yaw(cloud, float(angle), up=axis).astype(np.float32)
                batch = torch.from_numpy(rotated[None]).float().cuda()
                with torch.no_grad():
                    img = pcv2.render(pipeline, batch)
                # View 0 only: the first of the ten viewpoints.
                frame = img[0, 0].detach().cpu().numpy()
                ax = axes[row, col]
                ax.imshow(frame, cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(f"{name}\nup={axis_name}", fontsize=8)
                if row == 0:
                    ax.set_title(f"{angle}°", fontsize=9)
            row += 1

    fig.suptitle(
        "Which axis is up? The right one turns the object without tipping it over.\n"
        "View 0 of the upstream 10-view projection, ModelNet40 test objects, 1024 points.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs("results/figures", exist_ok=True)
    out = "results/figures/up_axis_check.png"
    fig.savefig(out, dpi=110)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
