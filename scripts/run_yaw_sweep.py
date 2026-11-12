"""Yaw sweep: accuracy as a function of the object's rotation about the up axis.

Claims C2 (does accuracy degrade?) and C3 (is the damage concentrated in
orientation-dependent classes?).

The rotation is applied to the point cloud **before** projection, which is where
a robot's pose variation enters. Rotating the rendered image instead would be a
different experiment.

The pipeline is built once and reused across angles, so a 12-angle sweep costs
roughly 12 evaluations and no reloads.

Usage:
    python scripts/run_yaw_sweep.py configs/yaw_sweep_modelnet40_published.yaml
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import pcv2  # noqa: E402
from src.data import load_split  # noqa: E402
from src.metrics import accuracy_with_ci, mean_class_accuracy, per_class_accuracy  # noqa: E402
from src.yaw import UP_AXIS, rotate_yaw, yaw_grid  # noqa: E402


def git_commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10)
        return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def main(config_path: str) -> int:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    name = cfg.get("name") or os.path.splitext(os.path.basename(config_path))[0]
    out_dir = cfg.get("out_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    up = cfg.get("up_axis", UP_AXIS)
    seed = int(cfg.get("seed", 0))

    points, labels, classnames = load_split(cfg["dataset"], cfg["data_root"], cfg["split"])
    num_points = cfg.get("num_points", 1024)
    if num_points:
        points = points[:, :num_points]
    if cfg.get("limit"):
        points, labels = points[: cfg["limit"]], labels[: cfg["limit"]]
        print(f"LIMIT {cfg['limit']} objects - smoke test, not reportable")

    angles = cfg.get("angles") or yaw_grid(float(cfg.get("step_deg", 30.0))).tolist()
    print(f"{name}: {len(points)} objects, protocol {cfg['protocol']}, up axis {up}, {len(angles)} angles")

    pipeline = pcv2.build_pipeline(
        dataset=cfg["dataset"],
        classnames=classnames,
        backbone=cfg.get("backbone", "ViT-B/16"),
        protocol=cfg["protocol"],
    )

    rows, all_preds = [], {}
    torch.cuda.reset_peak_memory_stats()
    for angle in angles:
        transform = None if angle == 0 else (lambda c, a=angle: rotate_yaw(c, float(a), up=up).astype(np.float32))
        t0 = time.perf_counter()
        preds = pcv2.predict(pipeline, points, batch_size=cfg.get("batch_size", 16), transform=transform, progress=False)
        elapsed = time.perf_counter() - t0

        acc = accuracy_with_ci(preds, labels, seed=seed)
        per_class = per_class_accuracy(preds, labels, n_classes=len(classnames))
        macro = mean_class_accuracy(per_class)
        rows.append(
            {
                "angle_deg": float(angle),
                "accuracy": acc.as_dict(),
                "macro_accuracy": macro,
                "per_class_accuracy": {classnames[k]: v.value for k, v in per_class.items()},
                "seconds": elapsed,
            }
        )
        all_preds[f"angle_{int(angle)}"] = preds
        print(f"  {angle:5.0f} deg -> {acc}  macro {100 * macro:.2f}%  ({elapsed:.0f}s)")

    best = max(rows, key=lambda r: r["accuracy"]["value"])
    worst = min(rows, key=lambda r: r["accuracy"]["value"])
    print(
        f"\nbest {100 * best['accuracy']['value']:.2f}% at {best['angle_deg']:.0f} deg | "
        f"worst {100 * worst['accuracy']['value']:.2f}% at {worst['angle_deg']:.0f} deg | "
        f"spread {100 * (best['accuracy']['value'] - worst['accuracy']['value']):.2f} points"
    )
    mean_rotated = float(np.mean([r["accuracy"]["value"] for r in rows if r["angle_deg"] != 0]))
    at_zero = next(r for r in rows if r["angle_deg"] == 0)["accuracy"]["value"]
    print(f"canonical (0 deg) {100 * at_zero:.2f}%  vs  mean over the other angles {100 * mean_rotated:.2f}%")

    result = {
        "name": name,
        "config": cfg,
        "commit": git_commit(),
        "up_axis": up,
        "protocol": pipeline.protocol,
        "num_views": pipeline.num_views,
        "forward_passes_per_object": pipeline.forward_passes_per_object,
        "torch_scatter_shim": pipeline.shim_used,
        "n_objects": int(len(points)),
        "seed": seed,
        "classnames": classnames,
        "angles": rows,
        "accuracy_at_0": at_zero,
        "mean_accuracy_rotated": mean_rotated,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, f"{name}_predictions.npz"), labels=labels, **all_preds)
    print(f"wrote {out_dir}/{name}.json and {name}_predictions.npz")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
