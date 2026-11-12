"""Run one evaluation and write a results file. Every reported number comes from here.

Usage:
    python scripts/run_eval.py configs/baseline_modelnet40_published.yaml

The results file holds the config, the commit, the timings, the accuracy with its
bootstrap interval, the per-class table, and the per-item predictions. The
predictions are what make later comparisons *paired*, which is how the project
reports differences honestly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import pcv2, yaw as yawmod  # noqa: E402
from src.data import load_split  # noqa: E402
from src.metrics import accuracy_with_ci, mean_class_accuracy, per_class_accuracy  # noqa: E402


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def build_transform(cfg: dict):
    """The cloud transform applied before projection, from the config."""
    yaw_deg = cfg.get("yaw_deg")
    canon = cfg.get("canonicalise")
    up = cfg.get("up_axis", yawmod.UP_AXIS)

    if yaw_deg in (None, 0) and not canon:
        return None, "none"

    parts = []
    if yaw_deg:
        parts.append(f"yaw={yaw_deg}")
    if canon:
        parts.append(f"canonicalise={canon}")
    parts.append(f"up_axis={up}")

    def transform(cloud: np.ndarray) -> np.ndarray:
        out = cloud
        if yaw_deg:
            out = yawmod.rotate_yaw(out, float(yaw_deg), up=up)
        if canon == "pca":
            out, _ = yawmod.canonicalise_yaw(out, up=up)
        elif canon == "extent":
            out = yawmod.rotate_yaw(out, -yawmod.extent_yaw(out, up=up), up=up)
        elif canon:
            raise ValueError(f"unknown canonicalise {canon!r}")
        return out.astype(np.float32)

    return transform, ", ".join(parts)


def main(config_path: str) -> int:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    name = cfg.get("name") or os.path.splitext(os.path.basename(config_path))[0]
    out_dir = cfg.get("out_dir", "results")
    os.makedirs(out_dir, exist_ok=True)

    points, labels, classnames = load_split(cfg["dataset"], cfg["data_root"], cfg["split"])

    # Upstream keeps only the FIRST num_points of each cloud, by truncation, not
    # random sampling: Dassl3D/dassl/data/data_manager.py:200-201 does
    # `num_points = 1024; pointcloud = item.impath[:num_points]`. The HDF5 files
    # hold 2048 points, so feeding all of them produces denser depth maps and a
    # materially different accuracy. Default to 1024 to match the published runs.
    num_points = cfg.get("num_points", 1024)
    if num_points:
        if num_points > points.shape[1]:
            raise ValueError(f"asked for {num_points} points, clouds have {points.shape[1]}")
        points = points[:, :num_points]

    limit = cfg.get("limit")
    if limit:
        points, labels = points[:limit], labels[:limit]
        print(f"LIMIT {limit} objects - a smoke test, not a reportable number")

    print(f"{name}: {len(points)} objects, {len(classnames)} classes, protocol {cfg['protocol']}")

    t0 = time.perf_counter()
    pipeline = pcv2.build_pipeline(
        dataset=cfg["dataset"],
        classnames=classnames,
        backbone=cfg.get("backbone", "ViT-B/16"),
        protocol=cfg["protocol"],
    )
    load_s = time.perf_counter() - t0
    print(f"pipeline ready in {load_s:.1f}s | views {pipeline.num_views} | shim {pipeline.shim_used}")

    transform, transform_desc = build_transform(cfg)
    print(f"transform: {transform_desc}")

    import torch

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    preds = pcv2.predict(pipeline, points, batch_size=cfg.get("batch_size", 16), transform=transform)
    eval_s = time.perf_counter() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3

    seed = int(cfg.get("seed", 0))
    acc = accuracy_with_ci(preds, labels, seed=seed)
    per_class = per_class_accuracy(preds, labels, n_classes=len(classnames))
    macro = mean_class_accuracy(per_class)

    print(f"\noverall accuracy: {acc}")
    print(f"macro accuracy:   {100 * macro:.2f}%")
    print(f"time: {eval_s:.1f}s ({1000 * eval_s / len(points):.1f} ms/object) | peak VRAM {peak_mem_gb:.2f} GB")

    result = {
        "name": name,
        "config_path": config_path,
        "config": cfg,
        "commit": git_commit(),
        "transform": transform_desc,
        "protocol": pipeline.protocol,
        "prompts": pipeline.prompts,
        "view_weights": pipeline.view_weights.tolist(),
        "num_views": pipeline.num_views,
        "forward_passes_per_object": pipeline.forward_passes_per_object,
        "torch_scatter_shim": pipeline.shim_used,
        "n_objects": int(len(points)),
        "limit": limit,
        "seed": seed,
        "accuracy": acc.as_dict(),
        "macro_accuracy": macro,
        "per_class_accuracy": {classnames[k]: v.as_dict() for k, v in per_class.items()},
        "eval_seconds": eval_s,
        "ms_per_object": 1000 * eval_s / len(points),
        "peak_vram_gb": peak_mem_gb,
        "pipeline_load_seconds": load_s,
    }
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, f"{name}_predictions.npz"), preds=preds, labels=labels
    )
    print(f"wrote {out_dir}/{name}.json and {name}_predictions.npz")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
