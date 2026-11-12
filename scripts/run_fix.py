"""The fix and its controls: claims C4, C5, C6.

Conditions, all evaluated on the SAME objects at the SAME yaw angles so every
comparison is paired:

| condition      | what it does                                   | CLIP passes/object |
|----------------|------------------------------------------------|--------------------|
| baseline       | upstream, unchanged                            | 10                 |
| canon_pca      | PCA yaw canonicalisation before projection     | 10                 |
| canon_extent   | bounding-box canonicalisation (ablation)       | 10                 |
| ens_yaw        | average logits over k yaw-spread copies        | 10k                |
| ens_jitter     | **the same-budget control**: average over k     | 10k                |
|                | copies that are jittered but NOT yaw-diverse   |                    |
| canon_pca_ens  | both halves together                           | 10k                |

Why `ens_jitter` is the control that matters (C5): `ens_yaw` buys its accuracy
with k times the compute, so it must be compared against something that spends
the same compute without spreading the object over yaw. Repeating an identical
render k times would be a no-op - the features are deterministic - so the
control perturbs each copy with a small random rotation about a *horizontal*
axis plus a scale jitter. Same number of forward passes, same kind of averaging,
no yaw diversity.

C6 is covered by including angle 0, the canonical pose, in every condition: if a
fix trades clean accuracy for robustness, it shows up there.

Usage:
    python scripts/run_fix.py configs/fix_modelnet40_published.yaml
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
from src.metrics import accuracy_with_ci, mean_class_accuracy, paired_bootstrap_delta, per_class_accuracy  # noqa: E402
from src.yaw import UP_AXIS, canonicalise_yaw, extent_yaw, rotate_yaw, yaw_ensemble_angles  # noqa: E402


def git_commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10)
        return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def base_transform(angle: float, up: int):
    """The test-time pose: the object as the robot happens to see it."""
    if angle == 0:
        return lambda c: c.astype(np.float32)
    return lambda c: rotate_yaw(c, float(angle), up=up).astype(np.float32)


def jitter(cloud: np.ndarray, rng: np.random.Generator, up: int) -> np.ndarray:
    """Small non-yaw perturbation: tilt about a horizontal axis plus a scale change.

    Deliberately NOT a rotation about the up axis - that is what the control is
    controlling for.
    """
    horizontal = [a for a in (0, 1, 2) if a != up]
    axis = horizontal[int(rng.integers(len(horizontal)))]
    tilted = rotate_yaw(cloud, float(rng.uniform(-10.0, 10.0)), up=axis)
    return (tilted * float(rng.uniform(0.95, 1.05))).astype(np.float32)


def run_condition(
    pipeline, points: np.ndarray, condition: str, angle: float, up: int, k: int, seed: int,
    span_deg: float = 45.0,
) -> tuple[np.ndarray, int]:
    """Return (predictions, CLIP passes per object) for one condition at one angle."""
    pose = base_transform(angle, up)
    views = pipeline.num_views

    if condition == "baseline":
        return pcv2.logits(pipeline, points, transform=pose).argmax(1), views

    if condition == "canon_pca":
        def t(c):
            out, _ = canonicalise_yaw(pose(c), up=up)
            return out.astype(np.float32)
        return pcv2.logits(pipeline, points, transform=t).argmax(1), views

    if condition == "canon_extent":
        def t(c):
            p = pose(c)
            return rotate_yaw(p, -extent_yaw(p, up=up), up=up).astype(np.float32)
        return pcv2.logits(pipeline, points, transform=t).argmax(1), views

    if condition in ("ens_yaw", "canon_pca_ens"):
        total = None
        for offset in yaw_ensemble_angles(k, span_deg=span_deg):
            def t(c, o=offset):
                p = pose(c)
                if condition == "canon_pca_ens":
                    p, _ = canonicalise_yaw(p, up=up)
                return rotate_yaw(p, float(o), up=up).astype(np.float32)
            scores = pcv2.logits(pipeline, points, transform=t)
            total = scores if total is None else total + scores
        return total.argmax(1), views * k

    if condition == "ens_jitter":
        total = None
        for i in range(k):
            rng = np.random.default_rng(seed * 1000 + i)
            def t(c, r=rng):
                return jitter(pose(c), r, up)
            scores = pcv2.logits(pipeline, points, transform=t)
            total = scores if total is None else total + scores
        return total.argmax(1), views * k

    raise ValueError(f"unknown condition {condition!r}")


def main(config_path: str) -> int:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    name = cfg.get("name") or os.path.splitext(os.path.basename(config_path))[0]
    out_dir = cfg.get("out_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    up = cfg.get("up_axis", UP_AXIS)
    seed = int(cfg.get("seed", 0))
    k = int(cfg.get("ensemble_k", 4))
    span = float(cfg.get("ensemble_span_deg", 45.0))
    angles = cfg["angles"]
    conditions = cfg["conditions"]

    points, labels, classnames = load_split(cfg["dataset"], cfg["data_root"], cfg["split"])
    if cfg.get("num_points", 1024):
        points = points[:, : cfg.get("num_points", 1024)]
    if cfg.get("limit"):
        points, labels = points[: cfg["limit"]], labels[: cfg["limit"]]
        print(f"LIMIT {cfg['limit']} - smoke test, not reportable")

    print(f"{name}: {len(points)} objects | protocol {cfg['protocol']} | k={k} over {span} deg | angles {angles}")
    pipeline = pcv2.build_pipeline(
        dataset=cfg["dataset"], classnames=classnames, backbone=cfg.get("backbone", "ViT-B/16"), protocol=cfg["protocol"]
    )

    torch.cuda.reset_peak_memory_stats()
    rows, preds_store = [], {}
    for condition in conditions:
        for angle in angles:
            t0 = time.perf_counter()
            preds, passes = run_condition(pipeline, points, condition, float(angle), up, k, seed, span)
            elapsed = time.perf_counter() - t0
            acc = accuracy_with_ci(preds, labels, seed=seed)
            macro = mean_class_accuracy(per_class_accuracy(preds, labels, n_classes=len(classnames)))
            rows.append(
                {
                    "condition": condition,
                    "angle_deg": float(angle),
                    "accuracy": acc.as_dict(),
                    "macro_accuracy": macro,
                    "clip_passes_per_object": passes,
                    "seconds": elapsed,
                    "ms_per_object": 1000 * elapsed / len(points),
                }
            )
            preds_store[f"{condition}_{int(angle)}"] = preds
            print(f"  {condition:14s} {angle:5.0f} deg -> {acc}  macro {100*macro:.2f}%  "
                  f"[{passes} passes, {1000*elapsed/len(points):.0f} ms/obj]")

    # Paired comparisons against the baseline, at the same angle.
    print("\npaired differences against baseline, same objects and same angle:")
    deltas = []
    for condition in conditions:
        if condition == "baseline":
            continue
        for angle in angles:
            a = preds_store[f"baseline_{int(angle)}"]
            b = preds_store[f"{condition}_{int(angle)}"]
            d = paired_bootstrap_delta(a, b, labels, seed=seed)
            verdict = "no" if d.lo <= 0 <= d.hi else ("better" if d.value > 0 else "WORSE")
            deltas.append({"condition": condition, "angle_deg": float(angle), "delta": d.as_dict(), "verdict": verdict})
            print(f"  {condition:14s} {angle:5.0f} deg -> {100*d.value:+6.2f} pts "
                  f"[{100*d.lo:+6.2f}, {100*d.hi:+6.2f}]  {verdict}")

    result = {
        "name": name,
        "config": cfg,
        "commit": git_commit(),
        "up_axis": up,
        "protocol": pipeline.protocol,
        "ensemble_k": k,
        "ensemble_span_deg": span,
        "n_objects": int(len(points)),
        "seed": seed,
        "classnames": classnames,
        "rows": rows,
        "paired_vs_baseline": deltas,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, f"{name}_predictions.npz"), labels=labels, **preds_store)
    print(f"\nwrote {out_dir}/{name}.json")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
