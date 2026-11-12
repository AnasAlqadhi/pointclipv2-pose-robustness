"""Does the yaw ensemble's benefit depend on the span, as the mechanism predicts?

The ten viewpoints sit at azimuths 0, 45, ... 315, so the view set is closed
under a 90-degree turn of the object. The mechanism therefore predicts, before
the runs:

- span 90 and span 360 should be near no-ops: the ensemble re-renders the same
  images in a different order;
- spans below 45 should help, because they place views inside one tooth of the
  grid where the baseline has no coverage;
- the benefit should be largest at the trough angles (30, 60) and smallest at a
  peak (0).

A confirmation here is the second, independent route to the same conclusion as
the sawtooth in the yaw curve.

Usage:
    python scripts/run_span_sweep.py configs/span_sweep_modelnet40.yaml
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_fix import git_commit, run_condition  # noqa: E402
from src import pcv2  # noqa: E402
from src.data import load_split  # noqa: E402
from src.metrics import accuracy_with_ci, paired_bootstrap_delta  # noqa: E402
from src.yaw import UP_AXIS  # noqa: E402


def main(config_path: str) -> int:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    name = cfg.get("name") or os.path.splitext(os.path.basename(config_path))[0]
    out_dir = cfg.get("out_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    up = cfg.get("up_axis", UP_AXIS)
    seed = int(cfg.get("seed", 0))
    k = int(cfg.get("ensemble_k", 4))
    angles = cfg["angles"]
    spans = cfg["spans"]

    points, labels, classnames = load_split(cfg["dataset"], cfg["data_root"], cfg["split"])
    points = points[:, : cfg.get("num_points", 1024)]
    if cfg.get("limit"):
        points, labels = points[: cfg["limit"]], labels[: cfg["limit"]]
        print(f"LIMIT {cfg['limit']} - smoke test, not reportable")

    print(f"{name}: {len(points)} objects | protocol {cfg['protocol']} | k={k} | spans {spans} | angles {angles}")
    pipeline = pcv2.build_pipeline(
        dataset=cfg["dataset"], classnames=classnames, backbone=cfg.get("backbone", "ViT-B/16"), protocol=cfg["protocol"]
    )

    rows, store = [], {}
    torch.cuda.reset_peak_memory_stats()
    for angle in angles:
        base_preds, _ = run_condition(pipeline, points, "baseline", float(angle), up, k, seed)
        store[f"baseline_{int(angle)}"] = base_preds
        base_acc = accuracy_with_ci(base_preds, labels, seed=seed)
        print(f"  baseline      {angle:5.0f} deg -> {base_acc}")
        rows.append({"span_deg": None, "angle_deg": float(angle), "accuracy": base_acc.as_dict(), "delta": None})

        for span in spans:
            t0 = time.perf_counter()
            preds, passes = run_condition(pipeline, points, "ens_yaw", float(angle), up, k, seed, float(span))
            acc = accuracy_with_ci(preds, labels, seed=seed)
            d = paired_bootstrap_delta(base_preds, preds, labels, seed=seed)
            verdict = "no" if d.lo <= 0 <= d.hi else ("better" if d.value > 0 else "WORSE")
            store[f"span{span}_{int(angle)}"] = preds
            rows.append(
                {
                    "span_deg": float(span),
                    "angle_deg": float(angle),
                    "accuracy": acc.as_dict(),
                    "delta": d.as_dict(),
                    "verdict": verdict,
                    "clip_passes_per_object": passes,
                    "seconds": time.perf_counter() - t0,
                }
            )
            print(f"  span {span:6.2f} {angle:5.0f} deg -> {acc}   delta {100*d.value:+6.2f} "
                  f"[{100*d.lo:+6.2f}, {100*d.hi:+6.2f}]  {verdict}")

    result = {
        "name": name,
        "config": cfg,
        "commit": git_commit(),
        "protocol": pipeline.protocol,
        "ensemble_k": k,
        "n_objects": int(len(points)),
        "seed": seed,
        "rows": rows,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, f"{name}_predictions.npz"), labels=labels, **store)
    print(f"wrote {out_dir}/{name}.json")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
