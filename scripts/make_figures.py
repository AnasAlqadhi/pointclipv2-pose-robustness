"""The remaining report figures: backbones, the fix summary, and the failure gallery.

Run after the experiments:
    python scripts/make_figures.py

Design rules kept from scripts/plot_yaw_sweep.py: one y axis, identity never by
colour alone, bootstrap intervals drawn wherever an accuracy is drawn, recessive
grid and axes, text in ink rather than series colour.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS, FIGURES = "results", "results/figures"
INK, MUTED, GRID = "#222222", "#6b6b6b", "#d9d9d9"
# Okabe-Ito, separable under the common colour-vision deficiencies.
SERIES_COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def load(name):
    p = os.path.join(RESULTS, f"{name}.json")
    if not os.path.exists(p):
        print(f"missing {p} - skipping")
        return None
    with open(p) as f:
        return json.load(f)


def figure_backbones():
    runs = [
        ("yaw_sweep_modelnet40_rn50", "RN50"),
        ("yaw_sweep_modelnet40_rn101", "RN101"),
        ("yaw_sweep_modelnet40_vit_b32", "ViT-B/32"),
        ("yaw_sweep_modelnet40_published", "ViT-B/16"),
    ]
    loaded = [(load(n), label) for n, label in runs]
    loaded = [(r, l) for r, l in loaded if r]
    if not loaded:
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    style(ax)
    for tick in range(0, 360, 90):
        ax.axvline(tick, color="#c4c4c4", linewidth=0.9, linestyle=":", zorder=1)
    for (run, label), colour in zip(loaded, SERIES_COLOURS):
        a = np.array([r["angle_deg"] for r in run["angles"]])
        v = np.array([100 * r["accuracy"]["value"] for r in run["angles"]])
        lo = np.array([100 * r["accuracy"]["lo"] for r in run["angles"]])
        hi = np.array([100 * r["accuracy"]["hi"] for r in run["angles"]])
        ax.fill_between(a, lo, hi, color=colour, alpha=0.13, linewidth=0)
        ax.plot(a, v, color=colour, linewidth=2, marker="o", markersize=4)
        ax.annotate(label, xy=(a[-1], v[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center")
    ax.set_xticks(range(0, 360, 30))
    ax.set_xlim(-8, 395)
    ax.set_xlabel("object yaw applied before projection (degrees)", color=MUTED, fontsize=9)
    ax.set_ylabel("top-1 accuracy (%)", color=MUTED, fontsize=9)
    ax.set_title(
        "Every backbone shows the same pose sensitivity\n"
        "ModelNet40, published protocol; dotted lines at 0/90/180/270 degrees",
        color=INK, fontsize=10, loc="left")
    ax.legend(handles=[plt.Line2D([], [], color=c, linewidth=2, label=l)
                       for (_, l), c in zip(loaded, SERIES_COLOURS)],
              frameon=False, fontsize=9, loc="lower right", ncol=2)
    fig.tight_layout()
    out = os.path.join(FIGURES, "backbones.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)


def figure_fix_summary():
    run = load("fix_modelnet40_published")
    if not run:
        return
    labels = {
        "baseline": "baseline",
        "canon_extent": "canonicalise (bbox)",
        "canon_pca": "canonicalise (PCA)",
        "ens_yaw": "yaw ensemble",
        "ens_jitter": "jitter ensemble (control)",
        "canon_pca_ens": "canonicalise + ensemble",
    }
    order = ["baseline", "canon_extent", "canon_pca", "ens_yaw", "ens_jitter", "canon_pca_ens"]
    angles = sorted({r["angle_deg"] for r in run["rows"]})

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    style(ax)
    width = 0.13
    xs = np.arange(len(angles))
    for i, cond in enumerate(order):
        vals, los, his = [], [], []
        for ang in angles:
            row = next(r for r in run["rows"] if r["condition"] == cond and r["angle_deg"] == ang)
            vals.append(100 * row["accuracy"]["value"])
            los.append(100 * row["accuracy"]["lo"])
            his.append(100 * row["accuracy"]["hi"])
        vals, los, his = np.array(vals), np.array(los), np.array(his)
        colour = "#6b6b6b" if cond == "baseline" else SERIES_COLOURS[(i - 1) % len(SERIES_COLOURS)]
        hatch = "//" if "jitter" in cond else None
        ax.bar(xs + (i - 2.5) * width, vals, width * 0.92, color=colour, alpha=0.9,
               edgecolor="white", linewidth=1.0, hatch=hatch, label=labels[cond])
        ax.errorbar(xs + (i - 2.5) * width, vals, yerr=[vals - los, his - vals],
                    fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, alpha=0.55)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(a)}°" for a in angles])
    ax.set_ylim(45, 68)
    ax.set_xlabel("object yaw", color=MUTED, fontsize=9)
    ax.set_ylabel("top-1 accuracy (%)", color=MUTED, fontsize=9)
    ax.set_title(
        "Canonicalisation trades the canonical pose for pose independence\n"
        "ModelNet40, published protocol; hatched bar is the same-compute control; bars carry 95% intervals",
        color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    fig.tight_layout()
    out = os.path.join(FIGURES, "fix_summary.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)


def figure_failures():
    """Objects the rotation breaks, rendered, next to ones it does not.

    The brief asks where failure begins; a table of per-class numbers says which
    classes break but not what the model is looking at when they do.
    """
    import torch

    from src import pcv2
    from src.data import load_modelnet40
    from src.yaw import UP_AXIS, rotate_yaw

    sweep = load("yaw_sweep_modelnet40_published")
    if not sweep:
        return
    rows = sweep["angles"]
    classnames = sweep["classnames"]
    at0 = next(r for r in rows if r["angle_deg"] == 0)["per_class_accuracy"]
    rot = {c: float(np.mean([r["per_class_accuracy"][c] for r in rows if r["angle_deg"] != 0]))
           for c in classnames}
    drop = {c: at0[c] - rot[c] for c in classnames}
    worst = sorted(classnames, key=lambda c: -drop[c])[:3]
    stable = sorted(classnames, key=lambda c: abs(drop[c]))[:2]
    chosen = worst + stable

    points, labels, _ = load_modelnet40("data/modelnet40_ply_hdf5_2048", "test")
    points = points[:, :1024]
    pipeline = pcv2.build_pipeline("modelnet40", classnames, protocol=pcv2.PROTOCOL_PUBLISHED)

    angles = [0, 30, 60, 90]
    fig, axes = plt.subplots(len(chosen), len(angles), figsize=(2.0 * len(angles), 2.0 * len(chosen)))
    for r, cname in enumerate(chosen):
        idx = int(np.flatnonzero(labels == classnames.index(cname))[0])
        for c, ang in enumerate(angles):
            cloud = rotate_yaw(points[idx], float(ang), up=UP_AXIS).astype(np.float32)
            with torch.no_grad():
                img = pcv2.render(pipeline, torch.from_numpy(cloud[None]).float().cuda())
            ax = axes[r, c]
            ax.imshow(img[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                tag = "breaks" if cname in worst else "stable"
                ax.set_ylabel(f"{cname.replace('_', ' ')}\n{drop[cname] * 100:+.0f} pts ({tag})", fontsize=8)
            if r == 0:
                ax.set_title(f"{ang}°", fontsize=9)
    fig.suptitle("What the model sees as the object turns\nview 0 of 10; top three classes break, bottom two do not",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIGURES, "failure_gallery.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def main():
    os.makedirs(FIGURES, exist_ok=True)
    figure_backbones()
    figure_fix_summary()
    try:
        figure_failures()
    except Exception as exc:  # the gallery needs a GPU; the other two do not
        print(f"failure gallery skipped: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
