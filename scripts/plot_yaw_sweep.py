"""Figures for the yaw sweep: the curve (C2) and the per-class damage (C3).

Design choices, so they are not re-litigated later:
- two series only (published and plain protocol), drawn in an Okabe-Ito pair
  that stays separable under the common colour-vision deficiencies, each one
  direct-labelled as well as in the legend, so identity never rests on colour;
- one y axis, never two;
- bootstrap intervals drawn as bands, because a curve without them invites
  reading noise as structure;
- the per-class figure is a single hue by magnitude, with the rotationally
  symmetric classes marked by hatching rather than by a second colour.

Usage:
    python scripts/plot_yaw_sweep.py
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

RESULTS = "results"
FIGURES = "results/figures"

SERIES = [
    ("yaw_sweep_modelnet40_published", "published protocol", "#0072B2"),
    ("yaw_sweep_modelnet40_plain", "plain protocol", "#D55E00"),
]

# Classes whose canonical pose is (near) invariant to a turn about the vertical
# axis. Fixed a priori from the class names, before looking at any result.
SYMMETRIC = {"bottle", "bowl", "cone", "cup", "flower_pot", "lamp", "vase", "tent", "stool"}

INK = "#222222"
MUTED = "#6b6b6b"
GRID = "#d9d9d9"


def style(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def load(name: str) -> dict | None:
    path = os.path.join(RESULTS, f"{name}.json")
    if not os.path.exists(path):
        print(f"missing {path} - skipping")
        return None
    with open(path) as f:
        return json.load(f)


def figure_curve(runs: list[tuple[dict, str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    style(ax)
    for run, label, colour in runs:
        angles = np.array([r["angle_deg"] for r in run["angles"]])
        value = np.array([100 * r["accuracy"]["value"] for r in run["angles"]])
        lo = np.array([100 * r["accuracy"]["lo"] for r in run["angles"]])
        hi = np.array([100 * r["accuracy"]["hi"] for r in run["angles"]])
        ax.fill_between(angles, lo, hi, color=colour, alpha=0.16, linewidth=0)
        ax.plot(angles, value, color=colour, linewidth=2, marker="o", markersize=5)
        ax.annotate(
            label,
            xy=(angles[-1], value[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK,
            fontsize=9,
            va="center",
        )
        ax.scatter([0], [value[0]], s=70, facecolor="white", edgecolor=colour, zorder=5, linewidth=2)

    ax.set_xlabel("object yaw applied before projection (degrees)", color=MUTED, fontsize=9)
    ax.set_ylabel("top-1 accuracy (%)", color=MUTED, fontsize=9)
    ax.set_xticks(np.arange(0, 360, 30))
    ax.set_xlim(-8, 395)
    ax.set_ylim(0, 72)
    ax.set_title(
        "PointCLIP V2 on ModelNet40: accuracy against object yaw\n"
        "95% bootstrap bands; the ringed point is the canonical pose",
        color=INK,
        fontsize=10,
        loc="left",
    )
    ax.legend(
        handles=[plt.Line2D([], [], color=c, linewidth=2, label=l) for _, l, c in runs],
        frameon=False,
        fontsize=9,
        loc="lower right",
    )
    fig.tight_layout()
    out = os.path.join(FIGURES, "yaw_sweep.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)


def figure_per_class(run: dict, label: str) -> None:
    classnames = run["classnames"]
    rows = run["angles"]
    at_zero = next(r for r in rows if r["angle_deg"] == 0)["per_class_accuracy"]
    rotated = {
        name: float(np.mean([r["per_class_accuracy"][name] for r in rows if r["angle_deg"] != 0]))
        for name in classnames
    }
    drop = {name: 100 * (at_zero[name] - rotated[name]) for name in classnames}
    order = sorted(classnames, key=lambda n: -drop[n])
    values = [drop[n] for n in order]

    vmax = max(max(values), 1e-6)
    fig, ax = plt.subplots(figsize=(7.6, 8.4))
    style(ax)
    ax.grid(axis="y", visible=False)

    for i, name in enumerate(order):
        v = drop[name]
        shade = 0.25 + 0.55 * (abs(v) / vmax)
        colour = plt.matplotlib.colors.to_rgba("#0072B2", alpha=min(shade, 0.95))
        ax.barh(
            i,
            v,
            height=0.72,
            color=colour,
            edgecolor="white",
            linewidth=1.0,
            hatch="//" if name in SYMMETRIC else None,
        )
        ax.text(
            v + (0.6 if v >= 0 else -0.6),
            i,
            f"{v:+.0f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
            color=MUTED,
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([n.replace("_", " ") for n in order], fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_xlabel("accuracy lost when the object is rotated (percentage points)", color=MUTED, fontsize=9)
    ax.set_title(
        f"Where the damage lands - {label}\n"
        "canonical pose minus the mean over the other 11 yaw angles; hatched = rotationally symmetric class",
        color=INK,
        fontsize=10,
        loc="left",
    )
    ax.legend(
        handles=[
            plt.matplotlib.patches.Patch(facecolor="#0072B2", alpha=0.6, hatch="//", label="symmetric class"),
            plt.matplotlib.patches.Patch(facecolor="#0072B2", alpha=0.6, label="oriented class"),
        ],
        frameon=False,
        fontsize=9,
        loc="lower right",
    )
    fig.tight_layout()
    out = os.path.join(FIGURES, "yaw_per_class.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)

    sym = [drop[n] for n in classnames if n in SYMMETRIC]
    ori = [drop[n] for n in classnames if n not in SYMMETRIC]
    print(
        f"mean drop: symmetric classes {np.mean(sym):+.2f} pts (n={len(sym)}), "
        f"oriented classes {np.mean(ori):+.2f} pts (n={len(ori)})"
    )


def figure_fine(run: dict) -> None:
    """The 5-degree sweep, with the view azimuths marked.

    The claim this figure has to support is that the teeth sit at multiples of
    45 degrees - the union of the projection's two groups of four azimuthal
    views - so the view positions are drawn on the axis and the reader can
    check the alignment instead of taking it on trust.
    """
    angles = np.array([r["angle_deg"] for r in run["angles"]])
    value = np.array([100 * r["accuracy"]["value"] for r in run["angles"]])
    lo = np.array([100 * r["accuracy"]["lo"] for r in run["angles"]])
    hi = np.array([100 * r["accuracy"]["hi"] for r in run["angles"]])

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    style(ax)
    for tick in range(0, 360, 45):
        ax.axvline(tick, color="#9c9c9c", linewidth=0.9, linestyle=":", zorder=1)
    ax.fill_between(angles, lo, hi, color="#0072B2", alpha=0.16, linewidth=0)
    ax.plot(angles, value, color="#0072B2", linewidth=1.8, zorder=3)
    ax.scatter(angles, value, s=12, color="#0072B2", zorder=4)
    ax.set_xticks(range(0, 360, 45))
    ax.set_xlim(-5, 360)
    ax.set_xlabel("object yaw applied before projection (degrees)", color=MUTED, fontsize=9)
    ax.set_ylabel("top-1 accuracy (%)", color=MUTED, fontsize=9)
    ax.set_title(
        "5-degree yaw sweep, published protocol\n"
        "dotted lines mark the projection's eight azimuthal viewpoints, every 45 degrees",
        color=INK,
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    out = os.path.join(FIGURES, "yaw_sweep_fine.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)

    on = value[np.isclose(angles % 45, 0)]
    off = value[~np.isclose(angles % 45, 0)]
    print(f"on a view azimuth: {on.mean():.2f}% (n={on.size})   between: {off.mean():.2f}% (n={off.size})")


def main() -> int:
    os.makedirs(FIGURES, exist_ok=True)
    fine = load("yaw_sweep_modelnet40_published_fine")
    if fine:
        figure_fine(fine)
    runs = [(run, label, colour) for name, label, colour in SERIES if (run := load(name))]
    if not runs:
        print("no sweep results found - run scripts/run_yaw_sweep.py first")
        return 1
    figure_curve(runs)
    figure_per_class(runs[0][0], runs[0][1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
