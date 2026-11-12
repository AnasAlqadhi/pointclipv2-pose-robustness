"""Verify the ModelNet40 HDF5 download before any experiment runs.

Checks what the upstream loader will rely on: the file list, the class names, the
object and point counts, the label range, and that the clouds are already
normalised the way the projection code expects. Reports the per-class test counts
too, because the per-class analysis for claim C3 needs them and an imbalanced
split changes how the mean should be read.

Usage:
    python scripts/check_modelnet40.py data/modelnet40_ply_hdf5_2048
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np


def load_split(root: str, list_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a split the way upstream does: a text file listing .h5 files."""
    list_path = os.path.join(root, list_name)
    with open(list_path) as f:
        rel = [line.strip() for line in f if line.strip()]
    data, labels = [], []
    for entry in rel:
        # The lists ship with paths like "data/modelnet40_ply_hdf5_2048/ply_data_test0.h5",
        # relative to the upstream working directory. Resolve by basename against root
        # so this check works wherever the data actually sits.
        path = os.path.join(root, os.path.basename(entry))
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} (listed in {list_name} as {entry})")
        with h5py.File(path, "r") as h5:
            data.append(np.asarray(h5["data"]))
            labels.append(np.asarray(h5["label"]).ravel())
    return np.concatenate(data), np.concatenate(labels)


def main(root: str) -> int:
    print(f"root: {os.path.abspath(root)}")

    names_path = os.path.join(root, "shape_names.txt")
    with open(names_path) as f:
        classnames = [line.strip() for line in f if line.strip()]
    print(f"classes: {len(classnames)}")

    test_x, test_y = load_split(root, "test_files.txt")
    train_x, train_y = load_split(root, "train_files.txt")

    print(f"test  objects: {test_x.shape[0]}  points/object: {test_x.shape[1]}  dims: {test_x.shape[2]}")
    print(f"train objects: {train_x.shape[0]}")
    print(f"label range: {int(test_y.min())}..{int(test_y.max())}  dtype {test_x.dtype}")

    # Geometry sanity: the clouds should be roughly unit-sphere normalised and centred.
    radii = np.linalg.norm(test_x, axis=2)
    centre = np.abs(test_x.mean(axis=1)).max()
    print(f"max radius: {radii.max():.3f}   mean radius: {radii.mean():.3f}   max |centroid|: {centre:.3f}")

    counts = np.bincount(test_y, minlength=len(classnames))
    order = np.argsort(-counts)
    print("test objects per class (most to fewest):")
    print("  " + ", ".join(f"{classnames[i]}={counts[i]}" for i in order))
    print(f"imbalance: largest {counts.max()}, smallest {counts.min()}")

    ok = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal ok
        print(f"  [{'ok' if condition else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
        ok = ok and condition

    print("checks:")
    check("40 class names", len(classnames) == 40, f"got {len(classnames)}")
    check("2468 test objects", test_x.shape[0] == 2468, f"got {test_x.shape[0]}")
    check("9840 train objects", train_x.shape[0] == 9840, f"got {train_x.shape[0]}")
    check("2048 points per object", test_x.shape[1] == 2048, f"got {test_x.shape[1]}")
    check("xyz only", test_x.shape[2] == 3, f"got {test_x.shape[2]}")
    check("labels cover all classes", len(np.unique(test_y)) == 40, f"got {len(np.unique(test_y))}")
    check("no empty class in the test split", counts.min() > 0)
    check("clouds are finite", bool(np.isfinite(test_x).all()))
    check("roughly unit-sphere normalised", radii.max() <= 1.5, f"max radius {radii.max():.3f}")

    print("RESULT:", "data looks correct" if ok else "SOMETHING IS WRONG - do not run experiments yet")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
