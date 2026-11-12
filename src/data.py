"""Dataset loading, matching the upstream preprocessing exactly.

The clouds are returned untouched, as they come out of the HDF5 files, because
the reproduction depends on the preprocessing being identical to the one the
published numbers were measured on. Any transform (yaw rotation,
canonicalisation) is applied later, explicitly, by the experiment script.
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np


def load_modelnet40(root: str, split: str = "test") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (points [N, 2048, 3], labels [N], classnames [40])."""
    list_name = f"{split}_files.txt"
    with open(os.path.join(root, list_name)) as f:
        entries = [line.strip() for line in f if line.strip()]
    data, labels = [], []
    for entry in entries:
        path = os.path.join(root, os.path.basename(entry))
        with h5py.File(path, "r") as h5:
            data.append(np.asarray(h5["data"], dtype=np.float32))
            labels.append(np.asarray(h5["label"]).ravel().astype(np.int64))
    with open(os.path.join(root, "shape_names.txt")) as f:
        classnames = [line.strip() for line in f if line.strip()]
    return np.concatenate(data), np.concatenate(labels), classnames


# The upstream loader's class order for ScanObjectNN, from trainers/post_search.py.
SCANOBJECTNN_CLASSNAMES = [
    "bag", "bin", "box", "cabinet", "chair", "desk", "display", "door",
    "shelf", "table", "bed", "pillow", "sink", "sofa", "toilet",
]

SCANOBJECTNN_SPLITS = {
    # name used in our configs -> file under data/scanobjectnn
    "obj_bg": "main_split/test_objectdataset.h5",
    "obj_only": "main_split_nobg/test_objectdataset.h5",
    "pb_t50_rs": "main_split/test_objectdataset_augmentedrot_scale75.h5",
}


def load_scanobjectnn(root: str, split: str = "obj_only") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (points, labels, classnames) for one ScanObjectNN test split.

    Note for the report: `pb_t50_rs` is the split whose objects are already
    rotation- and scale-perturbed by the dataset authors, so it is NOT a clean
    canonical-pose condition and must not be used as one.
    """
    if split not in SCANOBJECTNN_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SCANOBJECTNN_SPLITS)}")
    with h5py.File(os.path.join(root, SCANOBJECTNN_SPLITS[split]), "r") as h5:
        points = np.asarray(h5["data"], dtype=np.float32)
        labels = np.asarray(h5["label"]).ravel().astype(np.int64)
    return points, labels, list(SCANOBJECTNN_CLASSNAMES)


def load_split(dataset: str, root: str, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if dataset == "modelnet40":
        return load_modelnet40(root, split)
    if dataset == "scanobjectnn":
        return load_scanobjectnn(root, split)
    raise ValueError(f"unknown dataset {dataset!r}")


def modelnet40_object_names(root: str, split: str = "test") -> list[str] | None:
    """The per-object source file names, if the id2file JSONs are present.

    Useful for the failure gallery: it lets a broken object be named rather than
    referred to by index. Returns None when the files are missing.
    """
    names: list[str] = []
    i = 0
    while True:
        path = os.path.join(root, f"ply_data_{split}_{i}_id2file.json")
        if not os.path.exists(path):
            break
        with open(path) as f:
            names.extend(json.load(f))
        i += 1
    return names or None
