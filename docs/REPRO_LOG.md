# Reproduction log

**Append-only.** One row per run, added when the run finishes, never edited or deleted afterwards - including
runs that failed, crashed or gave a disappointing number. This log is the evidence that the results are real,
and the report's "deviations from the source" section is written from it.

If a run was wrong, add a new row saying so and why. Do not remove the old one.

## Environment as of 2026-11-12 (dev laptop)

| | |
|---|---|
| OS | Windows 11 Home 10.0.26300 |
| GPU | NVIDIA GeForce RTX 5070 Laptop, 8151 MiB, driver 621.14, compute capability 12.0 (Blackwell) |
| Python | 3.12.10 in `project/.venv` |
| torch | **2.11.0+cu128**, CUDA build 12.8, `cuda.is_available() == True`, `sm_120` supported, 7.9 GB usable. Installed from the cu128 index 2026-11-12. Full list in `docs/pip-freeze-dev.txt` |
| torch-scatter | **not installed.** The pinned `2.0.9` predates this torch and has no matching wheel. The one call that uses it has a native replacement, with equivalence tests in `tests/test_scatter_equivalence.py`. Using it is a deviation and must be stated in the report |
| tests | 29 passed, 3 skipped (the skips are the `torch_scatter` comparisons, which need that package) |
| upstream PointCLIP_V2 commit | `17df8a269c14cb1601b596b3bc19274964fbd0d9`, dated 2025-08-12, cloned 2026-11-12 |
| ModelNet40 version | `modelnet40_ply_hdf5_2048`, 417 MB, downloaded 2026-11-12 from the **upstream authors' own Google Drive** (folder `145flu-CtXPlhJ2nrSUUe7tmUj1DTts7t`, linked from `zeroshot_cls/README.md`). Verified by `scripts/check_modelnet40.py`: 40 classes, 2,468 test / 9,840 train objects, 2,048 points, xyz only, finite, unit-sphere normalised (max radius 1.000, centroid 0.000). All checks pass |
| ModelNet40 class balance | **imbalanced**: 18 classes have 100 test objects, 3 have 86, 1 has 50, 18 have 20. So overall accuracy is weighted towards the big classes and **macro accuracy must be reported next to it** |
| ScanObjectNN version | arrived in the same Drive folder, 1.8 GB, not separately requested. Test object counts verified by reading the HDF5 files: `main_split/test_objectdataset.h5` = 581 (OBJ_BG), `main_split_nobg/test_objectdataset.h5` = 581 (OBJ_ONLY), `main_split/test_objectdataset_augmentedrot_scale75.h5` = 2,882 (PB_T50_RS). `..._scale75_newsplit.h5` is an empty 800-byte file - broken in the source, and not needed. **Licence caveat below.** |
| Data location | `project/data/`, junctioned into `third_party/PointCLIP_V2/zeroshot_cls/data` so the upstream relative paths in `test_files.txt` resolve |

Fill each of these in as it becomes true, and run `pip freeze > docs/pip-freeze-dev.txt` once the
environment is complete.

## Published numbers (the paper's, not ours)

From PointCLIP V2, arXiv:2211.11682. **Every one of these must be verified against Table 2 of the PDF
before it is cited** - the secondary sources I found disagree with each other, so these are marked
unverified until someone reads the table.

**The paper and its own repository do not agree.** `zeroshot_cls/README.md` at commit `17df8a26` gives a
table for four backbones and lists **ModelNet40 ViT-B/16 = 64.26** and **ScanObjectNN ViT-B/16 = 35.66**,
with ViT-B/32 at 55.14 / 36.36, ResNet101 at 46.31 / 29.84 and ResNet50 at 39.38 / 35.32. The widely quoted
paper figure is 64.22. A 0.04 difference is small but it shows the quoted numbers are not stable, so the
report must say which source each one comes from. Note also that the repository's best ScanObjectNN backbone
is ViT-B/32, not the ViT-B/16 used for the ModelNet40 headline.

| Dataset / split | Published zero-shot accuracy | Verified in the PDF? |
|---|---|---|
| ModelNet40 | 64.22% (paper, as quoted) / **64.26% (upstream README, ViT-B/16)** | no |
| ModelNet10 | 73.13% | no |
| ScanObjectNN OBJ_BG | 41.22% | no |
| ScanObjectNN OBJ_ONLY | 50.09% | no |
| ScanObjectNN PB_T50_RS | 35.36% | no |

## Measured runs

| # | Date | Claim | Dataset / condition | Script + config | Commit | Seed | Result (measured) | Notes / deviations |
|---|---|---|---|---|---|---|---|---|
| - | - | - | - | - | - | - | *nothing measured yet* | scaffold only |

Columns to fill without exception: the script and config that produced it, the repo commit it ran at, the
seed, and every way the setup differed from the paper (backbone, number of points, views, CUDA version,
subset of the test set).

## Data provenance and the ScanObjectNN licence

ModelNet40's canonical host `shapenet.cs.stanford.edu` is dead (connection timeout, 2026-11-12), so both
datasets came from the copy the PointCLIP V2 authors redistribute. For ModelNet40 that is an advantage: it
is the exact preprocessing their reported numbers were measured on.

**ScanObjectNN is different and needs Anas's attention.** Its official distribution requires registering and
agreeing to its terms of use. This copy was obtained second-hand from a third party's Drive folder, not from
the official site. Before any ScanObjectNN result goes in the submitted report:

1. register on the official ScanObjectNN site and accept the terms, so the use is properly licensed;
2. state in the report where the files actually came from, and that the counts match the official splits
   (581 / 581 / 2,882).

Do not present a second-hand copy as an official download. If registration is refused or too slow, Track B
is dropped and the project stands on ModelNet40 plus Track C. That is Anas's call, not a model's.

## Environment incidents (kept because they change what the numbers mean)

| Date | Incident |
|---|---|
| 2026-11-12 | **Resolved, same day:** the block did not persist. On the next import, in `pytest`, torch loaded, and `torch.cuda.is_available()` is now `True` on the RTX 5070 (capability 12.0, `sm_120` in the arch list, 7.9 GB). So Smart App Control blocked the DLL on its **first** load and allowed it afterwards, which fits a reputation check that had not yet resolved for a freshly downloaded file. **No security setting was changed.** Keeping the row above because it will happen again on any machine that installs these wheels for the first time: if `import torch` fails this way, try it a second time before changing anything. |
| 2026-11-12 | **Smart App Control blocks the CUDA PyTorch wheels.** The cu128 install succeeded, but `import torch` fails with `OSError: [WinError 4551] An Application Control policy has blocked this file ... torch\lib\shm.dll`. Windows event log, Microsoft-Windows-CodeIntegrity/Operational, events 3033/3077/3118 at 13:27: the DLL "did not meet the Enterprise signing level requirements", policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`. `HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy\VerifiedAndReputablePolicyState = 1`, i.e. Smart App Control is **on**. The CPU wheels from PyPI load fine; the wheels from `download.pytorch.org` do not. This is a signature check, not a path check, so moving the venv will not help. **A1 is blocked on this.** Options in `SETUP.md` section 1a. No security setting was changed. |
| 2026-11-12 | `pip install -r requirements.txt` pulled **CPU-only torch 2.15.0+cpu** from PyPI as a dependency of `open_clip_torch`, before torch was installed deliberately. A later `pip install torch --index-url .../cu128` then reported "already satisfied" and changed nothing, so the environment looked complete while being CPU-only. Fixed by uninstalling torch and torchvision and reinstalling from the cu128 index. **Check `torch.cuda.is_available()` after any pip command that touches torch.** |

## Known limitations to carry into the report

1. **PCA yaw canonicalisation has a 180-degree ambiguity.** `principal_yaw` returns an angle in [0, 180)
   because PCA gives a direction, not an orientation. A chair and the same chair turned around are
   indistinguishable to it. This is a property of the method, not a bug, and the report must say so rather
   than quietly ignoring it. Resolving it needs a semantic cue.
2. **The up axis is assumed to be +Z** (`UP_AXIS` in `src/yaw.py`). If a dataset's preprocessing uses +Y,
   this must be changed in that one place and the affected runs re-done and re-logged.
3. **The published protocol tunes on the test labels.** `--post-search` in the upstream code selects the
   prompt and the per-view weights by maximising accuracy on the labels of the evaluated set. The baseline
   must therefore be reported both with and without post-search, and every comparison must hold the protocol
   fixed. Details and line numbers in `UPSTREAM_NOTES.md`, finding 1. This is a limitation of the *source*,
   not of our reproduction, and it belongs in the comparison matrix.
4. **8 GB of VRAM.** If the upstream default backbone or batch size does not fit, the reduction is a
   deviation and goes in the log.
