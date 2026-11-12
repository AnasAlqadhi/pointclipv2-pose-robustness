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
| ModelNet40 version | not yet downloaded |
| ScanObjectNN version | not yet requested |

Fill each of these in as it becomes true, and run `pip freeze > docs/pip-freeze-dev.txt` once the
environment is complete.

## Published numbers (the paper's, not ours)

From PointCLIP V2, arXiv:2211.11682. **Every one of these must be verified against Table 2 of the PDF
before it is cited** - the secondary sources I found disagree with each other, so these are marked
unverified until someone reads the table.

| Dataset / split | Published zero-shot accuracy | Verified in the PDF? |
|---|---|---|
| ModelNet40 | 64.22% | no |
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
