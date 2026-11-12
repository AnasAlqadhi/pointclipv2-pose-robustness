# Task briefs (for Antigravity, or any model continuing this work)

**Read `../AGENTS.md` before starting any task.** Do one task at a time, in order. Touch only the files the
task lists. When a task is done, tick it here, add the row to `REPRO_LOG.md`, and add a row to
`../../AI_USE.md`. If you are blocked, write the blocker in that task's Notes and **stop** - do not switch
method, dataset or paper on your own.

Order matters: A1 → A2 → A4 → A5 → A6 → A7. A3 can be done any time and needs no GPU. R1 runs on the ZBook
in parallel.

Status: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked (see Notes)

---

## [x] A1 - Environment and upstream code - DONE 2026-11-12

**Goal:** a working venv with CUDA torch, and the upstream repository cloned at a recorded commit.

**Steps:** follow `../SETUP.md` sections 1 and 2 exactly.

**Files you may touch:** `docs/REPRO_LOG.md` (the environment table and nothing else),
`docs/pip-freeze-dev.txt` (new), `requirements.txt` (only to pin a version that turned out to be needed).

**Done when:** `torch.cuda.is_available()` is `True` and prints the RTX 5070; `pip freeze` is saved; the
upstream commit hash is written into the environment table; `pytest tests -q` still passes (25 tests).

**Do not:** install a CPU torch build; commit `third_party/`; give up on CUDA and silently continue on CPU -
if CUDA cannot be made to work, mark this `[!]` and write why.

**Notes (2026-11-12, Claude):** done so far - venv created, non-torch dependencies installed, upstream
cloned at commit `17df8a269c14cb1601b596b3bc19274964fbd0d9` and read (see `UPSTREAM_NOTES.md`), 25 tests
passing.

**Result:** `torch 2.11.0+cu128`, `cuda.is_available() == True`, RTX 5070 at capability 12.0, `sm_120`
supported, 7.9 GB, a GPU matmul and a GPU `scatter_reduce_` both verified. 29 tests pass, 3 skip.

**One trap for anyone repeating this:** the first `import torch` was blocked by Smart App Control
(`WinError 4551` on `torch\lib\shm.dll`; CodeIntegrity events 3033/3077/3118). It worked on the next
import - a reputation check that had not yet resolved for freshly downloaded files. **If this happens, try
again before changing anything, and never turn Smart App Control off to get past it:** that is irreversible
and is Anas's decision alone, not a model's.

The 5070 is Blackwell (sm_120) and needs cu128 wheels; a plain `pip install torch` gives a CPU build. Expect
the upstream repo to want older pins; record every substitution you make.

---

## [x] A2 - ModelNet40 data - DONE 2026-11-12

**Goal:** the ModelNet40 test split in place, preprocessed exactly as the upstream repo expects.

**Steps:** follow the upstream `DATASET.md`. Put everything under `data/`. The test split should come to
2,468 objects - check that count and write it down.

**Files you may touch:** `docs/REPRO_LOG.md` (environment table: dataset version, file name, size, date),
`SETUP.md` (only to correct a step that was wrong).

**Done when:** a script can load the test split and report 2,468 objects and 40 class names.

**Do not:** commit anything under `data/`; resample or re-normalise the clouds differently from upstream -
the preprocessing must match or the baseline will not reproduce, and that difference would be invisible later.

**Result (2026-11-12):** ModelNet40 verified, all checks pass - 40 classes, 2,468 test / 9,840 train, 2,048
points, unit-sphere normalised. The canonical Stanford host is dead; the data came from the upstream authors'
Drive folder, which is the preprocessing their numbers were measured on. The split is imbalanced (100 vs 20
objects per class), so macro accuracy must be reported alongside overall accuracy.

**ScanObjectNN arrived in the same folder** (1.8 GB; OBJ_BG 581, OBJ_ONLY 581, PB_T50_RS 2,882 - counts
match the literature), so Track B needs no separate download. **But it is a second-hand copy**: the official
distribution requires registering and accepting terms. Anas must register before any ScanObjectNN number
goes in the report, and the report must say where the files came from. See `REPRO_LOG.md`.

`project/data` is junctioned into `third_party/PointCLIP_V2/zeroshot_cls/data` so the upstream relative
paths resolve.

---

## [ ] A3 - Verify the published numbers (no GPU needed)

**Goal:** replace the `no` in the "Verified in the PDF?" column of `REPRO_LOG.md` with `yes` for each number
you have personally seen in the paper, and correct any that are wrong.

**Steps:** open arXiv:2211.11682, find the zero-shot classification table, read the ModelNet40, ModelNet10
and three ScanObjectNN split numbers, and note the table number and the exact setting (which backbone, how
many views).

**Files you may touch:** `docs/REPRO_LOG.md` (the published-numbers table only).

**Done when:** every row is either verified with the table number recorded, or corrected, or deleted with a
note saying the paper does not report it.

**Do not:** copy a number from a blog, a Papers-with-Code mirror or a summary site. Those sources already
disagree with each other, which is exactly why this task exists.

---

## [x] A4 - Baseline evaluation harness (claim C1) - DONE 2026-11-12

**Goal:** one script that runs PointCLIP V2 zero-shot classification over a dataset split and writes a
results table, so that every later experiment is the same code path with a different config.

**Files you may create or touch:** `scripts/run_eval.py`, `configs/baseline_modelnet40.yaml`,
`src/evaluate.py`, `results/`, `docs/REPRO_LOG.md`.

**Requirements:**
- the config holds dataset, split, backbone, number of views, yaw transform (`none` here), seed, batch size;
- predictions and labels are saved so later comparisons can be **paired** (`src/metrics.paired_bootstrap_delta`
  needs the per-item predictions, not just the accuracy);
- accuracy is reported through `src/metrics.accuracy_with_ci`, plus macro accuracy from
  `per_class_accuracy` - ModelNet40 is imbalanced;
- the script records its own config, the commit hash and the runtime into the results file.

**Done when:** a measured ModelNet40 number exists next to the published 64.22% in `REPRO_LOG.md`, with the
exact command, and the per-item predictions are saved under `results/`.

**Do not:** report only the mean; adjust anything to make the number closer to the paper's. If it does not
match, log the gap and investigate it in a new row - the brief explicitly asks for that.

**Result (2026-11-12):** harness is `src/pcv2.py` + `src/data.py` + `scripts/run_eval.py` + configs. It
drives the upstream projection and bundled CLIP from our own loop, so a transform can be applied to the
cloud before projection and per-item predictions are saved for paired statistics.

- published protocol: **63.61%** [61.71, 65.52], macro 55.81% - the published 64.22-64.26 is inside the
  interval, so C1 reproduces;
- plain protocol (fixed template prompt, uniform view weights): **32.50%** [30.63, 34.36];
- 22.6 ms/object, peak 1.11 GB of 7.9 GB, so the full test set takes under a minute. A 12-angle yaw sweep is
  about 12 minutes. Memory is not a constraint; batch 16 is fine.

**The one thing that cost us 7 points, and that anyone reproducing this must know:** upstream truncates each
cloud to its **first 1,024 points** (`Dassl3D/dassl/data/data_manager.py:200-201`), while the HDF5 files hold
2,048. Feeding all 2,048 gives 57.21%. `num_points: 1024` in the configs. Do not change it without logging a
new run.

---

## [x] A5 - Yaw sweep (claims C2 and C3) - DONE 2026-11-12

**Goal:** measure how accuracy changes as the input cloud is rotated about the up axis.

**Files you may create or touch:** `scripts/run_yaw_sweep.py`, `configs/yaw_sweep_*.yaml`, `results/`,
`docs/REPRO_LOG.md`.

**Requirements:**
- use `src.yaw.rotate_yaw` and `src.yaw.yaw_grid(30.0)` - do not re-implement the rotation;
- apply the rotation to the **point cloud before projection**, which is where a robot's pose variation
  enters; applying it to the rendered image instead is a different experiment and would be wrong here;
- one row per angle: overall accuracy with CI, macro accuracy, and the per-class table saved to disk;
- run it on ModelNet40 first, then ScanObjectNN if A2's follow-up has landed.

**Done when:** a table and a plot of accuracy against yaw angle exist, with error bars, and the per-class
results are saved so C3 can be examined.

**Do not:** average over angles and report one number - the shape of the curve *is* the finding. Do not drop
angles that look odd.

**Result (2026-11-12).** Up axis settled first by rendering (`scripts/check_up_axis.py`,
`results/figures/up_axis_check.png`): it is **+Y**, not the +Z the code originally assumed.

- published protocol: 0 deg 63.61%, worst 54.46% at 300 deg, **spread 9.16 pts**, paired
  [+7.58, +10.74]. The curve peaks at 0/90/180/270 - the projection's own view azimuths. On-grid 61.40%
  vs off-grid 55.47%, non-overlapping;
- plain protocol: spread only **1.54 pts**. The sensitivity is a property of the test-tuned prompts and
  view weights, not of the projection;
- per-class: symmetric +1.84 pts, oriented +6.17 pts - direction right, detail wrong (see `REPRO_LOG.md`,
  "Reading of runs 5-7"). Do not rewrite the hypothesis to fit.

Figures: `results/figures/yaw_sweep.png`, `results/figures/yaw_per_class.png`.

**What this means for A6.** The mechanism is now known, so the fix has a reason stated before the result:
if the loss comes from objects falling between fixed views, yaw canonicalisation should move them back onto
the grid and yaw-spread ensembling should cover the gaps. Also test the fix under **both** protocols - the
interesting question is whether it recovers the published protocol's lost 9 points, or whether it mostly
closes the gap that the test-tuned weights opened.

---

## [ ] A6 - The fix and its controls (claims C4, C5, C6)

**Goal:** test whether yaw canonicalisation and/or yaw ensembling recovers the loss, with the three controls
that make the result defensible.

**Files you may create or touch:** `scripts/run_fix.py`, `configs/fix_*.yaml`, `results/`,
`docs/REPRO_LOG.md`.

**The four conditions, all on the same rotated test items:**
1. baseline (no fix) - the rotated accuracy from A5;
2. canonicalisation only (`src.yaw.canonicalise_yaw`), same view budget as the baseline;
3. yaw ensemble only (`src.yaw.yaw_ensemble_angles(k)`), k views;
4. both.

**The three controls:**
- **same budget (C5):** condition 3 must be compared against k views that are *not* yaw-diverse (for example
  the upstream views repeated, or k elevations), so a gain cannot be attributed to extra compute;
- **clean condition (C6):** run every condition on the unrotated data too;
- **ablation:** the four conditions above already give it - report them all, including the ones that lose.

**Also record:** CLIP forward passes per object, wall-clock latency per object, peak GPU memory.

**Compare with:** `src.metrics.paired_bootstrap_delta` on the same items. If an interval contains zero, write
that the difference is not resolvable at this sample size. Do not call it an improvement.

**Done when:** a table with all conditions, both clean and rotated, with paired intervals and the resource
columns, plus the ablation of the alternative canonicaliser (`src.yaw.extent_yaw`) if time allows.

**Do not:** tune k, the canonicaliser or anything else on the test set. Fix them a priori and say so.

---

## [ ] A7 - Figures and tables for the report

**Goal:** the figures the report needs, generated by a script so they can be regenerated.

**Files you may create or touch:** `scripts/make_figures.py`, `results/figures/`.

**The figures:** accuracy against yaw with error bars (all conditions); per-class accuracy drop under
rotation, sorted, with symmetric classes marked; a small gallery of rendered depth views for objects that
break, next to the ones that survive.

**Do not:** hand-edit a figure or a number into the report. Everything traces back to a script.

---

## [ ] R1 - Robot clouds on the ZBook (Track C, behind the Saturday gate)

**Run on:** zbook (Ubuntu 22.04, ROS, `~/tb3_3d_ws`). **Not** on the dev laptop. No CLIP on the ZBook.

**Gate:** only start this once A4 and A5 are done. If they are not green by Saturday night, Track C is
cancelled and the project is complete without it. That decision is Anas's.

**Goal:** point clouds of a handful of objects as a robot actually sees them - partial, sparse, at the yaw
the robot approached from.

**Steps:** put a few distinguishable objects (chair, table, monitor, box - classes that exist in ModelNet40
or ScanObjectNN) in a Gazebo world; drive the robot past them from several directions; record a rosbag of the
3D cloud topic; crop each object into its own cloud; save as `.npy` with the approach yaw recorded; copy to
`data/robot/` on dev.

**Done when:** at least 5 objects x 4 approach directions exist as `(N, 3)` arrays with labels and recorded
yaw, plus a short note on the sensor, the world, and the cropping method.

**Do not:** install CUDA or CLIP on the ZBook; let a long Gazebo job run without checking the machine is
still on the network (it drops off - see `WORKSPACE_GUIDE.md` open item 1).

**Fallback if ROS costs more than a day:** simulate it on dev instead - hidden-point removal from one
viewpoint, subsample to LiDAR-like density, random yaw. Same scientific story, minus the real-sensor claim.
Mark R1 `[!]` and say the fallback was used.

---

## Session log

One line per working session, so the next model knows where things stood.

| Date | Who | Task | What happened |
|---|---|---|---|
| 2026-11-12 | Claude Code (Opus) | scaffold | Repo created, folder structure, `src/yaw.py` and `src/metrics.py` with 25 passing tests, docs and these briefs |
| 2026-11-12 | Claude Code (Opus) | A5 | Up axis settled as +Y by rendering (the code had assumed +Z). Yaw sweep run on both protocols: published loses 9.16 pts and peaks on the view grid, plain varies by 1.54. Per-class C3 partially supported. **Next: A6 (the fix and its controls).** |
| 2026-11-12 | Claude Code (Opus) | A4 | Harness written and both baselines measured: published protocol 63.61% (reproduces), plain protocol 32.50%. Found the 1024-point truncation that explained an initial 7-point gap. **Next: A5 (yaw sweep). Before the sweep, settle which axis is "up" in these clouds - `src/yaw.py` assumes +Z and that is UNVERIFIED; the upstream views rotate about the first Euler axis, and PointNet's own augmentation rotates about Y. Decide it with rendered images, not by assumption.** |
| 2026-11-12 | Claude Code (Opus) | A2 | ModelNet40 downloaded and verified (all checks pass); ScanObjectNN came with it but is a second-hand copy needing registration. Data junctioned into the upstream tree. **Next: A4 (baseline harness), and A3 which needs no GPU.** Nothing measured yet |
| 2026-11-12 | Claude Code (Opus) | A1 | Venv with torch 2.11.0+cu128 working on the 5070; upstream cloned at `17df8a26` and read (`UPSTREAM_NOTES.md`: post-search tunes on test labels; torch-scatter replaceable). A1 done. **Next: A2 (ModelNet40 data), and A3 which needs no GPU.** Nothing measured yet |

---

## [~] A8 - Deeper ablations (queued 2026-11-12, running unattended)

Added because the mechanism found in A5/A6 opens questions nobody has answered. Each is cheap because the
harness exists. **Predictions are written down before the runs; do not edit them afterwards.**

### A8.1 The 2x2 factorial - which half of the test-set tuning does what?
`configs/yaw_sweep_modelnet40_prompts_only.yaml` (searched prompts + uniform weights) and
`..._weights_only.yaml` (template prompt + searched weights) fill the two missing corners; `published` and
`plain` are the other two.

**Prediction:** the searched **weights** carry the pose sensitivity and the searched **prompts** carry the
accuracy, because weights are indexed by viewpoint and prompts are not. So `prompts_only` should be
accurate and comparatively flat across yaw; `weights_only` should be inaccurate and pose-sensitive.

### A8.2 Ensemble span sweep
`configs/span_sweep_modelnet40.yaml`, k=4, spans 11.25 / 22.5 / 45 / 90 / 360 at angles 0 and 30.

**Prediction:** spans 90 and 360 are near no-ops (the view set is closed under a 90 degree turn of the
object); small spans help; the benefit is largest at the trough (30) and smallest at the peak (0).

### A8.3 Fine yaw sweep, 5 degree steps
`configs/yaw_sweep_modelnet40_published_fine.yaml`, 72 angles.

**Prediction:** a sawtooth with teeth every 45 degrees, not 90 - the ten viewpoints are two groups of four,
at 0/90/180/270 and at 45/135/225/315, so their union has 45 degree spacing.

### Still open, in priority order
- **ScanObjectNN with a pre-registered prediction**: canonicalisation should help *more* on PB_T50_RS than
  on OBJ_ONLY, because PB_T50_RS is already rotation-perturbed by its authors. Register the prediction in
  writing first. The licence step (registering officially) is Anas's, and comes before any reported number.
- **Backbone ablation**: RN50, RN101, ViT-B/32 and ViT-B/16 all have shipped searched weights. Is the
  fragility backbone-specific?
- **R1, the robot clouds** - now a bonus rather than load-bearing, which is where risky work belongs.

## Backups

`scripts/backup.sh` copies results, docs, src, scripts, configs and tests to the OneDrive folder
`Backups/YZM5110-pointclipv2` (about 1.2 MB). Not copied: `data/` (2.2 GB, re-downloadable and recorded in
`REPRO_LOG.md`), `third_party/` (a pinned clone, recorded by commit hash) and `.venv/` (rebuildable from
`SETUP.md`). The GitHub repository is the primary offsite copy and OneDrive is the second. Run the script
after each batch of results.
