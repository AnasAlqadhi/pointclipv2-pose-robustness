# Task briefs (for Antigravity, or any model continuing this work)

**Read `../AGENTS.md` before starting any task.** Do one task at a time, in order. Touch only the files the
task lists. When a task is done, tick it here, add the row to `REPRO_LOG.md`, and add a row to
`../../AI_USE.md`. If you are blocked, write the blocker in that task's Notes and **stop** - do not switch
method, dataset or paper on your own.

Order matters: A1 → A2 → A4 → A5 → A6 → A7. A3 can be done any time and needs no GPU. R1 runs on the ZBook
in parallel.

Status: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked (see Notes)

---

## [ ] A1 - Environment and upstream code

**Goal:** a working venv with CUDA torch, and the upstream repository cloned at a recorded commit.

**Steps:** follow `../SETUP.md` sections 1 and 2 exactly.

**Files you may touch:** `docs/REPRO_LOG.md` (the environment table and nothing else),
`docs/pip-freeze-dev.txt` (new), `requirements.txt` (only to pin a version that turned out to be needed).

**Done when:** `torch.cuda.is_available()` is `True` and prints the RTX 5070; `pip freeze` is saved; the
upstream commit hash is written into the environment table; `pytest tests -q` still passes (25 tests).

**Do not:** install a CPU torch build; commit `third_party/`; give up on CUDA and silently continue on CPU -
if CUDA cannot be made to work, mark this `[!]` and write why.

**Notes:** the 5070 is Blackwell (sm_120), so it needs the cu128 wheels. A plain `pip install torch` gives a
CPU build. Expect the upstream repo to want older pins; record every substitution you make.

---

## [ ] A2 - ModelNet40 data

**Goal:** the ModelNet40 test split in place, preprocessed exactly as the upstream repo expects.

**Steps:** follow the upstream `DATASET.md`. Put everything under `data/`. The test split should come to
2,468 objects - check that count and write it down.

**Files you may touch:** `docs/REPRO_LOG.md` (environment table: dataset version, file name, size, date),
`SETUP.md` (only to correct a step that was wrong).

**Done when:** a script can load the test split and report 2,468 objects and 40 class names.

**Do not:** commit anything under `data/`; resample or re-normalise the clouds differently from upstream -
the preprocessing must match or the baseline will not reproduce, and that difference would be invisible later.

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

## [ ] A4 - Baseline evaluation harness (claim C1)

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

---

## [ ] A5 - Yaw sweep (claims C2 and C3)

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
| 2026-11-12 | Claude Code (Opus) | scaffold | Repo created, folder structure, `src/yaw.py` and `src/metrics.py` with 25 passing tests, docs and these briefs. No experiment run, no data downloaded, torch not yet installed in the venv |
