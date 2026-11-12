# Pose robustness of training-free zero-shot 3D recognition

Repository: https://github.com/AnasAlqadhi/pointclipv2-pose-robustness

YZM5110 Multimodal Learning, research project (30%), Anas Alqadhi. Due 18 November 2026.

## Research question

PointCLIP V2 ([arXiv:2211.11682](https://arxiv.org/abs/2211.11682)) does zero-shot 3D classification without
training on 3D data: it renders a point cloud to a small set of **fixed** depth views and classifies them
with frozen CLIP, reporting 64.22% on ModelNet40.

ModelNet40 objects are canonically aligned - every chair faces the same way. A robot never gets that
alignment for free. So:

> **Does the training-free zero-shot claim survive when the object's yaw is arbitrary, and can yaw
> canonicalisation recover what is lost?**

## Claims this repository tests

| # | Claim | Status |
|---|---|---|
| C1 | PointCLIP V2 reaches 64.22% zero-shot on ModelNet40 (reproduction) | **reproduced: 63.61%** [61.71, 65.52] under the published protocol; the published figure is inside our interval |
| C1b | That figure depends on prompts and view weights selected on the test labels | **confirmed: 32.50%** [30.63, 34.36] with a fixed template prompt and uniform view weights - about half |
| C2 | Accuracy degrades substantially under yaw rotation of the input cloud | **confirmed: 63.61% at 0°, 54.46% at worst, spread 9.16 pts** [+7.58, +10.74] paired. The curve peaks at 0/90/180/270°, the projection's own view azimuths |
| C2b | That sensitivity belongs to the test-tuned protocol, not to the projection | **confirmed: the plain protocol varies by only 1.54 pts** across the whole sweep |
| C3 | The degradation is concentrated in orientation-dependent classes, not symmetric ones | **partially supported: +1.84 pts (symmetric) vs +6.17 pts (oriented)**, but several oriented classes *gain* under rotation and `chair`/`desk` barely move. See `docs/REPRO_LOG.md` |
| C4 | Yaw canonicalisation and/or yaw-ensembled views recover part of the loss | **supported, as a trade:** canonicalisation flattens the sweep (spread 8.14 → **0.16** pts) and raises the mean from 58.69 to **60.66**; +4.94 [+3.32, +6.56] at 30°, +5.11 [+3.48, +6.69] at 60° |
| C5 | The gain is not explained by the extra compute alone (same-view-budget control) | **passed:** the same-budget jitter ensemble gains nothing (−0.41 at 30°). The 10-pass fix also beats the 40-pass ensemble at every angle |
| C6 | The fix does not harm the canonical, aligned case | **refuted, and reported as such: −2.96 pts** [−3.93, −2.03] at 0°. Robustness is bought by giving up the benchmark's supplied alignment |
| C7 | The fix changes nothing where there is no pose sensitivity (negative control) | **confirmed:** under the plain protocol no condition improves at any angle |

Published numbers and measured numbers are kept strictly separate: see `docs/REPRO_LOG.md`.

**Headline so far.** The published protocol reproduces (63.61% against a published 64.22-64.26). But the
prompts and per-view weights that produce it were selected by maximising accuracy on the test labels, and
they ship hard-coded in `trainers/best_param.py` under keys literally named `..._test_prompts` and
`..._test_weights`, so they are used even without the `--post-search` flag. Replace them with one fixed
template prompt and uniform weights and the same pipeline scores **32.50%**. About half of the headline
zero-shot result comes from test-set selection. Details: `docs/UPSTREAM_NOTES.md`.

## Layout

```
src/            yaw rotation, canonicalisation, metrics, the evaluation harness
scripts/        one script per experiment, each writing a table into results/
configs/        experiment configs (dataset, views, yaw grid, seed)
results/        committed tables and figures
docs/           REPRO_LOG.md (every run recorded), SETUP.md, the task briefs
third_party/    PointCLIP_V2 upstream clone - not committed, see SETUP.md
data/           datasets - not committed, see SETUP.md
```

## Status

Scaffolded 2026-11-12. Nothing has been measured yet. See `docs/REPRO_LOG.md` for what has actually run and
`../PROJECT.md` for the plan and the schedule.

## Reproducing

See `SETUP.md`. Every reported number must come from a script in `scripts/` with its config in `configs/`,
and must have a row in `docs/REPRO_LOG.md` recording the upstream revision, dependency versions, data
version, seed and exact command.

## AI assistance

Disclosed as the course requires, in `../AI_USE.md`.

## Start here

| If you want to... | Read |
|---|---|
| continue this work anywhere (the full plan) | **`docs/WORK_PLAN.md`** |
| know what is left before submission | `docs/FINISH_LINE.md` |
| write the report | `docs/REPORT_DRAFT.md` (sections 1, 3-7 written; 2 and 8 are Anas's) |
| read the five papers | `docs/LITERATURE.md` |
| build the slides | `docs/PRESENTATION.md` |
| check where a number came from | `docs/REPRO_LOG.md` (append-only, 19 runs) |
| continue this with any AI model | `AGENTS.md`, then `docs/ANTIGRAVITY_TASKS.md` |
