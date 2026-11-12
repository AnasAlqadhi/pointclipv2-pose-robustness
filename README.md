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
| C1 | PointCLIP V2 reaches 64.22% zero-shot on ModelNet40 (reproduction) | not run |
| C2 | Accuracy degrades substantially under yaw rotation of the input cloud | not run |
| C3 | The degradation is concentrated in orientation-dependent classes, not symmetric ones | not run |
| C4 | Yaw canonicalisation and/or yaw-ensembled views recover part of the loss | not run |
| C5 | The gain is not explained by the extra compute alone (same-view-budget control) | not run |
| C6 | The fix does not harm the canonical, aligned case | not run |

Published numbers and measured numbers are kept strictly separate: see `docs/REPRO_LOG.md`.

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
