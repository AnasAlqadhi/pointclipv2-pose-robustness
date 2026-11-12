# Pose robustness of training-free zero-shot 3D recognition

**DRAFT of 2026-11-12. Anas Alqadhi. YZM5110 Research Project.**
Final file name: `Alqadhi_YZM5110_Research.pdf`. Repository URL goes on page 1.
Repository: https://github.com/AnasAlqadhi/pointclipv2-pose-robustness

> **How to use this draft.** Sections 3-7 are written from measurements that exist and can be regenerated
> from the repository. **Sections 2 and 8 are yours to write** - section 2 requires reading five papers,
> which no tool can do for you, and section 8 is your judgement. Every number below is traceable to a row
> in `REPRO_LOG.md`. Before submitting, check each one against that log rather than trusting this file.

---

## 1. Research question, scope, motivation, contributions

PointCLIP V2 performs zero-shot 3D object classification without training on 3D data: it renders a point
cloud to ten fixed depth views and classifies them with frozen CLIP, reporting 64.22% on ModelNet40. Every
object in that benchmark is canonically aligned - every chair faces the same way. A robot never receives
that alignment.

**Question.** Does the training-free zero-shot claim survive an arbitrary object yaw, and can a cheap
canonicalisation recover what is lost?

**Contributions.**

1. A reproduction of the published ModelNet40 result: **63.61%** [61.71, 65.52] against a published
   64.22-64.26.
2. Evidence that the published protocol is not untuned: the prompts and per-view weights it ships were
   selected on the test labels, and replacing them with a single template prompt and uniform weights drops
   the same pipeline to **32.50%**.
3. A measurement of pose sensitivity that the original work does not report: accuracy falls by up to
   **9.16 points** [+7.58, +10.74] under yaw rotation, and the curve peaks at 0/90/180/270 degrees - the
   projection's own view azimuths.
4. A factorial decomposition showing that the pose sensitivity follows the searched **prompts**, not the
   searched view weights - **refuting my own stated prediction**.
5. A fix: bounding-box yaw canonicalisation, which removes the pose dependence (spread 8.14 → **0.16**
   points) and raises mean accuracy to **60.66%** for the same ten CLIP passes, at a cost of 2.96 points
   in the canonical pose.
6. Controls that rule out the obvious alternative explanations: a same-compute ensemble that gains nothing,
   and a negative control protocol where the fix correctly does nothing.

**Scope.** ModelNet40, ViT-B/16, inference only, single GPU. No training was performed; none is required.

---

## 2. Literature search and critical synthesis - **ANAS WRITES THIS**

Required: at least five papers, each with the seven fields from the brief, plus an account of how the
papers were found and selected. Candidates and their role in the argument are in `../PROJECT.md` section 4.

The shape of the argument, for you to verify against your own reading: two bodies of work exist that do not
cite each other. One projects point clouds into frozen CLIP (PointCLIP, PointCLIP V2) or trains a 3D
encoder into its space (ULIP). The other studies rotation robustness of point-cloud models (ART-Point,
LocoTrans, the 2025 pose benchmark). The second has never been applied to the first. **That gap is the
justification for this project** - but you must confirm it yourself before claiming it, because a single
counter-example paper would undo the claim.

Do not cite a number you have not read in the source PDF. The log already records one discrepancy: the
paper says 64.22 and the authors' own repository says 64.26.

---

## 3. Technical explanation of the method

PointCLIP V2 has three stages.

**Projection.** The cloud is quantised into a 112×112×8 grid, densified by 3D max-pooling, smoothed with a
3D Gaussian kernel, collapsed along depth by a maximum, inverted and repeated to three channels. Ten
viewpoints are used: eight azimuthal, at 0, 45, 90, ... 315 degrees in two groups of four, plus a top and a
bottom view, each with a small elevation bias. Implementation: `trainers/mv_utils_zs.py`.

**Text side.** One prompt per class, encoded by frozen CLIP and repeated across the ten views.

**Scoring.** Image features are L2-normalised, scaled by a per-view weight, concatenated across views, and
matched against the text features by cosine similarity.

**The detail that governs everything in this report.** The prompts and the per-view weights are not generic.
They ship hard-coded in `trainers/best_param.py` under keys named `<dataset>_<backbone>_test_prompts` and
`..._test_weights`, and they were produced by `trainers/post_search.py`, which loads the evaluated set's
own labels (lines 70 and 134) and keeps whichever candidate maximises accuracy against them (loops at
93-122 and 145-177). They are loaded by the default code path **whether or not `--post-search` is passed**
(`trainers/zeroshot.py`, lines 18-19 and 69). So the published figure is obtained with hyper-parameters
selected on the test set.

---

## 4. Reproduction, implementation and deviations

Upstream commit `17df8a269c14cb1601b596b3bc19274964fbd0d9` (2025-08-12). Data: `modelnet40_ply_hdf5_2048`
from the authors' own distribution, verified by `scripts/check_modelnet40.py` (2,468 test objects, 40
classes, 2,048 points, unit-sphere normalised). Environment: Python 3.12, torch 2.11.0+cu128, RTX 5070
(8 GB), seed 0, exact versions in `docs/pip-freeze-dev.txt`.

| | published | measured |
|---|---|---|
| ModelNet40, ViT-B/16, published protocol | 64.22 (paper) / 64.26 (repo) | **63.61%** [61.71, 65.52] |

The published figure lies inside the interval.

**Deviations, all deliberate and all logged.**

1. **Our own evaluation loop** instead of the upstream `main.py`, because the experiments need a transform
   applied to the cloud before projection and per-item predictions saved for paired statistics. The
   upstream projection and bundled CLIP are used unchanged (`src/pcv2.py`).
2. **`torch-scatter` replaced** by the native `scatter_reduce_`, since the pinned 2.0.9 has no wheel for
   this torch. Equivalence is tested in `tests/test_scatter_equivalence.py`, including the empty-bucket
   behaviour that `include_self` controls.
3. **1,024 points per cloud**, matching `Dassl3D/dassl/data/data_manager.py:200-201`, which truncates each
   cloud. This one cost a 7-point discrepancy before it was found: using all 2,048 points gives 57.21%.
   The failed run is kept as row 3 of the log. The gap was our setup, not the method.

**Protocols.** `published` = shipped prompts + shipped weights. `plain` = one fixed template prompt per
class, "a depth map of a {class}.", chosen a priori and never tuned, with uniform view weights. Protocols
are never mixed inside a comparison.

---

## 5. Proposed improvement and its rationale

The yaw sweep (section 7) shows accuracy peaking at 0, 90, 180 and 270 degrees - exactly the azimuths of
the projection's fixed viewpoints. The loss is therefore not generic damage from rotation but objects
falling **between fixed views**. Two interventions follow from that mechanism, and both were specified
before being run:

- **Canonicalisation**: estimate the object's own yaw and remove it, so every object is presented in a
  consistent orientation. Two estimators are compared - PCA on the horizontal plane, and the yaw minimising
  the horizontal bounding-box area. Cost: one small eigen-decomposition per object, **no extra CLIP passes**.
- **Yaw ensembling**: average logits over k rotated copies, spread over 45 degrees - one tooth of the view
  grid. Cost: k times the CLIP passes.

A note on the ensemble span that belongs in the report: offsets of 90 degrees are nearly a **no-op**,
because the ten viewpoints are closed under a quarter turn, so the same images are re-rendered in a
different order. Measured on a 32-object probe, such an ensemble changed the logits by at most 4.3 and
flipped no predictions. This is why the span is 45 degrees.

---

## 6. Experimental design

Dataset ModelNet40 test split, all 2,468 objects, every condition on the same objects. Yaw applied to the
cloud **before** projection. Up axis +Y, established by rendering four objects about each candidate axis
(`results/figures/up_axis_check.png`) rather than assumed - the code initially assumed +Z, which would have
measured objects tipping over.

Metrics: top-1 accuracy with a 95% percentile bootstrap interval, macro accuracy alongside it because the
split is imbalanced (100 objects in 18 classes, 20 in 18 others), per-class accuracy, and **paired**
bootstrap deltas for every comparison, since all conditions see identical items. Resource measures: CLIP
forward passes per object, milliseconds per object, peak VRAM.

Controls: (i) same view budget - a jitter ensemble spending the same 40 passes without yaw diversity;
(ii) clean condition - every condition is also measured at 0 degrees; (iii) ablation - two canonicalisers,
ensembling alone, and both together; (iv) a negative control - the whole experiment repeated under the
plain protocol, which has no pose sensitivity to repair.

Nothing was tuned on the test set by us. The template prompt, the ensemble k and the span were fixed a
priori.

---

## 7. Results

### 7.1 The protocol accounts for half the headline

| protocol | accuracy |
|---|---|
| published (shipped prompts + weights) | **63.61%** [61.71, 65.52] |
| plain (template prompt + uniform weights) | **32.50%** [30.63, 34.36] |

### 7.2 Pose sensitivity, and where it comes from

Published protocol: 63.61% at 0 degrees, 54.46% at worst, **spread 9.16 points**, paired 0 vs 300 =
+9.16 [+7.58, +10.74]. On-grid angles (0/90/180/270) 61.40% [60.43, 62.37] against off-grid 55.47%
[54.79, 56.17]. Plain protocol: spread only 1.54 points.

**Factorial.** Spread across the 12-angle sweep, accuracy at 0 degrees in brackets:

| | searched weights | uniform weights |
|---|---|---|
| searched prompts | 9.16 (63.61%) | 6.93 (61.10%) |
| template prompt | 2.31 (33.02%) | 1.54 (32.50%) |

**My prediction was that the weights carried the pose sensitivity. It was wrong.** The accuracy half holds -
prompts are worth about 30 points, weights 1 to 2 - but the sensitivity follows the prompts: 6.93 points of
spread survive with uniform weights, while a template prompt leaves only 1.5 to 2.3 whatever the weights do.
The weights add about 2.2 points of spread on top.

The revised explanation, consistent with the data but not proved by it: the searched prompts describe the
object **as seen from the canonical viewpoints** - "a side view depth map of a bed looks like a rectangle
with a smaller rectangle on top of it" - so they were fitted to canonical-pose renders and stop matching
once the object turns. A direct test would be hand-written pose-neutral prompts.

### 7.3 Per-class damage

Symmetric classes lose 1.84 points on average, oriented classes 6.17 - the predicted direction, a factor of
3.4. But the detail does not fit: the worst hit are flat box-like objects (`keyboard` +30, `glass_box` +30,
`mantel` +26), `chair` and `desk` barely move, and several classes **improve** when rotated (`xbox` −20,
`tent` −14), which pure information loss cannot explain. Reported as partial support, with the a priori
symmetric class list left unedited.

### 7.4 The fix

| condition | 0° | 30° | 60° | 90° | mean | spread | passes |
|---|---|---|---|---|---|---|---|
| baseline | **63.61** | 55.71 | 55.47 | 59.97 | 58.69 | 8.14 | 10 |
| canonicalise (bbox) | 60.66 | **60.66** | **60.58** | 60.74 | **60.66** | **0.16** | 10 |
| canonicalise (PCA) | 59.32 | 59.36 | 59.48 | 59.40 | 59.39 | 0.16 | 10 |
| yaw ensemble k=4 | 59.12 | 57.78 | 57.94 | 58.83 | 58.42 | 1.34 | 40 |
| jitter ensemble (control) | 61.47 | 55.31 | 55.71 | 59.24 | 57.93 | 6.16 | 40 |
| canonicalise + ensemble | 57.90 | 58.06 | 58.06 | 58.31 | 58.08 | 0.41 | 40 |

Paired against baseline: **+4.94** [+3.32, +6.56] at 30 degrees, **+5.11** [+3.48, +6.69] at 60 degrees,
**−2.96** [−3.93, −2.03] at 0 degrees.

**Controls.** The same-budget jitter ensemble gains nothing (−0.41 at 30 degrees), so the ensemble's benefit
is yaw diversity rather than compute. The ten-pass canonicalisation beats the forty-pass ensemble at every
angle, and combining them is worse than canonicalisation alone. Under the plain protocol no condition
improves at any angle, which is correct: there is no pose sensitivity there to repair.

### 7.5 Failure analysis - **to add**

The gallery of objects that break, with their rendered views, from `scripts/make_figures.py`.

---

## 8. Limitations, conclusions, open questions - **ANAS WRITES THIS; raw material below**

- The fix costs 2.96 points in the canonical pose. Whether that trade is worth making depends on whether
  deployment supplies alignment; for a robot it does not, for this benchmark it does.
- PCA canonicalisation has an unavoidable 180-degree ambiguity: it returns a direction, not an orientation.
- One dataset, one backbone, one deadline. ScanObjectNN is downloaded but its licence step is outstanding,
  and the backbone ablation has not been run.
- The explanation in 7.2 is consistent with the evidence, not proved by it.
- The upstream protocol's use of test labels is a property of the released code as of commit `17df8a26`;
  it is not an accusation about intent, and should be written as an observation about the artefact.

---

## Appendix: what the repository contains

`src/yaw.py` the rotation maths, `src/metrics.py` bootstrap and paired statistics, `src/pcv2.py` the
pipeline, `scripts/` one script per experiment, `configs/` one config per run, `results/` every table and
figure, `docs/REPRO_LOG.md` the append-only record of every run including the ones that failed,
`docs/UPSTREAM_NOTES.md` the reading of the upstream code, `AGENTS.md` the working rules.
Tests: 29 passing, 3 skipped. Pin the submitted version with a tag.
