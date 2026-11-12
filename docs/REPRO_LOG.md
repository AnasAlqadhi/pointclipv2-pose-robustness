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

| # | Date | Claim | Dataset / condition | Script + config | Seed | Result (measured) | Notes / deviations |
|---|---|---|---|---|---|---|---|
| 1 | 2026-11-12 | C1 | ModelNet40 test, 2,468 objects, ViT-B/16, **published protocol** (shipped prompts + shipped view weights), 1,024 points | `scripts/run_eval.py configs/baseline_modelnet40_published.yaml` | 0 | **63.61%** [61.71, 65.52], macro 55.81% | **Reproduced.** The published 64.26 (upstream README) / 64.22 (paper) lies inside our interval. 22.6 ms/object, peak 1.11 GB. Deviations: torch_scatter shim; our own evaluation loop instead of `main.py` |
| 2 | 2026-11-12 | C1 | same, **plain protocol** (one fixed template prompt per class, uniform view weights) | `scripts/run_eval.py configs/baseline_modelnet40_plain.yaml` | 0 | **32.50%** [30.63, 34.36], macro 31.77% | Nothing selected on test data. Roughly **half** the published accuracy |
| 3 | 2026-11-12 | - | same as run 1 but with all 2,048 points | `configs/baseline_modelnet40_published.yaml` with `num_points: null` | 0 | 57.21% [55.27, 59.16], macro 49.60% | **Kept as the record of a resolved discrepancy.** Our first attempt used all 2,048 points and fell 7 points short. Upstream truncates each cloud to its first 1,024 points (`Dassl3D/dassl/data/data_manager.py:200-201`). With that corrected, run 1 reproduces. The gap was our setup, not the method |
| 5 | 2026-11-12 | C2, C3 | ModelNet40 test, **yaw sweep** 0-330 deg in 30 deg steps, published protocol, up axis +Y | `scripts/run_yaw_sweep.py configs/yaw_sweep_modelnet40_published.yaml` | 0 | 0 deg **63.61%**, worst 300 deg **54.46%**, **spread 9.16 pts**; canonical 63.61% vs 56.88% mean over the other 11 angles | **C2 confirmed.** Paired 0 deg minus 300 deg = **+9.16 pts [+7.58, +10.74]**, interval excludes zero. On-grid angles (0/90/180/270) **61.40%** [60.43, 62.37] vs off-grid **55.47%** [54.79, 56.17] |
| 6 | 2026-11-12 | C2 | same sweep, **plain protocol** | `configs/yaw_sweep_modelnet40_plain.yaml` | 0 | 0 deg 32.50%, worst 150 deg 31.04%, **spread 1.54 pts**; canonical 32.50% vs 31.53% mean | **The rotation sensitivity is almost entirely absent here.** On-grid 32.25% vs off-grid 31.29%; paired 0 vs 300 = +1.30 pts [+0.45, +2.19]. Small but not zero |
| 7 | 2026-11-12 | C3 | per-class drop, published protocol | `scripts/plot_yaw_sweep.py` | 0 | symmetric classes **+1.84 pts** mean drop (n=9), oriented classes **+6.17 pts** (n=31) | **C3 partially supported** - see the honest reading below |
| 8 | 2026-11-12 | C4, C5, C6 | ModelNet40, angles 0/30/60/90, six conditions, **published protocol** | `scripts/run_fix.py configs/fix_modelnet40_published.yaml` | 0 | see the table below | **C4 supported, C5 passed, C6 shows a real cost** |
| 9 | 2026-11-12 | C4, C5 | same, **plain protocol** | `configs/fix_modelnet40_plain.yaml` | 0 | every condition within +-1 pt of baseline; no condition "better" at any angle | **The fix does nothing where there is nothing to fix.** Exactly the right negative control |

### Run 8 in full (published protocol, accuracy %)

| condition | 0 deg | 30 deg | 60 deg | 90 deg | mean | spread | CLIP passes |
|---|---|---|---|---|---|---|---|
| baseline | **63.61** | 55.71 | 55.47 | 59.97 | 58.69 | 8.14 | 10 |
| canonicalise (bbox) | 60.66 | **60.66** | **60.58** | 60.74 | **60.66** | **0.16** | 10 |
| canonicalise (PCA) | 59.32 | 59.36 | 59.48 | 59.40 | 59.39 | 0.16 | 10 |
| yaw ensemble k=4, span 45 | 59.12 | 57.78 | 57.94 | 58.83 | 58.42 | 1.34 | 40 |
| jitter ensemble k=4 (**control**) | 61.47 | 55.31 | 55.71 | 59.24 | 57.93 | 6.16 | 40 |
| canonicalise + ensemble | 57.90 | 58.06 | 58.06 | 58.31 | 58.08 | 0.41 | 40 |

Paired against the baseline at the same angle: bbox canonicalisation **+4.94** [+3.32, +6.56] at 30 deg and
**+5.11** [+3.48, +6.69] at 60 deg, but **-2.96** [-3.93, -2.03] at 0 deg.

**What the run says, stated honestly:**

1. **C4 supported, as a trade and not a free win.** Canonicalisation flattens the curve: the spread falls
   from 8.14 points to 0.16, and the mean over the four angles rises from 58.69 to 60.66. It costs 2.96
   points at the canonical pose - which is C6 answering in the negative, and must be reported as such. The
   honest summary is: *the method becomes pose-independent and better on average, by giving up the
   alignment that the benchmark happens to supply.*
2. **C5 passed.** The jitter ensemble spends the same 40 forward passes as the yaw ensemble and gains
   nothing (-0.41 at 30 deg, -2.15 at 0 deg). So the yaw ensemble's gain comes from yaw diversity, not from
   the extra compute.
3. **The cheap fix beats the expensive one at every angle.** 10-pass canonicalisation is above 40-pass
   ensembling everywhere. Four times the compute for a worse result. Combining them is worse than
   canonicalisation alone, which suggests the ensemble's extra views mostly add noise once the pose is
   already fixed.
4. **The bounding-box canonicaliser beats PCA by about 1.3 points everywhere.** Keeping two canonicalisers
   as an ablation was worth it: reporting only PCA would have understated the fix.
5. **Run 9 is the control that matters most for honesty.** Under the plain protocol, which the yaw sweep
   showed is already pose-flat, no condition improves anything at any angle. A fix that "helped" there
   would have meant the gain was an artefact of the procedure rather than a repair of the pathology.

| 10 | 2026-11-12 | A8.1 | ModelNet40 yaw sweep, **searched prompts + uniform weights** | `configs/yaw_sweep_modelnet40_prompts_only.yaml` | 0 | 0 deg 61.10%, spread **6.93 pts**, peaks again at 0/90/180/270 | part of the factorial below |
| 11 | 2026-11-12 | A8.1 | ModelNet40 yaw sweep, **template prompt + searched weights** | `configs/yaw_sweep_modelnet40_weights_only.yaml` | 0 | 0 deg 33.02%, spread **2.31 pts** | part of the factorial below |

### The 2x2 factorial - my prediction was wrong

| | searched weights | uniform weights |
|---|---|---|
| **searched prompts** | 63.61%, spread **9.16** | 61.10%, spread **6.93** |
| **template prompt** | 33.02%, spread **2.31** | 32.50%, spread **1.54** |

**The prediction, recorded in `ANTIGRAVITY_TASKS.md` before these runs, was that the searched *weights*
carry the pose sensitivity and the prompts carry the accuracy.** The reasoning was that weights are indexed
by viewpoint and prompts are not. The accuracy half is right - prompts are worth about 30 points, weights
about 1 to 2. The pose half is **wrong**: the sensitivity follows the **prompts**. With uniform weights and
searched prompts the spread is still 6.93 points, while with a template prompt it is 1.5 to 2.3 points
whatever the weights do. The weights add roughly 2.2 extra points of spread on top (9.16 against 6.93), so
they contribute, but they are not the cause.

**A better explanation, and it is more interesting than the prediction was.** The searched prompts are not
pose-neutral descriptions. Read them in `trainers/best_param.py`: "a side view depth map of a bed looks
like a rectangle with a smaller rectangle on top of it", "the chair would be a dark object on the grayscale
map", "a depth map of a cone would look like a cone shaped object". They describe what the object looks
like **from the canonical viewpoints, in the canonical pose**. They were selected on exactly those renders.
Rotate the object and the description stops matching the image. So the fragility is not in a weighting
vector; it is baked into the language that was tuned against canonically posed depth maps.

Do not quietly replace the old prediction with this one in the report. State the prediction, state that it
was refuted, and give this account as the revised explanation - with the caveat that it is an explanation
the data is consistent with, not one the data proves. A direct test would be to write pose-neutral prompts
by hand and check whether accuracy survives rotation better; that is the obvious next experiment.

| 12 | 2026-11-12 | A8.2 | ModelNet40, ensemble span sweep, k=4, published protocol | `scripts/run_span_sweep.py configs/span_sweep_modelnet40.yaml` | 0 | at 30 deg: span 11.25 **+1.38** [+0.53, +2.23], 22.5 **+2.92** [+1.94, +3.93], 45 **+2.07**, 90 **+2.31**, 360 **-0.20** [-0.89, +0.49] (no effect). At 0 deg every span is WORSE | prediction half right - see below |

### The span sweep, and a second correction to my own prediction

The prediction was that spans of **90 and 360** would be near no-ops. Only **360** is: its offsets are
[0, 90, 180, 270], multiples of a quarter turn, and the ten viewpoints are closed under that rotation, so
the ensemble re-renders the same images. Measured: -0.20 points [-0.89, +0.49] at 30 degrees, i.e. nothing.

Span **90 is not** a no-op (+2.31 at 30 degrees) and I was wrong to predict it would be. With k=4 the
offsets are spread *within* the span: span 90 gives [0, 22.5, 45, 67.5], which are not multiples of 90 and
do place views between the grid teeth. The mistake was conflating the span with the step. The correct
statement is: **an ensemble helps exactly when its offsets are not multiples of 90 degrees**, and the
sweep supports it - 11.25, 22.5, 45 and 90 all help at a trough; 360, whose offsets are all multiples of
90, does not.

The best span at 30 degrees is 22.5 (+2.92), slightly better than 45 (+2.07), but no span of the ensemble
reaches bounding-box canonicalisation (+4.94 at the same angle, at a quarter of the compute). Every span
is worse than baseline at 0 degrees, which is the same peak-for-flatness trade seen in run 8.

| 13 | 2026-11-12 | A8.3 | ModelNet40, yaw sweep at **5 degree steps**, 72 angles, published protocol | `scripts/run_yaw_sweep.py configs/yaw_sweep_modelnet40_published_fine.yaml` | 0 | best 63.61% at 0 deg, worst **53.93% at 295 deg**, spread **9.68 pts**. On a view azimuth (multiples of 45): **60.59%** (n=8); between them: **56.41%** (n=64) | **Prediction confirmed.** Figure: `results/figures/yaw_sweep_fine.png` |

### The fine sweep confirms the mechanism

The prediction was a sawtooth with teeth every **45** degrees, because the eight azimuthal viewpoints are
two groups of four (0/90/180/270 and 45/135/225/315) whose union is 45-degree spaced. The curve has
exactly eight teeth and every peak sits on a multiple of 45: the eight highest angles in the whole sweep
are 0, 180, 225, 90, 45, 270, 135 and 355 degrees.

Folding the curve by candidate periods and measuring the within-bin variance gives 0.44 for a 45-degree
period against 3.70 for 30 degrees and 3.64 for 120 - about eight times tighter, so the 45-degree
structure is not an artefact of where the coarse sweep happened to sample.

Two honest qualifications:
- the peaks are **not equal in height**. 0 degrees (63.61%) and 180 degrees (62.52%) stand above the
  other six, which is what you would expect if ModelNet40's canonical alignment and the prompts fitted to
  it add something beyond the view geometry;
- folding by 90 and 180 degrees also gives low variance, because the view set is closed under a quarter
  turn and many of these objects are near-symmetric under a half turn. The decisive comparison is 45
  against 30 and 120, not against its own multiples.

This is the third independent route to the same conclusion - the coarse sweep's peaks, the span sweep's
no-op at offsets that are multiples of 90, and now the tooth spacing - so the mechanism does not rest on
one measurement.

| 14 | 2026-11-12 | A9 | **ScanObjectNN**, published protocol, ViT-B/16, three test splits | `scripts/run_eval.py configs/baseline_scanobjectnn_*.yaml` | 0 | **PB_T50_RS 35.60%** [33.90, 37.40] (n=2882) · **OBJ_ONLY 41.65%** [37.69, 45.61] (n=581) · **OBJ_BG 37.87%** [34.08, 41.82] (n=581) | see below. **Licence caveat applies: not reportable until Anas registers officially** |

### What the ScanObjectNN baselines say

**PB_T50_RS reproduces.** 35.60% against the upstream README's ScanObjectNN figure of 35.66 for ViT-B/16,
and the 35.36 quoted by secondary sources. So the single number the upstream README reports for
"ScanObjectNN" is almost certainly the PB_T50_RS split.

**The other two splits do not match the quoted numbers**, and the gap is instructive rather than a failure:
OBJ_ONLY measured 41.65% against a quoted 50.09, OBJ_BG 37.87% against 41.22. Note what the code does -
`trainers/best_param.py` holds **one** set of prompts and view weights per dataset and backbone, keyed
`scanobjectnn_vit_b16_*`, with no split in the key. Those were selected on one split's labels; applied to
the other two they transfer poorly.

That is the same finding as on ModelNet40, arriving from a different direction: the tuned artefacts are
fitted to the exact evaluation condition they were searched on, and they degrade when that condition
changes - whether the change is the object's pose or the dataset split. Before putting this in the report,
verify the quoted split-level numbers against the paper's own table (task A3), because the 50.09 and 41.22
figures come from secondary sources that already disagree with each other.

| 15 | 2026-11-12 | A9 | ScanObjectNN yaw sweeps, published protocol | `configs/yaw_sweep_scanobjectnn_{obj_only,pb_t50_rs}.yaml` | 0 | **OBJ_ONLY: spread 2.07 pts**, and 0 deg is *not* the best angle (210 deg is). **PB_T50_RS: spread 2.32 pts**, 0 deg best | no meaningful pose sensitivity on either split |
| 16 | 2026-11-12 | A9 | ScanObjectNN, six conditions, both splits | `configs/fix_scanobjectnn_*.yaml` | 0 | **no condition improves at any angle on either split.** Largest effect: canon_pca +2.24 [-0.34, +4.82] on OBJ_ONLY at 90 deg, interval includes zero | the fix does nothing here |

### A9: the main prediction was refuted, and the refutation supports the explanation

**Prediction 3, recorded before the runs, was that canonicalisation would help *more* on PB_T50_RS than on
OBJ_ONLY.** It helps on neither. Every paired comparison on both splits contains zero, except two small
losses at 0 degrees.

**Prediction 2 was also wrong**: OBJ_ONLY was expected to show ModelNet40-like pose sensitivity. It does
not - the spread is 2.07 points, and the canonical pose is not even the best angle, with 210 degrees above
it.

The consistent reading, and it strengthens rather than weakens the project:

1. **There is no pose sensitivity on either ScanObjectNN split to repair.** Both spreads, about 2 points,
   are the same size as the *plain* protocol's 1.54 on ModelNet40 - the condition where C7 already
   established that the fix correctly does nothing. So the fix behaving as a no-op here is exactly what
   the C7 control predicts, not a failure of the fix.
2. **Why is there no sensitivity? Because of what the prompts were tuned on.** `best_param.py` holds one
   `scanobjectnn_vit_b16` prompt set, and run 14 showed it matches the PB_T50_RS split - the split whose
   objects the dataset's authors **already rotated at random**. Prompts selected against randomly posed
   renders cannot encode a canonical appearance, so they are pose-agnostic, and accuracy barely moves when
   the object turns. ModelNet40's prompts were selected against canonically aligned renders and are
   strongly pose-dependent.
3. That is an independent confirmation of the factorial's revised explanation: **the pose sensitivity
   lives in the prompts, and tracks the pose distribution they were searched on.** The factorial showed it
   by removing the prompts; this shows it by changing the pose distribution they were fitted to. The two
   arrive at the same place from opposite directions.

Caveats to state with it: the prompts/weights key has no split in it, so OBJ_ONLY and OBJ_BG are being
evaluated with artefacts tuned for a different split (run 14), which depresses their accuracy and may also
flatten their curves; PB_T50_RS perturbs all three axes and rescales, so a yaw-only canonicaliser could not
fully repair it even if there were something to repair; and both ScanObjectNN splits are small
(581 objects), so their intervals are roughly four points wide and only a large effect would be visible.

**Licence caveat stands: none of A9 may be reported until the official registration is done.**

| 17 | 2026-11-12 | A10 | ModelNet40 yaw sweep, published protocol, **RN50 / RN101 / ViT-B/32** | `configs/yaw_sweep_modelnet40_{rn50,rn101,vit_b32}.yaml` | 0 | table below | three more reproduction points; prediction 1 confirmed, 2 refuted, 3 not supported |

### A10: the backbone ablation

| backbone | measured at 0 deg | published | spread | relative spread | on-grid | off-grid | gap |
|---|---|---|---|---|---|---|---|
| RN50 | 39.26% [37.36, 41.17] | 39.38 | 6.40 | 16.3% | 38.49 | 33.63 | **+4.86** |
| RN101 | 44.04% [42.10, 45.95] | 46.31 | 2.92 | 6.6% | 44.20 | 42.81 | **+1.38** |
| ViT-B/32 | 54.86% [52.88, 56.85] | 55.14 | 4.98 | 9.1% | 53.70 | 50.63 | **+3.07** |
| ViT-B/16 | 63.61% [61.71, 65.52] | 64.26 | 9.16 | 14.4% | 61.40 | 55.47 | **+5.93** |

(on-grid = angles that are multiples of 90; off-grid = the rest.)

**Reproduction: three of four inside the interval.** RN50, ViT-B/32 and ViT-B/16 all contain their
published figure. **RN101 does not**: 46.31 lies above our [42.10, 45.95]. That is a genuine, unexplained
2.3-point shortfall and must be reported as one rather than rounded away. Candidate explanations, none
tested: the channel width for RN101 in our pipeline (512) may be wrong - check `clip.build_model`'s output
dimension for that backbone before anything else - or its shipped prompt set may assume something our path
does not reproduce. **This is the first thing to check if there is time**; if it is a channel-width bug it
would invalidate the RN101 row only.

**Prediction 1 - confirmed, and it was the one that mattered.** Every backbone shows the on-grid advantage:
+4.86, +1.38, +3.07, +5.93 points. The sawtooth is not a quirk of ViT-B/16. Since all four backbones have
*different* searched prompts but share the projection geometry and the canonical-pose data they were
searched against, a flat backbone would have undermined the explanation. None is flat.

**Prediction 2 - refuted.** I predicted absolute spread would scale with accuracy. It does not: RN50 is the
weakest backbone (39.26%) yet has a larger spread (6.40) than both RN101 (2.92) and ViT-B/32 (4.98).
Accuracy does not determine pose sensitivity.

**Prediction 3 - not supported.** Relative spread ranges from 6.6% to 16.3%, a factor of 2.5, so the
sensitivity is not a constant property of the shared pipeline either.

**What 2 and 3 together mean.** Pose sensitivity varies per backbone in a way that neither accuracy nor the
shared geometry explains. The remaining candidate, consistent with everything else in this project, is the
**prompt set**: each backbone has its own, and they differ in how strongly they describe the canonical
appearance. RN101's prompts give the flattest curve and also the worst reproduction, which is suggestive
but not evidence. Testing it would mean running one backbone's prompts through another backbone - a clean
experiment, not yet run, and a good "unresolved question" for the report.

| 18 | 2026-11-12 | A10 follow-up | diagnosis of the RN101 shortfall - **no re-run, code inspection only** | `clip.build_model` dims; `configs/trainers/PointCLIPV2_ZS/*.yaml`; `trainers/post_search.py` | - | **not a channel-width bug.** Measured embedding dimensions are RN50 1024, RN101 512, ViT-B/32 512, ViT-B/16 512 - exactly what our pipeline assumes | see below |

### The RN101 shortfall: diagnosed, and it sharpens the central criticism

The first suspect was the channel width. It is wrong: the four backbones' real embedding dimensions match
ours exactly. The four trainer configs are also identical apart from batch size and worker count, neither
of which changes a result.

The explanation is elsewhere, and it matters for the whole report. **Our runs use the shipped
`best_param.py` prompts and weights and stop there. The published numbers additionally run
`--post-search`, which searches again at evaluation time.** Look at the scope of that search
(`trainers/post_search.py`, lines 100-122): for each of 2 passes, for each of 40 classes, for each of
**~1,198 candidate prompts** from `prompts/modelnet40_1000.json`, it swaps the prompt in, recomputes
accuracy **against the test labels**, and keeps the swap if accuracy improved. That is on the order of
**96,000 greedy accept-or-reject decisions taken against the test set**, starting from prompts that were
themselves already selected that way. A separate loop does the same for the ten view weights.

Our four measurements sit at or just below the published figures, by 0.12 (RN50), 2.27 (RN101), 0.28
(ViT-B/32) and 0.65 (ViT-B/16) points. That pattern is what the explanation predicts: we are measuring the
**pre-search starting point**, and the published figure is the **post-search endpoint**. RN101 is simply
the backbone where the extra hill-climbing bought the most.

**Revision to row 17.** It called RN101 an unexplained shortfall with a channel-width suspicion. The
channel width is now ruled out and the likely cause named. It is still not *proved*: proving it means
re-running the search and seeing RN101 climb to about 46.3.

**The experiment that would settle it, not yet run.** Re-implement the greedy prompt search on cached
features. Only one class's text vector changes per step, so each step is a rank-one update to the logits
and the whole 96,000-step search is cheap. Two numbers would come out of it: whether RN101 reaches its
published figure, and - more valuable for the report - **how many accuracy points the evaluation-time
search is worth on each backbone**. That is the quantity the central criticism currently describes
qualitatively.

| 4 | 2026-11-12 | - | ModelNet40, first 64 objects only | `configs/smoke_modelnet40.yaml` | 0 | 56.25% - **not reportable** | Smoke test of the harness. The test split is ordered by class, so the first 64 objects are not a sample of anything |

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

## Reading of runs 5-7, written before the fix is implemented

1. **The curve has four peaks, at 0, 90, 180 and 270 degrees.** Those are exactly the azimuths of the
   upstream projection's fixed viewpoints. Accuracy recovers when the object's yaw lands on the view grid
   and sags between the teeth: on-grid 61.40% against off-grid 55.47%, non-overlapping intervals. So the
   loss is not generic degradation from rotation; it is the object falling between fixed views. That is a
   mechanism, and it is what motivates the fix.
2. **Almost all of the sensitivity belongs to the test-tuned protocol, not to the projection.** The same
   pipeline with a fixed template prompt and uniform view weights varies by only 1.54 points across the
   whole sweep, against 9.16 points for the published protocol. The per-view weights were selected on
   canonically aligned test data, so they encode that pose; rotate the object and the weighting is wrong.
   **This is a second, sharper statement of the post-search problem**: the selected weights do not merely
   inflate the number, they inflate it in a way that does not survive a change of pose.
3. **C3 is only partially supported, and the report must say so.** The predicted direction holds in the
   mean - symmetric classes lose 1.84 points, oriented classes 6.17, a factor of about 3.4. But the detail
   does not match the simple story: the biggest losers are flat, box-like objects (`keyboard` +30,
   `glass_box` +30, `mantel` +26, `dresser` +26, `bed` +23), while `chair` and `desk`, which the hypothesis
   named explicitly, barely move (-2 each). Several classes **improve** under rotation (`xbox` -20,
   `tent` -14, `sofa` -7, `piano` -6), which a pure "rotation destroys information" account cannot explain.
   A better account is that the canonical pose is simply the pose the prompts and weights were fitted to,
   and some classes were fitted badly. Do not quietly rewrite the hypothesis afterwards - report the
   prediction, the result, and this revision.
4. The a priori symmetric set is imperfect: `stool` and `lamp` were marked symmetric and lose 13 points
   each, `tent` was marked symmetric and gains 14. The set was fixed from class names before any result was
   seen, and it stays as it was. Note the mismatch rather than editing the set.

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
