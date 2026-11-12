# Presentation plan - Week 7, 18 November 2026

Built on the same evidence as the report. The brief's criterion is "a defensible presentation", so the
test is not whether the slides look good but whether you can answer "how do you know?" for every number.

Assume 10-12 minutes plus questions unless told otherwise. One idea per slide. Every figure already exists
in `results/figures/`; do not make new ones for the talk.

## The slides

| # | Slide | What is on it | Say |
|---|---|---|---|
| 1 | Title | Title, your name, the repository URL | One sentence: can a training-free zero-shot 3D method survive an object that is not facing the camera? |
| 2 | The method in one picture | A few depth renders from `up_axis_check.png` (top row only) | Cloud goes in, ten fixed views come out, frozen CLIP classifies them. No 3D training at all |
| 3 | The benchmark's hidden assumption | Two renders of the same chair, 0 and 60 degrees | ModelNet40 is canonically aligned. Every chair faces the same way. A robot never gets that |
| 4 | Reproduction | Table: 63.61% [61.71, 65.52] against 64.22 / 64.26 | It reproduces. The published figure is inside my interval. State the three deviations in one line |
| 5 | **The first finding** | 63.61% vs 32.50% | Same pipeline. Swap the shipped prompts and view weights for one template prompt and uniform weights, and half the accuracy disappears. Those shipped values were selected on the test labels |
| 6 | **The sawtooth** | `yaw_sweep_fine.png` | Rotate the object and accuracy falls up to 9.16 points. Every peak sits on one of the projection's view azimuths. This is objects falling between fixed views |
| 7 | Which half causes it | The 2x2 factorial table | I predicted the view weights. I was wrong: the sensitivity follows the prompts. They describe the object *as seen from the canonical viewpoints* |
| 8 | **The fix** | `fix_summary.png` | Canonicalise the yaw before projection. Spread 8.14 to 0.16 points, mean 58.69 to 60.66, **no extra compute**. It costs 2.96 points at the canonical pose - say this out loud, do not let a marker find it |
| 9 | The controls | Four lines: same-budget, clean-condition, ablation, negative control | The same 40 passes without yaw diversity gain nothing. The fix does nothing under the protocol that has no pose sensitivity. That is what makes the gain believable |
| 10 | It generalises | `backbones.png` | All four backbones show it. And replicating the evaluation-time search recovers all four published numbers, which sizes the test-set tuning at 0.2 to 2.2 points |
| 11 | What I got wrong | The list of refuted predictions | Five predictions recorded in advance and refuted. Say why that is a feature: they were recorded before the runs and are still in the log |
| 12 | Limitations and next | Three bullets | One dataset family, yaw only, ScanObjectNN's licence step. Next: swap prompts across backbones; hand-written pose-neutral prompts; robot clouds |

## Questions you should expect, and the honest answers

- **"Isn't your fix just worse at 0 degrees?"** Yes, by 2.96 points. It trades the benchmark's supplied
  alignment for pose independence, and the mean over angles still improves. Whether that trade is right
  depends on whether deployment gives you alignment.
- **"Are you accusing the authors of cheating?"** No. The released code at commit `17df8a26` selects
  prompts and view weights against the evaluated set's labels; that is an observation about the artefact,
  not a claim about intent. Say it exactly that way.
- **"How do you know the up axis is Y?"** I rendered four objects about all three candidate axes and
  looked. Figure in the repository. The code originally assumed Z, which was wrong.
- **"Why not just use more views?"** Measured: the 40-pass yaw ensemble is worse than the 10-pass
  canonicalisation at every angle, and a same-budget non-yaw ensemble gains nothing.
- **"Is the per-class story clean?"** No. Direction holds in the mean, but several oriented classes gain
  under rotation and chair and desk barely move. Reported as partial support.
- **"How much of this did AI do?"** It built the harness and ran the experiments; the topic, the
  interpretation and the write-up are mine, and `AI_USE.md` records the split. Expect this question and
  answer it without defensiveness.

## Preparation checklist

- [ ] Rehearse once with a timer. The sawtooth slide and the fix slide are the two that must land
- [ ] Be able to state, without notes: 63.61 / 32.50 / 9.16 / 0.16 / 2.96
- [ ] Have `REPRO_LOG.md` open on a second screen for "where does that number come from"
- [ ] Have the repository open, not just the slides - a runnable repo is part of the grade
- [ ] Upload the report before the session; the presentation is in the same session, not later
