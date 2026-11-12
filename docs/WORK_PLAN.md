# The complete work plan - how to finish this project anywhere

Written 2026-11-12 so the work can continue in Antigravity, another Claude session, or by hand, without
this conversation. **Deadline: Wednesday 18 November 2026, before the Week 7 presentation session.** The
report, the public repository and the slides are all due at that moment.

Repository (now **public**): https://github.com/AnasAlqadhi/pointclipv2-pose-robustness
Working folder: `D:\Projects\school\courses\2026-fall-YZM5110-multimodal-learning`

---

## 0. The thirty-second briefing, for whoever picks this up

The project reproduces **PointCLIP V2** (zero-shot 3D classification: render a point cloud to ten fixed
depth views, classify with frozen CLIP) and asks whether its claim survives an object that is not
canonically aligned.

What has been established, all measured and logged:

- the published numbers reproduce on **all four backbones** once the evaluation-time search is included;
- **about half** the headline accuracy comes from prompts and view weights selected on the test labels
  (63.61% with them, 32.50% with a plain template prompt);
- rotating the object costs up to **9.16 points**, and the accuracy curve peaks exactly on the
  projection's **45-degree view azimuths** - objects fall between fixed views;
- the sensitivity lives in the **prompts**, not the view weights (shown twice, two ways);
- **yaw canonicalisation** removes the pose dependence (spread 8.14 to 0.16 points, mean 58.69 to 60.66)
  for **no extra compute**, at a cost of 2.96 points in the canonical pose;
- four controls passed, and **five predictions recorded in advance were refuted** and kept in the log.

**The experiments are finished.** Everything below is reading, writing and slides.

---

## 1. The rules, which apply to every model and every session

These are in `AGENTS.md` and are not negotiable, because the project is graded on honesty:

1. Never write a number that was not measured. If something has not run, its status stays "not run".
2. Published numbers and measured numbers never mix; label every figure's source.
3. `docs/REPRO_LOG.md` is **append-only**. Add rows; never edit or delete, including inconvenient runs.
4. Never weaken a test to make it pass. Run `pytest tests -q` before any commit (29 pass, 3 skip).
5. One task at a time; touch only that task's files. If the plan looks wrong, write it down and stop.
6. Never commit `data/`, `third_party/`, `.venv/`, or other people's PDFs.
7. Log every working session in `AI_USE.md` - the brief requires disclosure with specific contributions.

**Three traps that have already caught us once each:**
- clouds must be truncated to their **first 1,024 points** (upstream does this; using 2,048 costs 7 points);
- the up axis is **+Y**, not Z (established by rendering, figure in the repository);
- any `pip install` touching torch can silently install a CPU build - check `torch.cuda.is_available()`.

---

## 2. Day by day

### Friday 13 November - the papers (the critical path)

**Goal: section 2 of the report drafted.** This is the only part that cannot be accelerated by a tool,
because you must be able to defend the matrix.

The seven PDFs are in `..\papers\`. Budget half a day, not two:

| order | paper | minutes | read for |
|---|---|---|---|
| 1 | `03_PointCLIPV2_Zhu2023_TARGET.pdf` | 90 | the method, and the two questions below |
| 2 | `02_PointCLIP_Zhang2022.pdf` | 30 | the original idea; its stated limitations |
| 3 | `05_ARTPoint` **or** `06_LocoTrans` | 45 | why augmentation is not invariance |
| 4 | `04_ULIP_Xue2023.pdf` | 30 | the competing family: train a 3D encoder into CLIP space |
| 5 | `01_CLIP_Radford2021.pdf` | 20 | its zero-shot protocol and how prompts are chosen |

**The two questions to answer while in paper 03 - these decide your wording:**

1. *Does the paper describe the prompt and view-weight search, and on what data?*
   - If it does → write "the paper states this, and its consequences under rotation were never measured".
   - If it is silent or implies a held-out split → write "the released code at commit `17df8a26` does X,
     which the paper does not describe".
   Either is defensible. Guessing is not.
2. *Does the paper report any result under a non-canonical object pose?* (We believe not. Confirm it,
   because your novelty claim rests on it.)

Also do task A3 while you are there: record the table number and the printed numbers for ModelNet40,
ModelNet10 and the three ScanObjectNN splits, then set the "Verified in the PDF?" column in
`docs/REPRO_LOG.md`. Secondary sources already disagree with each other, so only the PDF counts.

For each paper, fill the seven-field block in `docs/LITERATURE.md` (question and contribution; modalities
and assumptions; mechanism and objective; data and protocol; baselines and evidence; resources and
reproducibility; limitations and connections). Keep each to a short paragraph.

**Then write the synthesis**, two or three paragraphs, not a summary each. The argument, for you to verify
rather than assume: two bodies of work exist - CLIP-projection methods and point-cloud rotation robustness -
and they do not cite each other. Your experiment sits in that gap.

**Done when:** section 2 of `docs/REPORT_DRAFT.md` contains five or more seven-field blocks plus the
synthesis and the account of how you found the papers, and the log's verification column is filled.

### Saturday 14 November - section 8 and verification

**Morning: write section 8.** Raw material is already in the draft. Suggested shape - one paragraph on
what was established, one on what it cost, one on what is open. The three claims that are safe because
each has a measurement behind it:

1. the published result reproduces exactly once the evaluation-time search is included;
2. a large part of the headline comes from prompts selected on test data, and that part does not survive
   a change of pose;
3. a yaw canonicalisation costing no extra compute removes the pose dependence, at the price of the
   benchmark's own alignment.

Name the open questions as open: swap prompts across backbones; hand-written pose-neutral prompts; extend
the canonicaliser beyond yaw; the robot clouds (Track C, descoped on time grounds).

**Afternoon: verify every number.** Go through section 7 line by line against `docs/REPRO_LOG.md`. For
each number ask: which run produced it, and is it labelled as measured or published? **Do not submit a
number you have not traced yourself** - you defend these in the same session as the report.

**Done when:** sections 2 and 8 are written and every figure in section 7 is traced to a log row.

### Sunday 15 November - the slides

Follow `docs/PRESENTATION.md`: 12 slides, the figure named for each, what to say, and the six likely
questions with honest answers. All figures already exist in `results/figures/` - do not make new ones.

The two slides that must land are the **sawtooth** (`yaw_sweep_fine.png`) and the **fix**
(`fix_summary.png`). Rehearse once with a timer.

**Done when:** the deck exists and you have run it once end to end out loud.

### Monday 16 November - polish and submission mechanics

- export the report as `Alqadhi_YZM5110_Research.pdf` with the **repository URL on page 1**;
- finish `AI_USE.md` (tools named, specific contributions stated);
- update the repository `README.md` claim table if anything changed;
- `git tag -a submission-v1 -m "YZM5110 submission" && git push origin submission-v1`;
- fill the class-sheet row: topic, paper title, paper URL, repository URL, report PDF.

### Tuesday 17 November - buffer

Deliberately empty. Something always slips. If nothing has, the optional extensions in section 4 below
are available, in value order.

### Wednesday 18 November - submit before the session, then present

---

## 3. How to hand a task to Antigravity (or any other model)

Open the repository folder, then give it this, filling in the one task:

> Read `AGENTS.md` first and follow it exactly, then read `docs/WORK_PLAN.md` and
> `docs/ANTIGRAVITY_TASKS.md`. Work on **one task only: <task>**. Touch only the files that task lists.
> Do not invent or estimate any number - if something has not been measured, say so. `docs/REPRO_LOG.md`
> is append-only. Run `pytest tests -q` before committing and do not weaken a test to make it pass. When
> you finish, add a row to `docs/ANTIGRAVITY_TASKS.md`'s session log and a row to `../AI_USE.md`. If you
> get blocked, write the blocker in the task's Notes and stop rather than improvising.

**Tasks that are safe to delegate**, because they are mechanical and checkable:
- regenerating figures (`python scripts/make_figures.py`, `python scripts/plot_yaw_sweep.py`);
- formatting the comparison matrix once **you** have written its content;
- the optional experiments in section 4, each of which has a written brief;
- proof-reading the report against the log for unlabelled or untraceable numbers.

**Tasks to keep for yourself:**
- reading the papers and writing section 2;
- writing section 8;
- deciding the wording of the central claim (it depends on what the paper says);
- anything that changes what is submitted.

---

## 4. Optional extensions, in value order

None is required. Start one only if sections 2 and 8 are already written.

1. **Swap prompts across backbones** (~1 hour). Run RN101's shipped prompts through ViT-B/16 and the
   reverse. Tests directly whether the per-backbone differences in pose sensitivity come from the prompt
   sets - currently the leading hypothesis with no direct evidence. Needs one new protocol variant in
   `src/pcv2.py` that takes the prompt key and the backbone separately.
2. **Hand-written pose-neutral prompts** (~2 hours). Write 40 prompts that describe each class without
   reference to a viewpoint, then sweep. Tests the explanation head-on: if the prompts are what encode
   canonical appearance, neutral prompts should be flatter under rotation.
3. **Replicate the view-weight search** (~1 hour) as well as the prompt search, to complete the
   measurement of what evaluation-time tuning is worth.
4. **Track C, the robot clouds** (one day, risky). Brief R1 in `docs/ANTIGRAVITY_TASKS.md`. Better saved
   for the Week 12 project.

---

## 5. Where everything is

| What | Path |
|---|---|
| what is left, one page | `docs/FINISH_LINE.md` |
| **this plan** | `docs/WORK_PLAN.md` |
| the report | `docs/REPORT_DRAFT.md` |
| the papers and how to read them | `docs/LITERATURE.md`, PDFs in `..\papers\` |
| the slides plan | `docs/PRESENTATION.md` |
| every run, append-only | `docs/REPRO_LOG.md` |
| what the upstream code does | `docs/UPSTREAM_NOTES.md` |
| task state for any model | `AGENTS.md`, `docs/ANTIGRAVITY_TASKS.md` |
| figures | `results/figures/` (7 of them) |
| raw results | `results/*.json`, predictions in `*.npz` |
| environment setup | `SETUP.md`, exact versions in `docs/pip-freeze-dev.txt` |

Backups: GitHub (public, tagged `work-complete-2026-11-12`) and
`C:\Users\anasm\OneDrive\Backups\YZM5110-pointclipv2`. Re-run `bash scripts/backup.sh` after any work.

---

## 6. If something breaks

| Symptom | Cause and fix |
|---|---|
| `import torch` fails with WinError 4551 | Smart App Control blocking a fresh DLL. **Try the import again** - it cleared on the second attempt. Never disable Smart App Control; it cannot be re-enabled without reinstalling Windows |
| `torch.cuda.is_available()` is False | a CPU wheel got pulled in. Reinstall from the cu128 index (`SETUP.md` section 1) |
| accuracy is ~57% instead of ~63.6% | the clouds are not truncated to 1,024 points |
| a yaw experiment shows objects tipping over | the up axis is wrong; it is **+Y** (`src/yaw.py`, `UP_AXIS = 1`) |
| `torch_scatter` will not install | it is not needed; the native shim in `src/pcv2.py` is tested in `tests/test_scatter_equivalence.py` |
| a number cannot be traced | it does not go in the report. Re-run the script in `scripts/` that produces it |
