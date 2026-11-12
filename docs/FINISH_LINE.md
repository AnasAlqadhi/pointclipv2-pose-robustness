# Finish line: everything left between here and submission

**Deadline: Wednesday 18 November 2026, before the Week 7 presentation session.** The report, the
repository and the slides are all due at that moment - the presentation is not a later deadline.

State as of 2026-11-12, end of day: **the experiments are done.** 19 logged runs, two datasets, four
backbones, a reproduction that now matches every published figure, a mechanism confirmed three ways, a fix
with four controls, and five predictions recorded in advance and refuted. Nothing below requires a GPU
except the optional items.

---

## The five things that must happen

### 1. Make the repository public (5 minutes, only Anas can)

The brief requires a public repository.

```
gh repo edit AnasAlqadhi/pointclipv2-pose-robustness --visibility public --accept-visibility-change-consequences
```

### 2. Register for ScanObjectNN, or cut it (30 minutes, only Anas can)

Our copy came second-hand from the upstream authors' Drive folder. Either register officially on the
ScanObjectNN site and state in the report where the files came from, or **delete section 7.8 from the
report**. Both are acceptable; presenting a second-hand copy as an official download is not.

### 3. Read the five papers and write section 2 (half a day)

`docs/LITERATURE.md` has the PDFs' location, the reading order, the time budget, the seven fields per
paper, and the targeted questions. This is the critical path - the experiments are far ahead of the
writing.

While reading PointCLIP V2, do task A3: record the table number and the printed numbers, and set the
"Verified in the PDF?" column in `REPRO_LOG.md`. Also settle whether the paper itself describes the
prompt and view-weight search - the wording of your central claim depends on the answer, and
`LITERATURE.md` explains how.

### 4. Write section 8, and check every number (half a day)

`docs/REPORT_DRAFT.md` has sections 1 and 3-7 written from measurements. Section 8 has raw material and
needs your judgement. Then go through section 7 line by line against `REPRO_LOG.md`. **Do not submit a
number you have not personally traced to a log row** - you have to defend these in the same session.

### 5. Build the slides (half a day)

`docs/PRESENTATION.md` has a 12-slide plan, the figure for each slide, the likely questions with honest
answers, and a rehearsal checklist. All figures already exist in `results/figures/`.

---

## Submission mechanics - do not leave these to Wednesday morning

- [ ] Report exported as **`Alqadhi_YZM5110_Research.pdf`**, with the **repository URL on page 1**
- [ ] Repository public, README current, `pip-freeze-dev.txt` present, exact commands in `SETUP.md`
- [ ] **Tag the submitted commit**: `git tag -a submission-v1 -m "YZM5110 submission" && git push origin submission-v1`
- [ ] `AI_USE.md` finished - the brief requires naming the tools and their specific contributions
- [ ] Class-sheet row completed: topic, paper title, paper URL, repository URL, report PDF
- [ ] Slides finished the night before, not the morning of

---

## Optional, only if sections 2 and 8 are already done

In value order. None of these is needed for a complete submission.

1. **Swap prompts across backbones.** Run RN101's prompts through ViT-B/16 and vice versa. Tests directly
   whether the per-backbone differences in pose sensitivity (7.6) come from the prompt sets. One new
   protocol variant and two runs, under an hour.
2. **Hand-written pose-neutral prompts.** Write 40 prompts that describe objects without reference to a
   viewpoint, and sweep. Tests the explanation in 7.2 head-on.
3. **Replicate the view-weight search** as well as the prompt search, to complete the measurement in 7.7.
4. **Track C, the robot clouds.** Brief R1 in `ANTIGRAVITY_TASKS.md`. Descoped when the deadline became
   the binding constraint; it would be a strong extension for the Week 12 project instead.

---

## If you continue this with another model

Point it at `AGENTS.md` first, then `docs/ANTIGRAVITY_TASKS.md`. The rules that matter: never write a
number that was not measured; keep published and measured numbers separate; `REPRO_LOG.md` is append-only;
never weaken a test; stay inside one task; and log the session in `AI_USE.md`.

Three traps that have already caught us once each, all recorded in the log: the clouds must be truncated
to their **first 1,024 points**; the up axis is **+Y**, not Z; and `pip install` of anything that depends
on torch can silently install a CPU build - check `torch.cuda.is_available()` afterwards.
