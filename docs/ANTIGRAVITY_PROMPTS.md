# Ready-to-paste prompts for Antigravity (or any other model)

Open the folder `D:\Projects\school\courses\2026-fall-YZM5110-multimodal-learning\project` first, so the
model can see `AGENTS.md`, `docs/` and `results/`. Then paste one of these. **One task per prompt** - do
not merge them.

---

## 1. The standard opener (use this for any task)

```
Read AGENTS.md first and follow it exactly. Then read docs/WORK_PLAN.md for context and
docs/ANTIGRAVITY_TASKS.md for task state.

Work on ONE task only: <PUT THE TASK HERE>.

Rules, which override anything else you think is a good idea:
- Never write a number that was not measured. If something has not run, say "not run".
- Published numbers and measured numbers never mix; label the source of each.
- docs/REPRO_LOG.md is append-only. Add rows; never edit or delete one, including runs
  that failed or gave a disappointing result.
- Never weaken, skip or delete a test to make it pass. Run `pytest tests -q` before you
  commit; it should report 29 passed, 3 skipped.
- Touch only the files this task needs. Do not start adjacent work.
- Never commit data/, third_party/, .venv/, or PDF papers.
- The environment is .venv with torch 2.11.0+cu128. Clouds are truncated to their first
  1024 points. The up axis is +Y (UP_AXIS = 1 in src/yaw.py).

When you finish: add a row to the session log at the bottom of docs/ANTIGRAVITY_TASKS.md
and a row to ../AI_USE.md saying what you did and what Anas did.

If you get blocked, write the blocker in that task's Notes in docs/ANTIGRAVITY_TASKS.md
and STOP. Do not switch method, dataset or paper on your own.
```

---

## 2. Check the report's numbers (safe, useful, no GPU)

```
<standard opener, with the task below>

Task: verify every number in section 7 of docs/REPORT_DRAFT.md.

1. Run `python scripts/collect_numbers.py` to regenerate docs/NUMBERS.md from the result files.
2. Go through section 7 one number at a time. For each, check:
   (a) it appears in docs/NUMBERS.md with the same value;
   (b) it is labelled as measured or as published, and published ones name their source;
   (c) an accuracy carries its confidence interval.
3. Produce a list of every mismatch: the number, where it is in the report, and what
   NUMBERS.md says. DO NOT silently correct the report - report the list to Anas and let
   him decide.

Watch for three specific things: numbers from the search-replication table must not be
mixed with the sweep tables (float32 versus half precision, up to 0.24 points apart);
smoke-test runs are marked not reportable and must not appear anywhere in the report;
section 7.8 (ScanObjectNN) is only valid if Anas has registered officially.
```

---

## 3. Format the comparison matrix (after Anas has written its content)

```
<standard opener, with the task below>

Task: format section 2 of docs/REPORT_DRAFT.md.

Anas has written the seven-field notes for each paper in docs/LITERATURE.md. Turn them
into one comparison matrix table with one row per paper and these columns: question and
contribution; modalities and assumptions; mechanism and objective; data and protocol;
baselines and evidence; resources and reproducibility; limitations and connections.

Keep every cell to one or two short sentences, and keep his wording - do not rewrite his
judgements, do not add claims he did not make, and do not add any paper he did not read.
If a cell is empty in his notes, leave it empty and list it at the end so he can fill it.
```

---

## 4. Regenerate the figures (needs the GPU only for the failure gallery)

```
<standard opener, with the task below>

Task: regenerate all report figures.

Run `python scripts/plot_yaw_sweep.py` and `python scripts/make_figures.py`. Confirm that
results/figures/ contains these seven files and that each one opens and is readable:
up_axis_check.png, yaw_sweep.png, yaw_sweep_fine.png, yaw_per_class.png, backbones.png,
fix_summary.png, failure_gallery.png.

Report anything that looks wrong - clipped titles, overlapping labels, unreadable axes -
but do not redesign a figure beyond fixing what is actually broken.
```

---

## 5. An optional experiment: swap prompts across backbones (needs the GPU)

This is the most valuable remaining experiment. Only start it once sections 2 and 8 are written.

```
<standard opener, with the task below>

Task: test whether the per-backbone differences in pose sensitivity come from the prompt
sets rather than the encoders.

Background: section 7.6 of docs/REPORT_DRAFT.md shows all four backbones have the same
45-degree sawtooth but very different spreads (RN101 2.92 points, ViT-B/16 9.16), and
neither accuracy nor the shared geometry explains the difference. The leading hypothesis,
with no direct evidence yet, is that each backbone's own searched prompt set differs in how
strongly it describes the canonical pose.

The experiment: run ViT-B/16 with RN101's shipped prompts, and RN101 with ViT-B/16's, each
over the usual 12-angle yaw sweep, and compare the spreads against the matched runs in
results/.

WRITE THE PREDICTION IN docs/ANTIGRAVITY_TASKS.md BEFORE YOU RUN ANYTHING, and do not edit
it afterwards. If the hypothesis is right, the spread should travel with the prompts rather
than with the backbone.

Implementation: src/pcv2.py's build_pipeline currently derives the best_param key from the
backbone. Add an optional argument that lets the prompt key and the backbone differ. Do not
change the behaviour of any existing protocol - the existing results must stay reproducible.
Add a test that the default behaviour is unchanged.
```

---

## 6. What NOT to delegate

Keep these for yourself. They are the parts the grade actually rests on, and a model cannot defend them
in the room on Wednesday:

- reading the five papers and writing section 2;
- writing section 8;
- deciding how to word the central claim (it depends on what the paper itself says about the search);
- deciding whether ScanObjectNN stays in the report;
- anything that changes what gets submitted.
