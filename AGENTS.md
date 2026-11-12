# Rules for any AI model working in this repository

This file is for **every** assistant: Claude Code, Antigravity, Cursor, Copilot, anything else. Read it
before you touch a file. It exists so that work can be handed between models without the project losing its
integrity. The project is graded on honesty and rigour, so damage here is not a bug - it is academic
misconduct.

If you are an assistant and you have read only one thing, read the eight rules.

## The eight rules

1. **Never write a number you did not measure.** Not in the README, not in the report, not in a commit
   message, not as a placeholder, not as "approximately". If an experiment has not run, the status stays
   `not run`. Fabricating or guessing a result is the single worst thing you can do in this repository.
2. **Published numbers and measured numbers never mix.** The paper's 64.22% is the paper's. Your measured
   baseline is yours. They go in separate columns, always labelled, even when they agree.
3. **`docs/REPRO_LOG.md` is append-only.** Add a new row for every run. Never edit or delete an existing
   row, never "tidy" the log, never remove a run that gave an inconvenient result. A failed or disappointing
   run is evidence and must survive.
4. **Never weaken a test to make it pass.** If `pytest` fails, the code is wrong, or the test's expectation
   was wrong *for a reason you can state in writing*. Deleting an assertion, loosening a tolerance, or
   marking a test skipped to get a green run is forbidden. Run `pytest tests -q` before every commit.
5. **Do not change the semantics of `src/yaw.py` or `src/metrics.py` without saying so loudly.** Every claim
   in the project rests on these two files. Bug fixes are welcome with a test that captures the bug.
   Refactors that alter behaviour, change conventions (the up axis, the angle range, the CI method), or
   "simplify" the maths are not.
6. **Stay inside your task.** Work one task at a time from `docs/ANTIGRAVITY_TASKS.md`, touch only the files
   that task lists, and do not start adjacent work you think is a good idea. If you believe the plan is
   wrong, write that in the task's Notes and stop. Do not switch method, dataset or paper on your own.
7. **Never commit data, checkpoints or upstream code.** `data/`, `third_party/` and `*.pth` are gitignored
   for a reason. Record the upstream commit hash, not a copy of someone else's repository. Never commit
   credentials or dataset files that came with a licence.
8. **Record what you did in `../AI_USE.md`.** The course requires AI assistance to be disclosed with the
   specific contribution of each tool. One row per working session. This is a course requirement, not a
   formality.

## What the project is (so you do not drift off it)

PointCLIP V2 does zero-shot 3D classification by rendering a point cloud to a few **fixed** viewpoints and
classifying the renders with frozen CLIP. Its benchmark, ModelNet40, is canonically aligned, so the reported
64.22% assumes the object already faces the camera. The project asks whether the claim survives arbitrary
object yaw, and whether cheap yaw canonicalisation recovers the loss.

The six claims under test are the table in `README.md` (C1-C6). Every experiment must serve one of them.
If a piece of work does not map to C1-C6, it is out of scope.

Three controls are not optional, because the whole result collapses without them:
- **same view budget**: the fix must be compared against the same number of CLIP forward passes without yaw
  diversity, or the gain is just extra compute;
- **clean condition**: the fix must be measured on the aligned data too, in case robustness costs accuracy;
- **ablation**: canonicalisation alone, ensembling alone, both.

## Hard requirements from the course brief

- The experiment must be runnable from this repository with exact commands.
- Every deviation from the original paper's setup (smaller backbone, fewer points, different CUDA, partial
  data) must be written down in `docs/REPRO_LOG.md` and carried into the report's deviations section.
- Do not tune anything on the test set. If a threshold or view count needs choosing, choose it on a
  validation split or state plainly that it was fixed a priori.
- A negative result is an acceptable outcome and must be reported as one. Do not keep searching for a
  configuration that finally wins and then present only that.

## Working agreement

- Python 3.12 in `.venv`. `torch` comes from the cu128 index - see `SETUP.md`. The RTX 5070 is Blackwell;
  a default `pip install torch` gives a CPU build and nothing will work.
- Each experiment = one script in `scripts/` + one config in `configs/` + one table in `results/` + one row
  in `docs/REPRO_LOG.md`. No experiment is run from a notebook or an ad-hoc shell one-liner.
- Set and record seeds. The bootstrap in `src/metrics.py` takes a seed; use it.
- Commit small, with messages that say what was measured, not what was attempted.
- If you are blocked, stop and write the blocker in the task Notes. A blocked task reported honestly costs an
  hour; a quietly invented workaround costs the grade.

## Who decides what

Anas decides the research direction, the topic and anything involving the submission. Models implement,
measure and write up. If a decision changes the research question, the datasets, the paper being reproduced,
or what gets submitted, it is Anas's call, not yours.
