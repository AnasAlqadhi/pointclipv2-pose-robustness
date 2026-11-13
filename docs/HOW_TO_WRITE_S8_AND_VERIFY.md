# How to do the two jobs left: verify section 7, and write section 8

Both are mechanical once you know the procedure. Budget an afternoon for the pair.

---

## Part A. Verify every number in section 7 (about 60 minutes)

Do this **first**. Writing conclusions on numbers you have not checked is how a defensible project
becomes an indefensible one, and it is the part a marker can test in thirty seconds.

### The tool that makes it easy

```bash
python scripts/collect_numbers.py
```

That regenerates `docs/NUMBERS.md` **straight from the result files**. It is machine-generated, so it
cannot have a typo that the report also has. Verification is then a comparison between two documents, not
a hunt through JSON.

### The procedure

Open `docs/REPORT_DRAFT.md` section 7 and `docs/NUMBERS.md` side by side. Go through the report **one
number at a time**, in order, and for each one ask three questions:

1. **Is it in `NUMBERS.md`, with the same value?**
   - yes → tick it, move on
   - no → either it came from somewhere else (then the report must say where - for example the on-grid
     versus off-grid averages, which are computed across angles) or it is wrong and must be corrected or
     removed.
2. **Is it labelled measured or published?** The brief requires these never to blur. Every published
   figure in your text should carry its source - "the paper", or "the upstream README" - because those two
   disagree (64.22 against 64.26).
3. **Does it carry an interval, where it is an accuracy?** A bare accuracy with no interval invites the
   reader to over-read small differences. Section 7 should have an interval on essentially every number.

Three specific things to check, because they are the easiest to get wrong:

- **Do not mix the sweep tables with the search-replication table.** The replication accumulates in
  float32 while the pipeline runs in half precision, so its starting points differ by up to 0.24 points
  (ViT-B/16 reads 63.74 there, 63.61 in the sweeps). Both are correct; they measure slightly different
  arithmetic. Say which table a number comes from.
- **Smoke tests are not results.** `smoke_modelnet40` (64 objects) and `smoke_factorial` (96) appear in
  `NUMBERS.md` marked not reportable. Neither may appear in the report.
- **ScanObjectNN section 7.8** stays only if you have registered officially. Otherwise delete the whole
  subsection, and also remove ScanObjectNN from the contributions list in section 1.

### When you are done

Add one line at the top of section 7: "All figures in this section were checked against
`docs/NUMBERS.md`, regenerated from the result files on <date>." That sentence is cheap and it tells a
marker exactly how much to trust the rest.

---

## Part B. Write section 8 (about 90 minutes)

The brief asks for "limitations, conclusions, and unresolved questions". Three or four paragraphs is
right; two pages is too many. **Write it yourself** - it is the section where the marker hears your
judgement rather than your results, and it is the easiest section in which borrowed phrasing shows.

### The structure that works

**Paragraph 1 - what was established.** Three claims, each with its number. Keep them narrow:

- the published result reproduces on all four backbones once the evaluation-time search is included;
- roughly half the headline accuracy comes from prompts and view weights selected on the test labels
  (63.61% against 32.50%), and that part does not survive a change of object pose;
- a yaw canonicalisation costing no extra CLIP passes removes the pose dependence (spread 8.14 to 0.16
  points), at the cost of 2.96 points in the canonical pose.

**Paragraph 2 - what it cost, and where the claims stop.** The honest boundaries:

- the fix is a **trade**, not a free improvement - it gives up the alignment the benchmark supplies;
- PCA canonicalisation has an unavoidable 180-degree ambiguity: it recovers a direction, not an
  orientation;
- the canonicaliser corrects **yaw only**, so a dataset perturbed about all three axes (PB_T50_RS) is
  outside what it can repair;
- one dataset family, one projection method, inference only, a single seed and a single GPU;
- the explanation for *why* the prompts carry the pose sensitivity is **consistent with** the evidence,
  not proved by it;
- the upstream protocol's use of test labels is an observation about the **released code at commit
  `17df8a26`**, not a claim about anyone's intent. Write that sentence carefully.

**Paragraph 3 - what you would do next, specifically.** Vague future work is worth nothing; a named
experiment with a stated prediction is worth a lot. You have four, already specified:

- run one backbone's prompts through another backbone, to test whether the per-backbone differences in
  pose sensitivity come from the prompt sets;
- hand-write pose-neutral prompts and sweep, to test the explanation head-on;
- extend the canonicaliser beyond yaw, so three-axis perturbations come into range;
- evaluate on robot-collected LiDAR clouds, which was descoped when the deadline became binding.

**Optional paragraph 4 - what you got wrong.** Five predictions were recorded before their runs and
refuted: that the view weights carried the pose sensitivity; that a 90-degree ensemble span would be a
no-op; that spread would scale with accuracy; that relative spread would be constant; and that
canonicalisation would help more on PB_T50_RS. Say so plainly, and say that each was recorded before the
run and left in `docs/REPRO_LOG.md`.

Most students have no such paragraph, because most students decide what they predicted after seeing the
results. Yours is evidence that the experiments were run honestly, and the brief rewards exactly that:
*"I will look at how you reached your conclusions, not only at whether your final score improved."*

### Three traps in section 8

1. **Do not overclaim beyond ModelNet40.** You measured one projection method on one dataset family.
   "PointCLIP V2 is pose-sensitive under these conditions" is defensible; "projection-based 3D recognition
   is fragile" is not.
2. **Do not soften the −2.96.** State it in the same sentence as the gain. A marker who finds it after
   you have claimed a clean win will discount everything else.
3. **Do not write "future work could explore..."** Name the experiment, and say what result would support
   or undermine your explanation.

---

## After both parts

```bash
python scripts/collect_numbers.py      # refresh, in case anything changed
pytest tests -q                        # 29 passed, 3 skipped
bash scripts/backup.sh                 # mirror to OneDrive
git add -A && git commit -m "Sections 2 and 8 written; section 7 verified against NUMBERS.md"
git push
```

Then the slides, from `docs/PRESENTATION.md`.
