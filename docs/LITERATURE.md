# The five papers: where they are, how to read them, what to write down

The brief wants **at least five papers**, each with seven fields, **plus an account of how you found and
selected them**. This file is the working sheet. Fill the tables as you read; section 2 of the report is
then mostly assembled rather than written from nothing.

## Where the PDFs are

`D:\Projects\school\courses\2026-fall-YZM5110-multimodal-learning\papers\`

| File | Paper | Why it is in the set |
|---|---|---|
| `01_CLIP_Radford2021.pdf` | Learning Transferable Visual Models From Natural Language Supervision | the frozen backbone everything here depends on |
| `02_PointCLIP_Zhang2022.pdf` | PointCLIP (CVPR 2022) | the original "project the cloud into CLIP" idea |
| `03_PointCLIPV2_Zhu2023_TARGET.pdf` | **PointCLIP V2 (ICCV 2023)** | **the reproduction target - read this one hardest** |
| `04_ULIP_Xue2023.pdf` | ULIP (CVPR 2023) | the competing family: train a 3D encoder into CLIP space |
| `05_ARTPoint_Wang2022.pdf` | ART-Point (CVPR 2022) | rotation robustness: why augmentation is not invariance |
| `06_LocoTrans_2024.pdf` | LocoTrans (2024) | rotation-invariant point-cloud analysis, the other tradition |
| `07_TipAdapter_Zhang2022.pdf` | Tip-Adapter | training-free adaptation, the cache-based alternative |

Seven downloaded for a minimum of five, so you can drop the two that serve you least after reading. They
are deliberately **outside** the git repository - do not commit someone else's PDFs.

## The order to read them, and the time to spend

Half a day, not two. You are reading to build an argument, not to master seven papers.

1. **PointCLIP V2 (03)** - 90 minutes, properly. It is the target.
2. **PointCLIP (02)** - 30 minutes. Skim the method, read the limitations.
3. **ART-Point (05)** or **LocoTrans (06)** - 45 minutes for whichever reads better. You need one voice
   from the rotation-robustness tradition, not both.
4. **ULIP (04)** - 30 minutes. You need to be able to say what the competing family does differently.
5. **CLIP (01)** - 20 minutes, targeted. You are not reviewing CLIP; you need its assumptions about
   natural images and its zero-shot protocol.
6. **Tip-Adapter (07)** - 20 minutes, only if you want a sixth row.

## The one task that is not optional

**Verify the numbers you are going to cite.** The log already records that the paper says 64.22 and the
authors' own repository says 64.26. Open `03` and find the zero-shot classification table:

- record the table number;
- record ModelNet40, ModelNet10 and the three ScanObjectNN splits as printed;
- record which backbone and how many views each number uses;
- then correct the "published numbers" table in `REPRO_LOG.md` and set its "Verified in the PDF?" column.

While you are in that paper, look specifically for **whether it describes the prompt and view-weight
search, and on which data it was performed**. That is the central claim of your report. If the paper is
explicit about it, your framing is "the paper states this and its consequences were not measured". If the
paper is silent or implies a held-out split, your framing is "the released code does X, which the paper
does not describe". Those are very different sentences and the right one depends on what you read.

## What to write down for each paper - the seven fields

Copy this block once per paper. Keep it short; a full page per paper is too long for the matrix.

```
### <paper>, <venue year>, <arXiv or DOI>

1. Question and contribution - what problem, and what is actually new?
2. Modalities and assumptions - what information exists at training time and at deployment?
   (For this project, note especially: does it assume a canonical object pose?)
3. Mechanism and objective - how do the modalities interact, and what signal trains the model?
4. Data and protocol - datasets, preprocessing, split units, evaluation conditions.
   (Note especially: is anything selected using test data?)
5. Baselines and evidence - are the comparisons fair, do the ablations support the explanation?
6. Resources and reproducibility - hardware, code, checkpoints, configuration details.
7. Limitations and connections - where does it fail, and how does it relate to the others here?
```

## Targeted questions - the things that matter for *your* argument

Read for these specifically; they are what turn a summary into a synthesis.

**PointCLIP V2 (03)**
- Does it report any result under a non-canonical object pose? (Our search suggests no. Confirm.)
- How does it describe the prompt and view-weight selection, and on what data?
- Does it state that ScanObjectNN's PB_T50_RS split is already rotation-perturbed?

**PointCLIP (02)**
- What does it say its own limitations are? Is pose among them?
- How many views, at what azimuths? Compare with V2's ten.

**ART-Point (05) / LocoTrans (06)**
- The claim you want: rotation augmentation produces robustness over a narrow range only, which is why
  invariant designs exist. Find where it is argued, and note the evidence given.
- Do either of them evaluate a CLIP-based or projection-based method? (We believe not - that absence is
  your gap, so check it rather than assert it.)

**ULIP (04)**
- Does training a 3D encoder into CLIP space remove the pose problem, or relocate it?
- What does it cost in data and compute compared with a training-free method?

**CLIP (01)**
- The zero-shot protocol: how are prompts chosen, and on what data? (Prompt engineering and ensembling are
  discussed in the paper - this is directly relevant to your prompt finding.)

## How you found the papers - draft, for you to correct

The brief asks for this explicitly, so write it in your own words. The honest account of what happened:

> The reproduction target was chosen first, from the course topic sheet, as the only untaken area involving
> 3D data. From it I followed two directions: backwards through its citations to the original projection
> idea and to CLIP itself, and sideways to the competing approach that trains a 3D encoder into the shared
> space. A separate search covered rotation robustness of point-cloud models. The two bodies of work do not
> cite each other, and that gap motivated the experiment.

Correct it to match what you actually did - including that an AI assistant ran some of the searches, which
`AI_USE.md` records.

## Then

Paste the seven-field blocks into section 2 of `REPORT_DRAFT.md`, write two or three paragraphs of
synthesis around them (not a summary each - a comparison), and the matrix is done.
