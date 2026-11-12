# Notes on the upstream code (PointCLIP_V2)

Clone: `https://github.com/yangyangyang127/PointCLIP_V2`
Commit used: `17df8a269c14cb1601b596b3bc19274964fbd0d9` (2025-08-12)
Read on 2026-11-12 before any experiment was run.

Relevant layout (`zeroshot_cls/`): `main.py` entry point, `trainers/zeroshot.py` the pipeline,
`trainers/mv_utils_zs.py` the depth projection, `trainers/post_search.py` the prompt and view-weight search,
`datasets/modelnet40.py` and `datasets/scanobjnn.py` the two loaders, `clip/` a bundled copy of OpenAI CLIP,
`Dassl3D/` the training framework it is built on.

The published run is `zeroshot_cls/zeroshot_cls.sh`: trainer `PointCLIPV2_ZS`, backbone `vit_b16`,
`--no-train --zero-shot --post-search`.

## Finding 1: `--post-search` tunes on the evaluation set's labels

This is the most important thing to know before reproducing, and it must be in the report.

In `trainers/post_search.py`, both `search_prompt_zs` (line 67) and `search_weights_zs` (line 127) load
`labels.pt` from the output directory - the labels of the set that was just evaluated - and then loop over
candidate prompts and view weights keeping whichever maximises `accuracy(logits, labels)` (lines 93-122 and
145-177). The reported figure is printed as "After prompt search / After view weight search, zero-shot
accuracy".

So the headline number is obtained by selecting hyper-parameters (prompt choice, per-view weights) **to
maximise accuracy on the test labels**. There is no separate validation split in this path. Whatever one
calls it, it is not a clean zero-shot protocol, and it inflates the number by an unknown amount.

**What this means for the project:**

1. The baseline must be reported **twice**: with `--post-search` (as published, comparable to 64.22%) and
   **without** it (the honest zero-shot number, using fixed prompts and uniform view weights). The gap
   between them is itself a result worth reporting, and it belongs in the comparison matrix under
   "baselines and evidence: are comparisons fair?".
2. Every yaw experiment and every condition of the fix must use the **same** protocol as whichever baseline
   it is compared against. Mixing a post-searched baseline with a non-searched fix, or the reverse, would
   make the comparison meaningless.
3. The project's own fix must not be tuned on the test set at all. `AGENTS.md` rule and the brief both
   forbid it. Fix the view count and the canonicaliser a priori and say so.
4. A plausible and interesting hypothesis to test: the searched view weights are themselves fitted to the
   canonical alignment of ModelNet40, so they should transfer badly under yaw rotation. If so, part of the
   rotation sensitivity is the *search* overfitting to pose, not the projection alone. The non-searched
   baseline is the control that separates those two explanations.

Verify all of this by reading the file yourself before citing it. Do not take this note's word for it in the
report.

## Finding 2: only one dependency needs compiling, and there is a native replacement

`zeroshot_cls/requirements.txt` pins `torch-scatter==2.0.9`, which is a compiled extension and is the one
real installation risk on Windows with a current PyTorch. There are **no `.cu` files** anywhere in the repo,
so nothing else needs a CUDA toolchain.

It is used in exactly one place, `trainers/mv_utils_zs.py`:

```
line 1:   from torch_scatter import scatter
line 129: grid = scatter(_z, coordinates.long(), dim=1, out=grid, reduce="max")
```

Modern PyTorch has this natively: `Tensor.scatter_reduce_(dim, index, src, reduce="amax",
include_self=True)`. If `torch-scatter` will not install, replace that one call rather than abandoning the
reproduction.

**If you make that substitution, it is a deviation and must be logged**, and it must be justified by a test
that shows the two produce identical output on random inputs (same shapes, same dtypes, including the
`out=` initial values, which `include_self` controls). Write that test in `tests/` before trusting it.

## Finding 3: other pins

`flake8==3.7.9`, `yapf==0.29.0`, `isort==4.3.21` are development tools only and are not needed to reproduce
anything - do not install them just because they are listed. `tb-nightly` is a TensorBoard nightly and is
also unnecessary for an inference-only run. `h5py`, `scipy`, `scikit-learn`, `ftfy`, `regex`, `tqdm`, `yacs`
and `gdown` are needed.

`Dassl3D/setup.py` is a plain Python package install (`pip install -e .` inside that folder), not a CUDA
build.

## Finding 4: the bundled CLIP

`zeroshot_cls/clip/` is a vendored copy of OpenAI's CLIP, so `open_clip_torch` in our `requirements.txt` is
not required by the upstream path. Harmless, but note that **it pulled a CPU-only `torch` from PyPI** when
our requirements were first installed on Windows, which silently produced a CPU environment. Install torch
from the cu128 index and check `torch.cuda.is_available()` after any `pip install` that touches torch.
