"""Replicate the upstream evaluation-time prompt search, to measure what it is worth.

**This script deliberately tunes on the test labels.** That is the point: it
reproduces what `trainers/post_search.py` does, so the gap between the shipped
starting point and the searched endpoint can be stated as a number instead of
described in words. Nothing it produces is a result of *our* method, and no
number from it may be presented as an accuracy our fix achieves.

What upstream does (`post_search.py`, lines 100-122): for each of 2 passes, for
each class, for each of ~1,198 candidate prompts from `prompts/<dataset>_1000.json`,
swap that class's prompt in, recompute accuracy against the test labels, and keep
the swap if accuracy improved. It starts from the prompts already shipped in
`best_param.py`, which were themselves produced this way.

Implementation note: swapping class i's prompt changes only column i of the
logit matrix, so all ~1,198 candidate columns for a class are computed in one
matmul and scored together. Greedy acceptance within a class then reduces to
taking the best candidate, and classes are still processed in order so later
classes see earlier accepted prompts - the same semantics as the sequential
loop, at a fraction of the cost.

Usage:
    python scripts/replicate_search.py ViT-B/16
    python scripts/replicate_search.py RN101
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_fix import git_commit  # noqa: E402
from src import pcv2  # noqa: E402
from src.data import load_modelnet40  # noqa: E402
from src.metrics import accuracy_with_ci  # noqa: E402

DATA_ROOT = "data/modelnet40_ply_hdf5_2048"
NAME2 = {"ViT-B/16": "vit_b16", "ViT-B/32": "vit_b32", "RN50": "rn50", "RN101": "rn101"}
PASSES = 2


@torch.no_grad()
def image_features(pipeline, points: np.ndarray, batch_size: int = 16) -> torch.Tensor:
    """Weighted, concatenated per-object image features: [N, num_views * channel]."""
    out = []
    for start in range(0, len(points), batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(points[start : start + batch_size])).float().cuda()
        images = pcv2.render(pipeline, batch).type(pipeline.dtype)
        feat = pipeline.clip_model.visual(images)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        weighted = feat.reshape(-1, pipeline.num_views, pipeline.channel) * pipeline.view_weights.reshape(1, -1, 1)
        out.append(weighted.reshape(-1, pipeline.num_views * pipeline.channel).float())
    return torch.cat(out)


@torch.no_grad()
def encode(pipeline, texts: list[str], upstream_clip, batch: int = 256) -> torch.Tensor:
    """Encode prompts and lay them out exactly as the pipeline expects: repeated, then normalised."""
    feats = []
    for i in range(0, len(texts), batch):
        tokens = torch.cat([upstream_clip.tokenize(t, truncate=True) for t in texts[i : i + batch]]).cuda()
        f = pipeline.clip_model.encode_text(tokens).float().repeat(1, pipeline.num_views)
        feats.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(feats)


def accuracy_of(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def main(backbone: str) -> int:
    name2 = NAME2[backbone]
    points, labels_np, classnames = load_modelnet40(DATA_ROOT, "test")
    points = points[:, :1024]
    labels = torch.from_numpy(labels_np).cuda()

    pipeline = pcv2.build_pipeline("modelnet40", classnames, backbone=backbone, protocol=pcv2.PROTOCOL_PUBLISHED)
    sys.path.insert(0, pcv2.upstream_dir())
    from clip import clip as upstream_clip  # type: ignore

    print(f"backbone {backbone} | {len(points)} objects | {pipeline.num_views} views x {pipeline.channel} ch")

    t0 = time.perf_counter()
    img = image_features(pipeline, points)
    print(f"image features in {time.perf_counter() - t0:.0f}s: {tuple(img.shape)}")

    text = pipeline.text_feat.float().clone()  # [40, V*C], the shipped starting point
    logits = 100.0 * img @ text.t()
    start_acc = accuracy_of(logits, labels)
    print(f"shipped starting point: {100 * start_acc:.2f}%")

    lib_path = os.path.join(pcv2.upstream_dir(), "prompts", "modelnet40_1000.json")
    with open(lib_path) as f:
        library = json.load(f)
    print(f"candidate library: {sum(len(v) for v in library.values())} prompts over {len(library)} classes")

    history = []
    accepted = 0
    for p in range(PASSES):
        for ci, cname in enumerate(classnames):
            candidates = library.get(cname) or library.get(cname.replace("_", " "))
            if not candidates:
                print(f"  no candidates for {cname!r} - skipped")
                continue
            cand = encode(pipeline, candidates, upstream_clip)  # [M, V*C]
            cols = 100.0 * img @ cand.t()  # [N, M] - every candidate's column at once
            best_acc, best_j = accuracy_of(logits, labels), -1
            kept = logits[:, ci].clone()
            for j in range(cols.shape[1]):
                logits[:, ci] = cols[:, j]
                a = accuracy_of(logits, labels)
                if a > best_acc:
                    best_acc, best_j = a, j
            if best_j >= 0:
                logits[:, ci] = cols[:, best_j]
                text[ci] = cand[best_j]
                accepted += 1
            else:
                logits[:, ci] = kept
            del cand, cols
            torch.cuda.empty_cache()
        acc_now = accuracy_of(logits, labels)
        history.append(acc_now)
        print(f"after pass {p + 1}: {100 * acc_now:.2f}%  ({accepted} prompts replaced so far)")

    final_acc = accuracy_of(logits, labels)
    preds = logits.argmax(dim=1).cpu().numpy()
    ci_final = accuracy_with_ci(preds, labels_np, seed=0)

    print(f"\n{backbone}: shipped {100 * start_acc:.2f}%  ->  searched {100 * final_acc:.2f}%  "
          f"(+{100 * (final_acc - start_acc):.2f} points, {accepted} prompts replaced)")
    print(f"searched, with interval: {ci_final}")

    result = {
        "backbone": backbone,
        "commit": git_commit(),
        "n_objects": int(len(points)),
        "passes": PASSES,
        "shipped_accuracy": start_acc,
        "searched_accuracy": final_acc,
        "gain_points": 100 * (final_acc - start_acc),
        "prompts_replaced": accepted,
        "pass_history": history,
        "searched_accuracy_ci": ci_final.as_dict(),
        "WARNING": "this run tunes on the test labels on purpose, to measure the upstream search; "
                   "it is not a result of our method",
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/search_replication_{name2}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote results/search_replication_{name2}.json")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in NAME2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
