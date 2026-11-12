"""The PointCLIP V2 zero-shot pipeline, driven from our own evaluation loop.

Why not run the upstream `main.py`? Because every experiment in this project
needs three things that its entry point does not give us: a transform applied to
the cloud *before* projection, per-item predictions saved for paired statistics,
and control over the protocol (which prompts and which view weights). So we
import the upstream's own projection and its bundled CLIP and drive them
ourselves. The brief allows "an existing, attributed implementation that you
understand"; the attribution is the pinned commit in docs/REPRO_LOG.md.

Two deviations from upstream, both deliberate and both logged:

1. `torch_scatter` is replaced by a native `scatter_reduce_` shim (see
   `install_torch_scatter_shim`). Equivalence tests:
   `tests/test_scatter_equivalence.py`.
2. The protocol is explicit. `PROTOCOL_PUBLISHED` uses the prompts and view
   weights that ship in `trainers/best_param.py`; `PROTOCOL_PLAIN` uses a fixed
   template prompt and uniform view weights. See the note below.

**The protocol note that matters.** Upstream `trainers/zeroshot.py` loads
`best_prompt_weight['<dataset>_<backbone>_test_prompts']` and
`..._test_weights` by default - the keys say `test` because those prompts and
weights were selected by maximising accuracy on the test labels
(`trainers/post_search.py`). So the published figure is not an untuned zero-shot
result, with or without the `--post-search` flag. This project therefore reports
both protocols and never mixes them within a comparison.
"""

from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass

import numpy as np
import torch

PROTOCOL_PUBLISHED = "published"  # searched prompts + searched view weights (as shipped)
PROTOCOL_PLAIN = "plain"  # fixed template prompt + uniform view weights

# Used for PROTOCOL_PLAIN. Fixed a priori, before seeing any result, and never
# tuned: the renders are depth maps, so this is the honest neutral description.
PLAIN_TEMPLATE = "a depth map of a {}."


def install_torch_scatter_shim() -> bool:
    """Make `from torch_scatter import scatter` work without torch-scatter.

    Returns True if the shim was installed, False if the real package is present
    (in which case upstream's own code path is used unchanged).
    """
    try:
        import torch_scatter  # noqa: F401

        return False
    except ImportError:
        pass

    def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
        if reduce != "max":
            raise NotImplementedError(
                f"the shim only implements reduce='max' (upstream's only use); got {reduce!r}"
            )
        if out is None:
            raise NotImplementedError("the shim requires out=, as upstream passes it")
        return out.scatter_reduce_(dim, index, src, reduce="amax", include_self=True)

    module = types.ModuleType("torch_scatter")
    module.scatter = scatter  # type: ignore[attr-defined]
    module.__SHIM__ = True  # type: ignore[attr-defined]
    sys.modules["torch_scatter"] = module
    return True


def upstream_dir(project_root: str | None = None) -> str:
    root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "third_party", "PointCLIP_V2", "zeroshot_cls")


@dataclass
class Pipeline:
    """Everything needed to classify a batch of clouds."""

    clip_model: object
    projection: object
    text_feat: torch.Tensor  # [num_classes, num_views * channel], normalised per class
    view_weights: torch.Tensor  # [num_views]
    num_views: int
    channel: int
    dtype: torch.dtype
    protocol: str
    prompts: list[str]
    shim_used: bool

    @property
    def forward_passes_per_object(self) -> int:
        """CLIP image-encoder passes per object - the compute axis for control C5."""
        return self.num_views


def build_pipeline(
    dataset: str,
    classnames: list[str],
    backbone: str = "ViT-B/16",
    protocol: str = PROTOCOL_PUBLISHED,
    project_root: str | None = None,
) -> Pipeline:
    """Load CLIP and the upstream projection, and build the text features."""
    shim_used = install_torch_scatter_shim()

    up = upstream_dir(project_root)
    if not os.path.isdir(up):
        raise FileNotFoundError(f"upstream not found at {up}; see SETUP.md")
    if up not in sys.path:
        sys.path.insert(0, up)

    from clip import clip as upstream_clip  # type: ignore
    from trainers.best_param import best_prompt_weight  # type: ignore
    from trainers.mv_utils_zs import Realistic_Projection  # type: ignore

    model_path = upstream_clip._download(upstream_clip._MODELS[backbone])
    try:
        jit_model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        jit_model, state_dict = None, torch.load(model_path, map_location="cpu")
    clip_model = upstream_clip.build_model(state_dict or jit_model.state_dict()).cuda().eval()

    name2 = {"ViT-B/16": "vit_b16", "ViT-B/32": "vit_b32", "RN50": "rn50", "RN101": "rn101"}[backbone]
    channel = {"ViT-B/16": 512, "ViT-B/32": 512, "RN50": 1024, "RN101": 512}[backbone]

    projection = Realistic_Projection()
    num_views = projection.num_views

    if protocol == PROTOCOL_PUBLISHED:
        prompts = list(best_prompt_weight[f"{dataset}_{name2}_test_prompts"])
        weights = torch.tensor(best_prompt_weight[f"{dataset}_{name2}_test_weights"]).float().cuda()
    elif protocol == PROTOCOL_PLAIN:
        prompts = [PLAIN_TEMPLATE.format(name.replace("_", " ")) for name in classnames]
        weights = torch.ones(num_views).float().cuda()
    else:
        raise ValueError(f"unknown protocol {protocol!r}")

    if len(prompts) != len(classnames):
        raise ValueError(f"{len(prompts)} prompts for {len(classnames)} classes")

    with torch.no_grad():
        tokens = torch.cat([upstream_clip.tokenize(p) for p in prompts]).cuda()
        text_feat = clip_model.encode_text(tokens).repeat(1, num_views)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    return Pipeline(
        clip_model=clip_model,
        projection=projection,
        text_feat=text_feat,
        view_weights=weights,
        num_views=num_views,
        channel=channel,
        dtype=clip_model.dtype,
        protocol=protocol,
        prompts=prompts,
        shim_used=shim_used,
    )


def render(pipeline: Pipeline, points: torch.Tensor, imsize: int = 224) -> torch.Tensor:
    """Depth-map renders for a batch of clouds: [B * num_views, 3, imsize, imsize]."""
    img = pipeline.projection.get_img(points).cuda()
    return torch.nn.functional.interpolate(img, size=(imsize, imsize), mode="bilinear", align_corners=True)


@torch.no_grad()
def classify_batch(pipeline: Pipeline, points: torch.Tensor) -> torch.Tensor:
    """Logits for a batch of clouds: [B, num_classes]. Mirrors upstream exactly."""
    images = render(pipeline, points).type(pipeline.dtype)
    feat = pipeline.clip_model.visual(images)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    weighted = feat.reshape(-1, pipeline.num_views, pipeline.channel) * pipeline.view_weights.reshape(1, -1, 1)
    weighted = weighted.reshape(-1, pipeline.num_views * pipeline.channel).type(pipeline.dtype)
    return 100.0 * weighted @ pipeline.text_feat.t()


@torch.no_grad()
def predict(
    pipeline: Pipeline,
    points: np.ndarray,
    batch_size: int = 16,
    transform=None,
    progress: bool = True,
) -> np.ndarray:
    """Predicted class per object.

    `transform(cloud_np) -> cloud_np` is applied per object **before**
    projection, which is where a robot's pose variation enters. Applying a
    rotation to the rendered image instead would be a different experiment.
    """
    preds = np.empty(len(points), dtype=np.int64)
    iterator = range(0, len(points), batch_size)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="eval", unit="batch")
        except ImportError:
            pass
    for start in iterator:
        chunk = points[start : start + batch_size]
        if transform is not None:
            chunk = np.stack([transform(cloud) for cloud in chunk]).astype(np.float32)
        batch = torch.from_numpy(np.ascontiguousarray(chunk)).float().cuda()
        preds[start : start + len(chunk)] = classify_batch(pipeline, batch).argmax(dim=-1).cpu().numpy()
    return preds
