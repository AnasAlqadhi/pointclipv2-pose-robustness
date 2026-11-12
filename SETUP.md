# Setup

Machine this is written for: **dev** - Windows 11, RTX 5070 Laptop (8 GB VRAM), Python 3.12, driver 621.14.
The ZBook is used only for ROS/Gazebo data collection (Track C) and never runs CLIP.

## 1. Python environment

The system Python has **CPU-only torch 2.14.1**, which is no use here. The RTX 5070 is Blackwell
(compute capability 12.0), so it needs a CUDA 12.8 build of PyTorch. Make a dedicated virtual environment:

```powershell
cd D:\Projects\school\courses\2026-fall-YZM5110-multimodal-learning\project
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Check it before going further - if this prints `False` or an `sm_120` warning, nothing else will work:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 1a. If `import torch` fails with WinError 4551

On this laptop it does. **Smart App Control is on** (`VerifiedAndReputablePolicyState = 1`), and it refuses
to load `torch\lib\shm.dll` from the `download.pytorch.org` wheels because they do not meet its signing
requirements. The CPU wheels from PyPI load; the CUDA ones do not. It is a signature check, so moving the
virtual environment to another drive or folder changes nothing.

Three ways out, in the order they are worth trying. Anas decides - one of them is irreversible.

**A. WSL2 with Ubuntu (recommended).** Smart App Control does not apply inside WSL, and the GPU is passed
through by the Windows driver, which is new enough (621.14). WSL2 is already enabled on this machine; no
distribution is installed yet. Anas runs, in an **administrator** terminal:

```powershell
wsl --install -d Ubuntu-24.04
```

then inside Ubuntu:

```bash
python3 -m venv ~/pcv2 && source ~/pcv2/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The repository is reachable from Ubuntu at `/mnt/d/Projects/school/...`, so the code does not move. Note that
file access across `/mnt/d` is slow; keep the datasets inside the Linux filesystem (`~/data`) and only the
code on `/mnt/d`.

**B. Turn Smart App Control off.** This is Anas's decision alone and **it cannot be undone** - re-enabling
Smart App Control requires reinstalling Windows. It also lowers protection for everything else on the
machine, not just this project. Not recommended for a course project.

**C. Run the GPU work on the ZBook.** Ubuntu, so no Application Control. But the Quadro M2000M is Maxwell
with 4 GB, which current cu128 wheels no longer support - it would need an older torch (cu118) and the
inference would be slow. Keep this as the fallback if WSL cannot be installed.

Whichever is chosen, it is a deviation from a plain Windows setup: record it in `docs/REPRO_LOG.md`.

Then the rest:

```powershell
pip install -r requirements.txt
```

Record the exact versions that end up installed:

```powershell
pip freeze > docs/pip-freeze-dev.txt
```

## 2. Upstream code

```powershell
git clone https://github.com/yangyangyang127/PointCLIP_V2 third_party/PointCLIP_V2
cd third_party/PointCLIP_V2
git rev-parse HEAD    # paste this commit hash into docs/REPRO_LOG.md
```

The zero-shot classification code is in its `zeroshot_cls/` folder. `third_party/` is gitignored - the repo
records the commit hash, not a copy of someone else's code.

**Expect friction here.** The repo pins older CUDA/PyTorch versions and may need `pointnet2`-style CUDA
extensions. Decision point is Thursday night: if the extensions will not build against CUDA 12.8, fall back
in this order:
1. run the depth-projection step on CPU (it is cheap) and keep CLIP on GPU;
2. PointCLIP V1 (`github.com/ZrrSkywalker/PointCLIP`), same research question;
3. TPT (`github.com/azshue/TPT`) - the backup topic in `../PROJECT.md`.

Whichever happens, write it down in `docs/REPRO_LOG.md`. Deviations are required to be reported, not hidden.

## 3. Data

Nothing goes in git. Put all of it under `data/`.

**ModelNet40** (Track A) - PointCLIP uses the 2,468-object test split of the 1024-point sampled version
(`modelnet40_normal_resampled` or the `modelnet40_ply_hdf5_2048` variant). Follow the upstream repo's
`DATASET.md`, since the preprocessing must match or the baseline will not reproduce.

**ScanObjectNN** (Track B) - splits `OBJ_ONLY` and `PB_T50_RS`. Requires an access request on the official
site. **Do this on Thursday**, because approval is not instant.

**Robot clouds** (Track C) - produced on the ZBook, copied to `data/robot/`. See
`docs/ANTIGRAVITY_TASKS.md` task R1.

Record for each dataset: the exact file you downloaded, its size, its date, and the preprocessing used.

## 4. Sanity check before any experiment

```powershell
python -m pytest tests -q
```

The tests cover the yaw maths only (rotation is a proper rotation, canonicalisation is idempotent, a yaw
applied then undone is the identity). They do not need CLIP, a GPU or any dataset, so they should pass
immediately after step 1.
