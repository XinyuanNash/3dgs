# 3dgs-work — Install procedure

> Tested environment (2026-08-07):
> Python 3.10.20 + PyTorch 2.7.1+cu128 + CUDA 12.8 + GCC 11.

This file mirrors the original Inria gaussian-splatting install recipe,
updated for the post-P0/P1/P2 modifications in this repo:

- New: PyTorch 2.7 (was 1.12) → requires GCC 11+ and modern CUDA 12.x
- New: `opencv-python` is a hard import (depth scaling + camera utils)
- New: `kornia` is opt-in (only loaded when `--igs_plus_use_edge` is set)
- New: `simple_knn` is built from source (was already required)
- Same: `diff_gaussian_rasterization` and `fused_ssim` are built from source

## Quick start (conda + pip)

```bash
# 1. Create the conda env (PyTorch + CUDA toolchain + Python deps)
conda env create -f environment.yml -n 3dgs
conda activate 3dgs

# 2. Pin pip-installed runtime deps (matches the working env exactly)
pip install -r requirements.txt
pip install -r requirements-build.txt

# 3. Build the CUDA submodules (compile against your CUDA + torch)
#    These are CUDA kernels — first build is slow (~3 min total).
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim

# 4. (Optional) Test deps
pip install -r requirements-test.txt

# 5. Verify
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import diff_gaussian_rasterization; print('dgr OK')"
python -c "import simple_knn._C; print('simple_knn OK')"
python -c "import fused_ssim; print('fused_ssim OK')"
```

## pip-only install (no conda)

If you already have a CUDA-12.8-capable driver + GCC 11 + system nvcc:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-build.txt
# torch 2.7.1 + cu128 is pip-installable from pytorch.org
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
# CUDA submodules
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim
```

## Version pinning rationale

| Package | Pinned | Why |
|---|---|---|
| `torch==2.7.1` | exact | Submodule CUDA extensions are ABI-pinned to this version's libtorch |
| `cudatoolkit=12.8` | exact | Must match torch's CUDA runtime (cu128) — PyTorch ships its own CUDA libs in the wheel, but nvcc must match for submodule JIT compile |
| `python=3.10` | minor | 3.8.10 has the `BooleanOptionalAction` ImportError; 3.11+ untested |
| `gcc=11` | major | PyTorch 2.7 wheels compiled with GCC 11.2 — older GCC may fail at import |
| `ninja>=1.10` | minor | PyTorch's default build backend |
| `kornia>=0.7.0` | minor | Required by P1-5 IGS+ phase-2 Canny edge scores |
| `opencv-python>=4.5` | minor | Imported unconditionally by camera_utils / depth scale |

## What is NOT included (and why)

- **scipy, matplotlib, tensorboard, open3d, pyyaml** — not used by the codebase
  (verified by `grep -rh "^import " *.py scene/ utils/ | sort -u`).
  `tensorboard` was dropped in 2026-08 refactor (was wrapped in
  try/except ImportError — never required).
- **networkx>=3.0** — included as a transitive utility; only some
  dataset readers / debug helpers use it.
- **MKL / oneAPI** — bundled inside the PyTorch wheel; no separate install.
- **CUDA driver (kernel-mode)** — host-level (installed via the GPU driver
  package, not pip or conda). Needs to support CUDA 12.8 (driver ≥ 570.x).

## CUDA-version matrix

| torch | CUDA wheel suffix | Min driver | Min nvcc |
|---|---|---|---|
| 2.7.1 | cu128 | 570.x | 12.8 |
| 2.4.x | cu124 | 550.x | 12.4 |
| 2.1.x | cu121 | 530.x | 12.1 |

If you need to downgrade, update `requirements.txt`, `environment.yml`,
and rebuild the 3 CUDA submodules (their `setup.py` bakes in `TORCH_CUDA_ARCH_LIST`).

## Verifying after install

```bash
# Smoke test (from repo root)
python -c "
import torch, diff_gaussian_rasterization, simple_knn._C, fused_ssim
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('dgr, simple_knn, fused_ssim: OK')
"

# Run the full smoke + unit tests (needs GPU)
pytest tests/test_opt_in_modules.py tests/test_train_py_opt_in_chain.py -v
python tests/smoke_opt_in_modules.py

# Run a 800-iter determinism check (campus dataset; ~2 min)
bash train_800_default_baseline.sh
```

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `libcudart.so.12: cannot open` | nvcc/runtime mismatch | Match cudatoolkit to torch CUDA version |
| `torch.utils.cpp_extension.CUDA_HOME not set` | nvcc not in PATH | `export CUDA_HOME=/usr/local/cuda` + `export PATH=$CUDA_HOME/bin:$PATH` |
| `undefined symbol: _ZN3c106detail23torchEmptyStrided...` | torch ABI mismatch after rebuild | Rebuild submodules with matching torch |
| `Python.h: No such file` | python3-dev missing | `apt install python3.10-dev` (Debian/Ubuntu) |
| `fatal error: cuda_runtime.h` | cudatoolkit headers missing | Install `cuda-nvcc` + `cuda-cudart-dev` via conda |

## What's changed vs the original INRIA gaussian-splatting install

- Dropped Python 3.7 → 3.10
- Dropped PyTorch 1.12 → 2.7.1 (cu128)
- Dropped CUDA 11.6 → 12.8
- Added `opencv-python`, `kornia`, `networkx`
- Added `requirements.txt` + `requirements-build.txt` + `requirements-test.txt`
  split (was a single environment.yml)
- Documented GCC 11 + nvcc 12.8 host prerequisites