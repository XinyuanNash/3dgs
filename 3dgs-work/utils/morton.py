"""Morton encoding (Z-order curve) for 3D Gaussian positions.

P3-Morton (doc/specs/P3-Morton-spec.md §2.1-2.2):
    Reorders 3D Gaussian attributes (xyz, scaling, rotation, opacity,
    SH-DC/rest, Adam state) by Morton order of their xyz positions, so
    spatially-close Gaussians sit next to each other in memory.
    Improves rasterizer L1/L2 cache locality on large models (~1M
    Gaussians).

Reference algorithm:
    Part1By2 magic-number interleave per Fabian Giesen's
    "Decoding Morton codes" — spreads 10 low bits of a 21-bit value
    by 2 so they can be OR'd together to produce a 30-bit Morton code
    (10 bits/axis × 3 axes = 30 bits). Mirrors the LFS reference in
    ``LichtFeld-Studio/src/io/cuda/morton_encoding.cu`` (Part1By2 +
    encodeMorton3), re-implemented in pure PyTorch bitwise ops so no
    CUDA kernel is needed. At 1.2M Gaussians on RTX A4500 the bitwise
    pipeline runs < 50 ms (well under densify-step overhead).

Why this lives in ``utils/`` (NOT an OptInTrainModule wrapper):
    Morton is a *data layout* decision — it does not affect loss,
    gradient, image transform, or per-iter scheduling. OptInTrainModule
    is for cross-cutting *train-time* concerns (BilateralGrid, PPISP,
    ADMM, scale_reg). Conflating the two would leak the abstract base
    class with a perf-only optimization.
"""
from __future__ import annotations

import torch


def part1by2(x: torch.Tensor) -> torch.Tensor:
    """Spread the low 10 bits of ``x`` by 2 (Morton helper).

    Classic Part1By2 by Fabian Giesen. Input: int64 tensor with only the
    low 10 bits meaningful. Output: int64 tensor where bit ``i`` of the
    input is now bit ``2*i`` of the output.

    Each step masks the half that has been processed so far so the
    cascade composes correctly when called on > 10-bit inputs (caller
    is expected to clamp to 10 bits upstream).
    """
    x = x & 0x000003FF
    x = (x ^ (x << 16)) & 0xFF0000FF
    x = (x ^ (x << 8))  & 0x0300F00F
    x = (x ^ (x << 4))  & 0x030C30C3
    x = (x ^ (x << 2))  & 0x09249249
    return x


def morton_encode_3d(xyz: torch.Tensor) -> torch.Tensor:
    """Compute Morton codes for ``[N, 3]`` positions → ``[N]`` int64 codes.

    Algorithm (per LFS morton_encoding.cu:21-33):
        1. Per-axis bbox-normalize → [0, 1] → scale × 1023 → int64.
           Clamp ``[0, 1023]`` so the 10-bit Part1By2 doesn't overflow.
        2. Part1By2 each axis (spreads 10 bits into 20-bit even slots).
        3. Interleave: ``(iz << 2) | (iy << 1) | ix``.
        4. Returns int64 codes in ``[0, 2^30)``.

    Returns codes on the SAME device as ``xyz`` (typically cuda). CPU
    tensors work too (used by unit tests).
    """
    assert xyz.dim() == 2 and xyz.shape[1] == 3, (
        f"morton_encode_3d expects [N, 3], got {tuple(xyz.shape)}"
    )
    mn = xyz.min(dim=0).values                # [3]
    mx = xyz.max(dim=0).values                # [3]
    rng = (mx - mn).clamp(min=1e-9)           # avoid div-by-zero on flat axes
    norm = ((xyz - mn) / rng * 1023.0).clamp(0, 1023).to(torch.int64)
    ix = part1by2(norm[:, 0])
    iy = part1by2(norm[:, 1])
    iz = part1by2(norm[:, 2])
    return (iz << 2) | (iy << 1) | ix         # [N] int64


def morton_sort_indices(morton_codes: torch.Tensor) -> torch.Tensor:
    """Permutation indices that would sort ``morton_codes`` ascending.

    Uses ``torch.argsort(stable=True)`` so Gaussians with identical
    Morton codes keep their original relative order — important for
    determinism (same input → same permutation) and for IGS+ densify
    not accidentally reordering recently-cloned twins.

    Returns int64 tensor of shape ``[N]``.
    """
    return torch.argsort(morton_codes, stable=True)