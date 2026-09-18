"""Eq. (9) from Kheradmand et al. 2024 — pure-PyTorch port of LFS MCMC relocate.

This module is the *algorithm core* of the MCMC densification step. When
a source Gaussian is sampled K_i times in one densification cycle (a
"ratio" of 1 + K_i with LFS's +1 baseline), the source's opacity and
scale must be mutated so the union of K_i+1 sub-clones preserves the
original opacity mass and footprint. That mutation is Eq. (9):

    new_opacity_post = 1 - (1 - opacity)^(1/n)
    denom_sum[i]     = sum_{i=1..n} sum_{k=0..i-1} coeffs[i-1, k] * new_opacity^(k+1)
    coeff            = opacity_post / denom_sum         (clamp +/- 1e6)
    new_scale_post   = max(|coeff * old_scale_post|, 1e-10)

The binomial-sign coefficient table is precomputed once via
``precompute_coefficients(n_max)``. The default ``n_max=51`` matches LFS
``mcmc.hpp:_n_max = 51`` and ``mcmc_kernels.cu:RELOCATION_N_MAX``.

This module is **algorithm only**, intentionally *decoupled* from
:class:`scene.strategy.mcmc.McmcStrategy`. It does not touch the model's
raw-logit vs post-sigmoid storage: callers pass post-sigmoid opacity and
post-exp scale and receive the same; the strategy layer (next commit) is
responsible for log/logit conversion at the model boundary. This split
lets us unit-test the math against LFS hand-computed fixtures without
spinning up a GaussianModel.

Reference LFS files (read-only):
  - ``src/training/kernels/mcmc_kernels.cu:37-55``   init_relocation_coefficients
  - ``src/training/kernels/mcmc_kernels.cu:60-105``  relocation_kernel
  - ``src/training/strategies/mcmc.cpp:265-272``    histogram (clone segment)

Reference paper:
  Kheradmand, P. et al., 2024. *"3D Gaussian Splatting as Markov Chain
  Monte Carlo"*. Eq. (9).
"""
from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Constants — mirror LFS mcmk_kernels.cu:50-54
# ---------------------------------------------------------------------------

RELOCATION_N_MAX = 51   # LFS _n_max (mcmc.hpp:87)
OPACITY_MIN      = 1e-6  # numerical floor on post-sigmoid opacity
OPACITY_MAX      = 1.0 - 1e-6  # numerical ceil on post-sigmoid opacity
DENOM_MIN        = 1e-8  # safe-division floor on denom_sum
COEFF_MAX        = 1e6   # clamp magnitude on the coeff (op/denom) ratio
SCALE_MIN        = 1e-10  # floor on post-exp scale (log clamp downstream)


# ---------------------------------------------------------------------------
# Binomial-sign coefficient table
# ---------------------------------------------------------------------------

def precompute_coefficients(n_max: int = RELOCATION_N_MAX) -> torch.Tensor:
    """Pre-compute the binomial-sign coefficient table of shape [n_max, n_max].

    coeffs[n, k] = (-1)^k * C(n, k) / sqrt(k + 1)

    Reference LFS ``mcmc_kernels.cu:init_relocation_coefficients`` (lines 37-55).
    Compute once at module import (or at strategy init); reuse for every
    iteration.

    Returns:
        ``[n_max, n_max]`` float32 tensor on CPU. Move to device at first use;
        the strategy caches it on the right CUDA device to avoid repeated
        host-device transfers.
    """
    coeffs = torch.zeros(n_max, n_max, dtype=torch.float32)
    for n in range(n_max):
        binom = 1.0
        for k in range(n + 1):
            sign = 1.0 if (k % 2 == 0) else -1.0
            coeffs[n, k] = binom * sign / math.sqrt(k + 1)
            # C(n, k+1) = C(n, k) * (n - k) / (k + 1)
            if k < n:
                binom *= (n - k) / (k + 1)
    return coeffs


# ---------------------------------------------------------------------------
# Histogram helper (LFS +1 baseline, clamped [1, n_max])
# ---------------------------------------------------------------------------

def histogram_counts(
    sampled_idxs: torch.Tensor,         # [m] int64; multinomial output
    n_total:      int,                  # total slot count (including dead)
    device:       torch.device,         # device to allocate the ratios buffer
    n_max:        int = RELOCATION_N_MAX,
) -> torch.Tensor:
    """Per-source occurrence count with LFS +1 baseline, clamped [1, n_max].

    Mirrors LFS ``mcmc.cpp:265-272``:

        ratios = ones_int32.slice(0, 0, N).clone();            # +1 baseline!
        ratios.index_add_(0, sampled_idxs, ones_int32.slice(0, 0, sampled_idxs.numel()));
        ratios = ratios.index_select(0, sampled_idxs).contiguous();
        ratios = ratios.clamp(1, n_max);

    CRITICAL: LFS initializes ``ratios`` to all ones *before* the scatter,
    giving ``n_idx[i] = 1 + K_i`` where K_i is the number of times row i
    was sampled. Dropping the +1 baseline yields n=1 for any
    once-sampled row and Eq.(9) becomes a no-op. The 2026-07-29 spec
    audit flagged this as the single most-likely cause of the MCMC
    parity regression.

    Args:
        sampled_idxs: ``[m]`` int64 output of ``torch.multinomial`` over
            ``live_indices`` — i.e., row indices in [0, n_total).
        n_total:      total GaussianModel slot count (including dead).
        device:       device to allocate the ratios buffer on.
        n_max:        clamp upper bound; default 51 (LFS _n_max).

    Returns:
        ``[m]`` int64 tensor, per-sample ratio, in [1, n_max].
    """
    if sampled_idxs.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    s64 = sampled_idxs.to(torch.int64)
    ratios = torch.ones(n_total, dtype=torch.int64, device=device)
    ratios.scatter_add_(0, s64, torch.ones_like(s64))
    return ratios.index_select(0, s64).clamp(1, n_max)


# ---------------------------------------------------------------------------
# Eq. (9) — opacity + scale mutation per source
# ---------------------------------------------------------------------------

def relocate_opacity_scale(
    src_opacities_post: torch.Tensor,    # [n] post-sigmoid opacity in (0, 1)
    src_scales_post:    torch.Tensor,    # [n, 3] post-exp scale (positive)
    src_ratios:         torch.Tensor,    # [n] int32/64; per-sample K_i count +1
    coeffs:             torch.Tensor,    # [n_max, n_max] float32; precomputed
    min_opacity_post:   float = 0.005,   # LFS OptimizationParameters.min_opacity
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Eq. (9) per source and return new (opacity, scale) post-activation.

    Caller's responsibility: convert between post-sigmoid/post-exp and
    raw-logit/log spaces when integrating with GaussianModel (which
    stores raw); see ``scene/strategy/mcmc.py`` for the integration
    pattern. This module never touches raw storage on its own.

    Mirrors LFS ``mcmc_kernels.cu:relocation_kernel:60-105``.

    Args:
        src_opacities_post: ``[n]`` float32, post-sigmoid opacity in (0, 1).
            Values outside (0, 1) are clamped to (OPACITY_MIN, OPACITY_MAX).
        src_scales_post:    ``[n, 3]`` float32, post-exp scale (positive);
            clamped internally to >= SCALE_MIN.
        src_ratios:         ``[n]`` int32 or int64. Will be clamped to
            ``[1, n_max]`` before use.
        coeffs:             ``[n_max, n_max]`` float32 output of
            ``precompute_coefficients``. Moved to ``src_opacities_post.device``
            if not already there.
        min_opacity_post:   post-sigmoid opacity floor. Default 0.005 matches
            LFS ``OptimizationParameters.min_opacity``. The output opacity
            is clamped at ``max(OPACITY_MIN, min_opacity_post)`` from below.

    Returns:
        Tuple ``(new_opacities_post: [n], new_scales_post: [n, 3])``.
        Caller may further clamp new_opacities_post to its valid range
        and convert to raw-logit before writing back to the GaussianModel.
    """
    assert src_opacities_post.dim() == 1, (
        f"src_opacities_post must be 1-D, got {src_opacities_post.shape}"
    )
    assert src_scales_post.dim() == 2 and src_scales_post.shape[1] == 3, (
        f"src_scales_post must be [n, 3], got {src_scales_post.shape}"
    )
    n = src_opacities_post.shape[0]
    assert src_ratios.shape == (n,), (
        f"src_ratios length mismatch: {src_ratios.shape[0]} vs n={n}"
    )

    device = src_opacities_post.device
    coeffs_dev = coeffs.to(device) if coeffs.device != device else coeffs

    # --- Clamp inputs to LFS numerical safety ranges ---
    opacity_post = src_opacities_post.clamp(OPACITY_MIN, OPACITY_MAX)
    ratios_clamped = src_ratios.clamp(1, RELOCATION_N_MAX).to(opacity_post.dtype)

    # --- new_opacity_post = 1 - (1 - op)^(1/n) ---
    inv_n = 1.0 / ratios_clamped
    one_minus_op = 1.0 - opacity_post
    new_opacity_post = (1.0 - torch.pow(one_minus_op, inv_n)).clamp(
        min=max(OPACITY_MIN, float(min_opacity_post)),
        max=OPACITY_MAX,
    )

    # --- denom_sum per row ---
    n_int64 = ratios_clamped.to(torch.int64)
    denom_sum = _denom_sum_vectorized(new_opacity_post, n_int64, coeffs_dev)

    # --- Safe division preserving sign (avoids 0/0 → NaN) ---
    abs_denom = denom_sum.abs().clamp(min=DENOM_MIN)
    sign_denom = torch.where(
        denom_sum < 0,
        -torch.ones_like(denom_sum),
        torch.ones_like(denom_sum),
    )
    safe_denom = abs_denom * sign_denom

    # --- coeff = opacity_post / denom_sum, clamped to +/- COEFF_MAX ---
    coeff = (opacity_post / safe_denom).clamp(-COEFF_MAX, COEFF_MAX)

    # --- new_scale_post[i] = max(|coeff * old_scale[i]|, SCALE_MIN) ---
    new_scale_post = (coeff.abs().unsqueeze(-1) * src_scales_post).clamp(min=SCALE_MIN)

    return new_opacity_post, new_scale_post


# ---------------------------------------------------------------------------
# Inner: denom_sum vectorized over rows + sample-ratios
# ---------------------------------------------------------------------------

def _denom_sum_vectorized(
    new_opacity: torch.Tensor,    # [n] post-sigmoid new opacity (clamped)
    ratios:      torch.Tensor,    # [n] int64 in [1, RELOCATION_N_MAX]
    coeffs:      torch.Tensor,    # [n_max, n_max] float32
) -> torch.Tensor:
    """Compute ``denom_sum[r] = sum_{i=1..ratios[r]} sum_{k=0..i-1}
    coeffs[i-1, k] * new_opacity[r]^(k+1)``.

    Vectorized via geometric-power outer product + 51-step accumulation:

        powers[r, p] = new_opacity[r]^p for p = 1..n_max
      Then for each i in 1..n_max:
        inner_kernels = powers[r, 1:i] @ coeffs[i-1, 0:i]   (sum over k+1)
        mask          = (ratios[r] >= i).to(dtype)
        out[r]       += inner_kernels[r] * mask[r]

    Cost: 51 inner products of length 1..51 over n elements = O(n * 51).
    At n=1M this is ~50 M elementwise ops in PyTorch; per-call wall-clock
    is single-digit ms on RTX A4500 (P1-MCMCParity §10.3 budget). If a
    CUDA relocation_kernel is needed for prod scale, see LFS
    ``mcmc_kernels.cu:relocation_kernel:60-105`` for the kernel structure.

    Args:
        new_opacity: ``[n]`` float32 post-sigmoid new opacity (already
            clamped to ``[OPACITY_MIN, OPACITY_MAX]``).
        ratios:      ``[n]`` int64 per-row ratio (already clamped to
            ``[1, RELOCATION_N_MAX]``).
        coeffs:      ``[n_max, n_max]`` float32; the precomputed
            binomial-sign table on the same device as ``new_opacity``.

    Returns:
        ``[n]`` float32, ``denom_sum`` per row.
    """
    n = new_opacity.shape[0]
    n_max = coeffs.shape[0]
    device = new_opacity.device

    # Geometric powers: powers[r, p] = new_opacity[r]^p   for p = 1..n_max.
    # log/new_op path avoids underflow on long products; log clamp at 1e-30.
    log_op = torch.log(new_opacity.clamp(min=1e-30))
    p_grid = torch.arange(1, n_max + 1, device=device, dtype=new_opacity.dtype)
    powers = torch.exp(p_grid.unsqueeze(0) * log_op.unsqueeze(-1))   # [n, n_max]

    out = torch.zeros_like(new_opacity)
    for i in range(1, n_max + 1):
        # For ratio = i, accumulate coeffs[i-1, 0..i-1] @ powers[r, 1..i].
        coef_row = coeffs[i - 1, :i]                                # [i]
        contrib = (powers[:, :i] * coef_row.unsqueeze(0)).sum(dim=-1)   # [n]
        # Apply only to rows where ratios[r] >= i.
        mask = (ratios >= i).to(new_opacity.dtype)
        out = out + contrib * mask

    return out
