#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    error_state_handle=0,
    gt_image=None,
    compute_error=False,
):
    """P1-7+ (OPTIMIZATIONS.md §13.6): extended signature.

    New args (all optional, backward-compatible):
      error_state_handle (int): handle returned by make_rasterizer_state();
          0 (default) means no error buffer is computed (legacy path).
      gt_image (Tensor or None): [3, H, W] float32 CUDA in [0, 1]. Required
          when compute_error=True; ignored otherwise.
      compute_error (bool): when True, post-blend kernel populates the
          error_buffer managed by the state behind `error_state_handle`.

    Returns: (color, radii, invdepths, error_buffer). error_buffer is an
    empty Tensor when compute_error=False (legacy callers can unpack with
    `_, _, _, _ = rasterize_gaussians(...)` and ignore the 4th slot).
    """
    if gt_image is None:
        gt_image_arg = torch.Tensor()
    else:
        gt_image_arg = gt_image
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        int(error_state_handle),
        gt_image_arg,
        bool(compute_error),
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        error_state_handle=0,
        gt_image=torch.Tensor(),
        compute_error=False,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug,
            int(error_state_handle),
            gt_image,
            bool(compute_error),
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths, error_buffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths, error_buffer

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth, grad_error_buffer):
        # grad_error_buffer is None when compute_error was False (the
        # backward kernel does not need to know about the buffer).
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D,
                radii,
                colors_precomp,
                opacities,
                scales,
                rotations,
                raster_settings.scale_modifier,
                cov3Ds_precomp,
                raster_settings.viewmatrix,
                raster_settings.projmatrix,
                raster_settings.tanfovx,
                raster_settings.tanfovy,
                grad_out_color,
                grad_out_depth,
                sh,
                raster_settings.sh_degree,
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,  # raster_settings
            None,  # error_state_handle (no grad)
            None,  # gt_image (no grad)
            None,  # compute_error (no grad)
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings, error_state_handle=0):
        super().__init__()
        self.raster_settings = raster_settings
        # P1-7+: optional handle to a C++ RasterizerState. 0 = legacy
        # path (no error buffer). The handle's lifetime is managed by the
        # caller (Python side stores it via make_rasterizer_state and
        # free_rasterizer_state).
        self.error_state_handle = int(error_state_handle)

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)

        return visible

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None,
                compute_error=False, gt_image=None):
        """P1-7+ (OPTIMIZATIONS.md §13.6): extended signature.

        New args (all optional):
          compute_error (bool): forward to the rasterizer.
          gt_image (Tensor or None): [3, H, W] float32 CUDA in [0, 1].
              Required iff compute_error=True.

        Returns: (color, radii, depth, error_buffer).
        """

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
            self.error_state_handle,
            gt_image,
            compute_error,
        )

# ─────────────────────────────────────────────────────────────────────────────
# BilateralGrid (P0-Bilateral-v2 — 2026-08-04 restart)
#
# Per-image affine+bias color transform parameterized as a 5D tensor
# [N, 12, L, H, W], where 12 = 3 output channels × 4 input channels
# (R, G, B, 1-bias). Trilinear interpolation in (x_norm, y_norm, luma)
# space. Identity-init means apply(rgb, idx) == rgb at the start of
# training; gradient then learns per-image appearance correction.
#
# Reference: Bao et al., "3D Gaussian Splatting with image-based
# appearance modeling" (LFS gsplat port). CUDA kernels in
# bilateral_grid/bilateral_grid.cu. Adam step and TV regularization are
# pure-PyTorch (the grid is tiny; ~25K params per image).
#
# v2 DESIGN NOTE (vs v1 which regressed -6 dB at 30K on campus):
#   The v1 implementation engaged the grid Adam updates from iteration 0.
#   On campus (LLFF clean, no per-image exposure drift) the grid had
#   nothing useful to learn, but it still drifted during densification
#   (500-25K) — and that drift made the rasterizer fit the bilateral-
#   corrected images instead of ground truth, a -6 dB feedback loop.
#   v2 adds a `start_iter` knob (default = densify_until_iter + 1000 =
#   26000) — until that iter the grid is FROZEN AT IDENTITY, so the
#   failure window is eliminated. After start_iter the grid engages
#   (data5 / in-the-wild datasets benefit; campus stays near-neutral).
# ─────────────────────────────────────────────────────────────────────────────

class _BilateralGridApply(torch.autograd.Function):
    """Wraps bilateral_grid_forward / bilateral_grid_backward for autograd.

    `grid_slice` is the per-image slice [12, L, H, W]. `grad_grid_buf` is
    the per-image slot in the accumulated grid gradient; we zero it before
    the CUDA kernel atomicAdds into it.
    """

    @staticmethod
    def forward(ctx, grid_slice, rgb, grid_W, grid_H, grid_L, grad_grid_buf):
        out = _C.bilateral_grid_forward(grid_slice, rgb)
        ctx.save_for_backward(grid_slice, rgb)
        ctx.grid_W = grid_W
        ctx.grid_H = grid_H
        ctx.grid_L = grid_L
        ctx.grad_grid_buf = grad_grid_buf
        return out

    @staticmethod
    def backward(ctx, grad_out):
        grid_slice, rgb = ctx.saved_tensors
        # CRITICAL: grad_out may arrive non-contiguous due to the
        # upstream .permute(2,0,1).contiguous() backward propagation.
        # Without this, the kernel's atomicAdd into accum_grads gets
        # racing/wrong addresses for most pixels (data-dependent: with
        # certain inputs only 1 pixel's contribution survives).
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()
        grad_rgb = _C.bilateral_grid_backward(
            grid_slice, rgb, grad_out, ctx.grad_grid_buf)
        return None, grad_rgb, None, None, None, None


class BilateralGrid(nn.Module):
    """Per-image appearance-correction grid (Bao et al., 3DGS+IBR).

    Args:
        num_images: total number of camera slots (train + test, by uid).
            Identical to LFS convention: uids stay in 0..N-1 across splits.
        grid_W, grid_H: spatial grid resolution. Default 16×16.
        grid_L: luma-bucket resolution. Default 8.
        lr: Adam learning rate. Default 2e-3 (LFS); we use 5e-4 by default
            per [[p0-bilateral-failed-longrun]] (LR=2e-3 + 800-iter
            schedule regresses ~3 dB on campus).
        start_iter: P0-Bilateral-v2 — iter at which the grid Adam begins
            stepping. Until this iter `engaged(iteration)` returns False
            and `optimizer_step()` is a no-op (frozen at identity).
            Default = 26000 (= densify_until_iter 25K + 1000 buffer).
        total_iterations: total training iterations (for LR schedule).
        warmup_steps: LR linear warmup length. Default 1000.
        warmup_start_factor: initial LR multiplier (1% of lr). Default 0.01.
        final_lr_factor: exponential decay final LR multiplier. Default 0.01.
        tv_weight: TV regularization weight. Default 10.0.
    """

    def __init__(self, num_images,
                 grid_W=16, grid_H=16, grid_L=8,
                 lr=5e-4, beta1=0.9, beta2=0.999, eps=1e-15,
                 warmup_steps=1000,
                 total_iterations=30000,
                 warmup_start_factor=0.01,
                 final_lr_factor=0.01,
                 tv_weight=10.0,
                 start_iter=26000):
        super().__init__()
        self.num_images = int(num_images)
        self.grid_W = int(grid_W)
        self.grid_H = int(grid_H)
        self.grid_L = int(grid_L)
        self.start_iter = int(start_iter)
        # Parameter [N, 12, L, H, W], identity-init channels [0,5,10]=1
        g = torch.zeros(self.num_images, 12, self.grid_L, self.grid_H, self.grid_W,
                        dtype=torch.float32, device='cuda')
        g[:, 0, ...] = 1.0
        g[:, 5, ...] = 1.0
        g[:, 10, ...] = 1.0
        self.grids = nn.Parameter(g)
        # Per-step accumulators (NOT parameters; not exposed to autograd)
        self.register_buffer('exp_avg', torch.zeros_like(self.grids.data))
        self.register_buffer('exp_avg_sq', torch.zeros_like(self.grids.data))
        self.register_buffer('accum_grads', torch.zeros_like(self.grids.data))

        # Hyperparameters
        self.lr = float(lr)
        self.initial_lr = float(lr)
        self._current_lr = float(lr * warmup_start_factor)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.warmup_steps = int(warmup_steps)
        self.total_iterations = int(total_iterations)
        self.warmup_start_factor = float(warmup_start_factor)
        self.final_lr_factor = float(final_lr_factor)
        self.tv_weight = float(tv_weight)
        self._step = 0

    # ── freeze gate (P0-Bilateral-v2) ──
    def engaged(self, iteration):
        """True iff the grid Adam should step at `iteration`.

        Until start_iter the grid stays frozen at identity — this is the
        v2 design fix that eliminates the failure mode observed on
        campus (the grid drifted during densification because it had
        nothing useful to learn but still received photometric grads).
        """
        return int(iteration) >= self.start_iter

    # ── forward ──
    def forward(self, rgb, image_idx):
        """Apply grid to image. rgb: [H, W, 3]; returns [H, W, 3]."""
        idx = int(image_idx)
        # Zero the ENTIRE accum_grads buffer (cheap: ~2.5 MB). The CUDA
        # backward kernel will atomicAdd only into slot[idx] from this
        # iter; other slots must be zeroed so Adam doesn't see stale
        # gradients from prior iters when it processes those slots below.
        self.accum_grads.zero_()
        return _BilateralGridApply.apply(
            self.grids[idx], rgb,
            self.grid_W, self.grid_H, self.grid_L,
            self.accum_grads[idx])

    # ── Adam step (fused-PyTorch, mirrors LFS fused-Adam kernel) ──
    @torch.no_grad()
    def optimizer_step(self):
        step = self._step + 1
        bc1_rcp = 1.0 / (1.0 - self.beta1 ** step)
        bc2_rcp = 1.0 / (1.0 - self.beta2 ** step)
        m = self.exp_avg
        v = self.exp_avg_sq
        m.mul_(self.beta1).add_(self.accum_grads, alpha=1.0 - self.beta1)
        v.mul_(self.beta2).addcmul_(self.accum_grads, self.accum_grads, value=1.0 - self.beta2)
        m_hat = m.mul(bc1_rcp)
        v_hat = v.mul(bc2_rcp)
        # Update: grids -= lr * m_hat / (sqrt(v_hat) + eps)
        v_sqrt = v_hat.sqrt().add_(self.eps)
        update = m_hat / v_sqrt
        self.grids.data.add_(update, alpha=-self._current_lr)

    @torch.no_grad()
    def scheduler_step(self):
        """Linear warmup from lr*warmup_start_factor -> lr over warmup_steps,
        then exponential decay to lr*final_lr_factor over remaining iters.
        Matches LFS gsplat BilateralGrid::scheduler_step."""
        self._step += 1
        if self._step <= self.warmup_steps:
            progress = self._step / self.warmup_steps
            scale = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * progress
        else:
            denom = max(1, self.total_iterations - self.warmup_steps)
            gamma = self.final_lr_factor ** (1.0 / denom)
            scale = gamma ** (self._step - self.warmup_steps)
        self._current_lr = self.initial_lr * scale

    @torch.no_grad()
    def zero_grad(self):
        self.accum_grads.zero_()

    # ── TV loss (pure PyTorch; grid is tiny, no CUDA needed) ──
    @torch.no_grad()
    def tv_backward(self, tv_weight):
        """Add TV gradient to accum_grads (matching LFS fused-kernel
        behavior: centered-TV / 6*N). TV = sum of squared diffs along
        the 3 spatial axes; gradient centers each cell (forward + back
        difference, divided by 6).

        accum_grads is expected to already contain the photometric
        gradient (added by the autograd.Fn backward). This adds the
        TV gradient on top, scaled by tv_weight.
        """
        if tv_weight == 0.0:
            return
        g = self.grids.data  # [N, 12, L, H, W]
        grad = self.accum_grads
        N = g.shape[0]
        s = tv_weight / (6.0 * N)
        sx_w = 1.0 / (g.shape[1] * g.shape[2] * (g.shape[3] - 1)) if g.shape[3] > 1 else 0.0
        sy_h = 1.0 / (g.shape[1] * (g.shape[2] - 1) * g.shape[3]) if g.shape[2] > 1 else 0.0
        sz_l = 1.0 / ((g.shape[1] - 1) * g.shape[2] * g.shape[3]) if g.shape[1] > 1 else 0.0

        # W axis: cell i contributes (v[i]-v[i-1]) + (v[i]-v[i+1])
        if g.shape[3] > 1:
            grad_w = torch.zeros_like(g)
            grad_w[..., 1:-1].add_(2 * g[..., 1:-1] - g[..., :-2] - g[..., 2:])
            grad_w[..., 0, :].add_(g[..., 0, :] - g[..., 1, :])
            grad_w[..., -1, :].add_(g[..., -1, :] - g[..., -2, :])
            grad.add_(grad_w, alpha=s * sx_w)

        # H axis (2nd-to-last spatial dim)
        if g.shape[2] > 1:
            grad_h = torch.zeros_like(g)
            grad_h[..., 1:-1, :].add_(2 * g[..., 1:-1, :] - g[..., :-2, :] - g[..., 2:, :])
            grad_h[..., 0, :, :].add_(g[..., 0, :, :] - g[..., 1, :, :])
            grad_h[..., -1, :, :].add_(g[..., -1, :, :] - g[..., -2, :, :])
            grad.add_(grad_h, alpha=s * sy_h)

        # L axis (3rd-to-last dim)
        if g.shape[1] > 1:
            grad_l = torch.zeros_like(g)
            grad_l[..., 1:-1, :, :].add_(2 * g[..., 1:-1, :, :] - g[..., :-2, :, :] - g[..., 2:, :, :])
            grad_l[..., 0, :, :, :].add_(g[..., 0, :, :, :] - g[..., 1, :, :, :])
            grad_l[..., -1, :, :, :].add_(g[..., -1, :, :, :] - g[..., -2, :, :, :])
            grad.add_(grad_l, alpha=s * sz_l)

    # ── Display-only scalar loss (no grad) ──
    @torch.no_grad()
    def tv_loss_value(self):
        """Display-only scalar (no grad)."""
        with torch.no_grad():
            g = self.grids.data
            N = g.shape[0]
            diff_x = g[..., 1:] - g[..., :-1]
            diff_y = g[..., 1:, :] - g[..., :-1, :]
            diff_z = g[..., 1:, :, :] - g[..., :-1, :, :]
            sx = diff_x.pow(2).sum() / (N * 12 * g.shape[2] * max(1, g.shape[3] - 1))
            sy = diff_y.pow(2).sum() / (N * 12 * g.shape[1] * max(1, g.shape[2] - 1))
            sz = diff_z.pow(2).sum() / (N * 12 * max(1, g.shape[1] - 1) * g.shape[2])
            return (sx + sy + sz).item()
