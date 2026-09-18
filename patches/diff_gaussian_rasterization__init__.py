#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
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
):
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
    ):
        # PyBind11 does NOT auto-convert Python `None` -> default-constructed
        # torch.Tensor for the C++ defaults (`torch::Tensor()`). Pass an empty
        # tensor explicitly for optional tensor args.
        if cov3Ds_precomp is None:
            cov3Ds_precomp = torch.Tensor()
        gt_image = raster_settings.gt_image
        if gt_image is None:
            gt_image = torch.Tensor()

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
            raster_settings.error_state_handle,  # NEW in your version
            gt_image,                              # NEW in your version
            raster_settings.compute_error,         # NEW in your version
        )

        # Invoke C++/CUDA rasterizer
        # C++ returns 8 things: (num_rendered, color, radii, geomBuffer,
        # binningBuffer, imgBuffer, invdepths, error)
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths, _error = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

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
            None,
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool
    antialiasing: bool
    # NEW in your optimized version
    error_state_handle: int = 0
    gt_image: torch.Tensor = None
    compute_error: bool = False


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
        return visible

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide exactly one of either SHs or precomputed colors!')

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
        )


class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr=0., betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr, betas, eps, weight_decay)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            max_exp_avg_sqs = []
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if amsgrad:
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    state_steps.append(state['step'])
                    if amsgrad:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

            try:
                _C.sparse_adam_step(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad,
                    group['lr'],
                    group['betas'][0],
                    group['betas'][1],
                    group['eps'],
                    group['weight_decay'],
                )
            except AttributeError:
                # If _C doesn't expose sparse_adam_step, fall back to torch.optim.Adam.step()
                super().step(closure=None)
        return loss


class BilateralGrid(nn.Module):
    """Per-image appearance correction via a learnable bilateral filter grid.

    Forward takes an image (3, H, W) and an image index, returns the corrected
    image. Backward accumulates grads into self.grid via _C.bilateral_grid_backward.
    """
    def __init__(self, num_images, grid_W=16, grid_H=16, grid_L=8,
                 lr=0.02, tv_weight=0.01, start_iter=26000, total_iterations=30000,
                 device='cuda'):
        super().__init__()
        self.num_images = num_images
        self.grid_W = grid_W
        self.grid_H = grid_H
        self.grid_L = grid_L
        self.lr = lr
        self.tv_weight = tv_weight
        self.start_iter = start_iter
        self.total_iterations = total_iterations

        grid = torch.ones(num_images, 12, grid_W, grid_H, grid_L, device=device)
        grid[:, :, :, :, -1] = 0.1
        self.grid = nn.Parameter(grid)
        self.color_scale = nn.Parameter(torch.ones(3, device=device))

    def forward(self, image_idx, image, iter):
        if iter < self.start_iter:
            return image
        if not torch.is_tensor(image_idx):
            image_idx = torch.tensor(image_idx, dtype=torch.long, device=image.device)
        idx = image_idx.long()
        # C++ binding expects grid [12, L, H, W] and rgb [h, w, 3].
        # self.grid slice is [12, grid_W, grid_H, grid_L]; permute to
        # [12, L, H, W] (L=grid_L, H=grid_H, W=grid_W).
        grid_slice = self.grid[idx:idx+1].squeeze(0).permute(0, 3, 2, 1).contiguous()
        # image is (3, H, W) — convert to HWC (h, w, 3) for the kernel.
        rgb_hwc = image.permute(1, 2, 0).contiguous()
        out = _C.bilateral_grid_forward(grid_slice, rgb_hwc)
        # out shape: (H, W, 3) → (3, H, W)
        return out.permute(2, 0, 1) * self.color_scale[:, None, None]

    def tv_loss(self):
        return self.tv_weight * (
            torch.mean(torch.abs(self.grid[:, :, :, :, :-1] - self.grid[:, :, :, :, 1:])) +
            torch.mean(torch.abs(self.grid[:, :, :, :-1, :] - self.grid[:, :, 1:, :])) +
            torch.mean(torch.abs(self.grid[:, :, :-1, :, :] - self.grid[:, :, 1:, :, :]))
        )

    def accum_grads(self, image_idx, img1, img2, opt_grad):
        if not torch.is_tensor(image_idx):
            image_idx = torch.tensor(image_idx, dtype=torch.long, device=img1.device)
        idx = image_idx.long()
        grid_slice = self.grid[idx:idx+1].squeeze(0).permute(0, 3, 2, 1).contiguous()
        img1_hwc = img1.permute(1, 2, 0).contiguous()
        img2_hwc = img2.permute(1, 2, 0).contiguous()
        opt_grad_hwc = opt_grad.permute(1, 2, 0).contiguous()
        _C.bilateral_grid_backward(grid_slice, img1_hwc, img2_hwc, opt_grad_hwc)
