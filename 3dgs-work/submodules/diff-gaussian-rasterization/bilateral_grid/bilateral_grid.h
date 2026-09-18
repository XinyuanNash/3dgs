/*
 * Bilateral Grid C++ binding declarations.
 *
 * The CUDA kernels live in bilateral_grid.cu and define:
 *   torch::Tensor BilateralGridForwardCUDA(grid, rgb);
 *   torch::Tensor BilateralGridBackwardCUDA(grid, rgb, grad_out, grad_grid);
 *
 * Layout:
 *   grid     : float [12, L, H, W]   (single image slice selected on Python side)
 *   rgb      : float [h, w, 3]       (HWC, in [0, 1])
 *   grad_out : float [h, w, 3]       (dL/d(out))
 *   grad_grid: float [12, L, H, W]   (IN/OUT — caller zeroes per step; atomicAdd here)
 *
 * Returns:
 *   forward  : float [h, w, 3]       (corrected rgb)
 *   backward : float [h, w, 3]       (dL/d(rgb))
 */

#pragma once
#include <torch/extension.h>

torch::Tensor BilateralGridForwardCUDA(
    torch::Tensor grid,
    torch::Tensor rgb);

torch::Tensor BilateralGridBackwardCUDA(
    torch::Tensor grid,
    torch::Tensor rgb,
    torch::Tensor grad_out,
    torch::Tensor grad_grid);
