/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"
#include "cuda_rasterizer/rasterizer_state.h"
#include "bilateral_grid/bilateral_grid.h"
#include <memory>

// P1-7+ (OPTIMIZATIONS.md §13.6): expose the C++ RasterizerState to
// Python via int64 handles. Python owns the lifetime via the deleter
// lambda we pass to pybind11 (which calls `delete` on the C++ side).
static RasterizerStateHandle make_rasterizer_state()
{
	auto* s = new CudaRasterizer::RasterizerState();
	return reinterpret_cast<RasterizerStateHandle>(s);
}

static void free_rasterizer_state(RasterizerStateHandle handle)
{
	if (handle != 0)
	{
		auto* s = reinterpret_cast<CudaRasterizer::RasterizerState*>(handle);
		delete s;
	}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  // P0-Bilateral-v2 (2026-08-04): per-image appearance correction via
  // BilateralGrid (LFS gsplat port). Enabled opt-in via
  // ``--use_bilateral_grid``. Forward = HWC layout [H, W, 3]; backward
  // atomicAdds photometric grads into the per-image slice of
  // ``grad_grid_buf`` (managed in Python by BilateralGrid.accum_grads).
  m.def("bilateral_grid_forward", &BilateralGridForwardCUDA,
        "Bilateral grid forward (HWC)");
  m.def("bilateral_grid_backward", &BilateralGridBackwardCUDA,
        "Bilateral grid backward (HWC) + atomicAdd into grad_grid");
  // P1-7+ rasterizer-state lifecycle: Python side calls make_rasterizer_state
  // once (returns handle), passes it to rasterize_gaussians each call, and
  // calls free_rasterizer_state on teardown.
  m.def("make_rasterizer_state", &make_rasterizer_state);
  m.def("free_rasterizer_state", &free_rasterizer_state);
}