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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	// [P0-WarpBallot] warp-granularity early exit. Replace per-lane `return`
	// at the 3 cull points with `bool active` flag + `cg::tiled_partition<32>`
	// ballot. Per-lane `return` below a cg-collective is undefined behavior;
	// lanes stay alive through every ballot and exit only when the entire
	// warp is culled. Tail lanes (idx >= P) stay inactive and vote 0.
	auto warp = cg::tiled_partition<32>(cg::this_thread_block());
	auto idx = cg::this_grid().thread_rank();
	bool active = (idx < P);

	// Variables that must cross a ballot boundary are hoisted to function
	// scope so the post-ballot `if(active)` block can read them.
	float3 p_view = { 0.f, 0.f, 0.f };
	float3 cov = { 0.f, 0.f, 0.f };
	float det = 0.f;
	float3 conic = { 0.f, 0.f, 0.f };
	float my_radius = 0.f;
	float2 point_image = { 0.f, 0.f };
	uint2 rect_min = { 0u, 0u }, rect_max = { 0u, 0u };
	float opacity = 0.f;
	float h_convolution_scaling = 1.0f;
	const float* cov3D = nullptr;

	if (active)
	{
		// Initialize radius and touched tiles to 0. If this isn't changed,
		// this Gaussian will not be processed further.
		radii[idx] = 0;
		tiles_touched[idx] = 0;

		// Perform near culling, quit if outside.
		if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
			active = false;
	}
	// Ballot #1: skip everything past in_frustum if the entire warp was culled.
	if (warp.ballot(active) == 0)
		return;

	if (active)
	{
		// Transform point by projecting
		float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
		float4 p_hom = transformPoint4x4(p_orig, projmatrix);
		float p_w = 1.0f / (p_hom.w + 0.0000001f);
		float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

		// If 3D covariance matrix is precomputed, use it, otherwise compute
		// from scaling and rotation parameters.
		if (cov3D_precomp != nullptr)
		{
			cov3D = cov3D_precomp + idx * 6;
		}
		else
		{
			computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
			cov3D = cov3Ds + idx * 6;
		}

		// Compute 2D screen-space covariance matrix
		cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

		constexpr float h_var = 0.3f;
		// [P0-AddBlur] delegate MipSplatting AA to add_blur() helper (auxiliary.h).
		// cov.x = c_xx, cov.y = c_xy, cov.z = c_yy.
		const float det_cov_plus_h_cov = add_blur(h_var, cov.x, cov.z, cov.y, h_convolution_scaling, antialiasing);

		// Invert covariance (EWA algorithm)
		det = det_cov_plus_h_cov;
		if (det == 0.0f)
			active = false;

		float det_inv = (det != 0.f) ? (1.f / det) : 0.f;
		conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

		// Compute extent in screen space (by finding eigenvalues of
		// 2D covariance matrix). Use extent to compute a bounding rectangle
		// of screen-space tiles that this Gaussian overlaps with. Quit if
		// rectangle covers 0 tiles.
		float mid = 0.5f * (cov.x + cov.z);
		float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
		float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
		my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
		point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
		getRect(point_image, my_radius, rect_min, rect_max, grid);
		if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
			active = false;
	}
	// Ballot #2: skip SH + stores if the entire warp was culled by det==0
	// or rect area==0.
	if (warp.ballot(active) == 0)
		return;

	if (active)
	{
		// If colors have been precomputed, use them, otherwise convert
		// spherical harmonics coefficients to RGB color.
		if (colors_precomp == nullptr)
		{
			glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
			rgb[idx * C + 0] = result.x;
			rgb[idx * C + 1] = result.y;
			rgb[idx * C + 2] = result.z;
		}

		// Store some useful helper data for the next steps.
		depths[idx] = p_view.z;
		radii[idx] = my_radius;
		points_xy_image[idx] = point_image;
		// Inverse 2D covariance and opacity neatly pack into one float4
		opacity = opacities[idx];

		conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };

		tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	}
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			if(invdepth)
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths, 
		depth);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}

// =====================================================================
// P1-7+ (OPTIMIZATIONS.md §13.6): per-Gaussian pixel-error accumulation
// kernel. Launched AFTER the BLEND kernel, ONLY when compute_error=true.
// Reverse-walks the same `point_list`/`ranges` that BLEND used, so it
// touches the same per-tile Gaussians in reverse order (matching the
// backward.cu:536-560 pattern exactly) and can compute alpha in the same
// way. For each contributing Gaussian g at pixel p, atomic-adds
//   alpha_g * T_after_g * pixel_err_p
// into `error_buffer[g]`. After the kernel, `error_buffer[g]` is the
// total reconstruction error attributable to g, summed over all pixels
// it contributed to.
//
// This mirrors LFS MCMC's `_error_score_max` accumulator (the difference
// is LFS uses per-camera ema max; we use one-shot accumulation and let
// Python do the ema downstream). The post-blend split (separate kernel,
// not a modification of BLEND) is REQUIRED because BLEND does not carry
// per-(Gaussian, alpha, T_after) tuples out of its inner accumulation
// loop — by the time final_color is written to out_color, the per-
// Gaussian contributions are gone. Re-deriving them in a second pass is
// cheap (same fetch, no new geometry work) and keeps BLEND bit-identical
// when compute_error=false.
// =====================================================================
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
attributeErrorsCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ out_color,        // [3, H, W] from BLEND
	const float* __restrict__ gt_image,         // [3, H, W] in [0,1]
	const float* __restrict__ final_Ts,         // [H*W] from BLEND
	const uint32_t* __restrict__ n_contrib,     // [H*W] from BLEND
	float* __restrict__ error_buffer            // [P], pre-zeroed on current stream
)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y, H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Reverse start: T = final_T, walk BACK through contributing Gaussians.
	const float T_final = inside ? final_Ts[pix_id] : 0.0f;
	float T = T_final;

	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	// Pre-compute per-pixel reconstruction error (broadcast scalar — same
	// value used for every contributing Gaussian). 0.5 mean L1 over 3
	// channels keeps the magnitude in [0, 1] so error_buffer stays sane
	// when summed over many pixels. This is the L1 counterpart to LFS's
	// SSIM-derived pixel_err; simpler & no autograd interaction.
	float pixel_err = 0.0f;
	if (inside)
	{
		const float diff_r = out_color[0 * H * W + pix_id] - gt_image[0 * H * W + pix_id];
		const float diff_g = out_color[1 * H * W + pix_id] - gt_image[1 * H * W + pix_id];
		const float diff_b = out_color[2 * H * W + pix_id] - gt_image[2 * H * W + pix_id];
		pixel_err = 0.5f * (fabsf(diff_r) + fabsf(diff_g) + fabsf(diff_b)) / 3.0f;
	}

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load a batch from the BACK of the point_list (mirror of backward.cu:540-553).
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Skip Gaussians that were not contributors in forward (mirror of
			// backward.cu:561-563).
			contributor--;
			if (contributor >= last_contributor)
				continue;

			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			// Reconstruct T_before = T_after / (1 - alpha) (same identity
			// backward.cu:578 uses for the gradient path). Then the weight
			// this Gaussian contributed to this pixel's color is alpha*T.
			T = T / (1.0f - alpha);
			const float weight = alpha * T;

			const int global_id = collected_id[j];
			atomicAdd(&error_buffer[global_id], weight * pixel_err);
		}
	}
}

void FORWARD::attributeErrors(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float4* conic_opacity,
	const float* out_color,
	const float* gt_image,
	const float* final_Ts,
	const uint32_t* n_contrib,
	float* error_buffer)
{
	attributeErrorsCUDA << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		conic_opacity,
		out_color,
		gt_image,
		final_Ts,
		n_contrib,
		error_buffer);
}
