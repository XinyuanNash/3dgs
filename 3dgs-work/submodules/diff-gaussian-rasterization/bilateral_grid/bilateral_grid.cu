/*
 * Bilateral Grid — port of LFS gsplat bilateral_grid_forward.cu / backward.cu
 * (LFS: /LichtFeld-Studio/src/training/kernels/)
 *
 * Layout:
 *   grid[N, 12, L, H, W]    — 12 = 3 (output R/G/B) × 4 (input R/G/B/1 bias)
 *   rgb[H, W, 3]            — HWC, [0, 1]
 *   per-channel ci:
 *     si = ci % 4   (0=R, 1=G, 2=B, 3=bias)
 *     di = ci / 4   (0=R, 1=G, 2=B)
 *   out[di, hi, wi] += grid[ci, ...]_interp * (R/G/B/1)[si]
 *
 * Identity init = channels [0, 5, 10] = 1.0, rest = 0.0  →  out == rgb.
 *
 * Bilateral coords:
 *   x = wi/(W-1)*(W-1) (edge-clamp if W==1)
 *   y = hi/(H-1)*(H-1)
 *   z = luma(R,G,B) * (L-1), luma = clamp(0.299*R+0.587*G+0.114*B, 0, 1)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

namespace {
constexpr int BLOCK_SIZE = 256;
constexpr float kC2G_r = 0.299f;
constexpr float kC2G_g = 0.587f;
constexpr float kC2G_b = 0.114f;
}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// FORWARD
//   grid_slice: float [12, L, H, W]
//   rgb       : float [H, W, 3]
//   out       : float [H, W, 3]   (caller-allocated, initialized to zero)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void bilateral_grid_forward_hwc_kernel(
    const float* __restrict__ grid,
    const float* __restrict__ rgb,
    float* __restrict__ out,
    const int L, const int H, const int W,
    const int h, const int w)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= h * w) return;

    const int wi = idx % w;
    const int hi = idx / w;

    const int rgb_base = idx * 3;
    const float sr = rgb[rgb_base + 0];
    const float sg = rgb[rgb_base + 1];
    const float sb = rgb[rgb_base + 2];
    const bool finite_in = isfinite(sr) && isfinite(sg) && isfinite(sb);
    const float r = finite_in ? sr : 0.5f;
    const float g = finite_in ? sg : 0.5f;
    const float b = finite_in ? sb : 0.5f;

    // 1) Coordinates
    const float x = (w > 1) ? (static_cast<float>(wi) / (w - 1) * (W - 1)) : 0.0f;
    const float y = (h > 1) ? (static_cast<float>(hi) / (h - 1) * (H - 1)) : 0.0f;
    const float lum = fminf(1.0f, fmaxf(0.0f, kC2G_r * r + kC2G_g * g + kC2G_b * b));
    const float z = lum * (L - 1);

    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    int z0 = static_cast<int>(floorf(z));
    const int x1 = min(x0 + 1, W - 1);
    const int y1 = min(y0 + 1, H - 1);
    int z1 = z0 + 1;
    z0 = min(max(z0, 0), L - 1);
    z1 = min(max(z1, 0), L - 1);
    const float fx = x - x0;
    const float fy = y - y0;
    const float fz = z - z0;

    // 2) Decode 12 channels → affine+bias accumulation per output channel
    float dr = 0.0f, dg = 0.0f, db = 0.0f;
    for (int ci = 0; ci < 12; ++ci) {
        const int di = ci >> 2;       // /4
        const int si = ci & 3;        // %4

        const int base = ci * L * H * W;

        // 8 corners (clamped in all dimensions; we already clamped z)
        const int v000_off = base + (z0 * H + y0) * W + x0;
        const int v001_off = base + (z0 * H + y0) * W + x1;
        const int v010_off = base + (z0 * H + y1) * W + x0;
        const int v011_off = base + (z0 * H + y1) * W + x1;
        const int v100_off = base + (z1 * H + y0) * W + x0;
        const int v101_off = base + (z1 * H + y0) * W + x1;
        const int v110_off = base + (z1 * H + y1) * W + x0;
        const int v111_off = base + (z1 * H + y1) * W + x1;

        const float v000 = grid[v000_off];
        const float v001 = grid[v001_off];
        const float v010 = grid[v010_off];
        const float v011 = grid[v011_off];
        const float v100 = grid[v100_off];
        const float v101 = grid[v101_off];
        const float v110 = grid[v110_off];
        const float v111 = grid[v111_off];

        const float c00 = v000 * (1 - fx) + v001 * fx;
        const float c01 = v010 * (1 - fx) + v011 * fx;
        const float c10 = v100 * (1 - fx) + v101 * fx;
        const float c11 = v110 * (1 - fx) + v111 * fx;
        const float c0  = c00  * (1 - fy) + c01  * fy;
        const float c1  = c10  * (1 - fy) + c11  * fy;
        const float v   = c0   * (1 - fz) + c1   * fz;

        const float a = (si == 0) ? r : (si == 1) ? g : (si == 2) ? b : 1.0f;

        float& dst = (di == 0) ? dr : (di == 1) ? dg : db;
        dst += v * a;
    }

    out[rgb_base + 0] = isfinite(dr) ? dr : 0.5f;
    out[rgb_base + 1] = isfinite(dg) ? dg : 0.5f;
    out[rgb_base + 2] = isfinite(db) ? db : 0.5f;
}

// ─────────────────────────────────────────────────────────────────────────────
// BACKWARD
//   grid_slice: float [12, L, H, W]   forward input
//   rgb       : float [H, W, 3]       forward input
//   grad_out  : float [H, W, 3]       dL/d(out)
//   grad_grid : float [12, L, H, W]   (IN/OUT — caller zeroes per step; atomicAdd here)
//   grad_rgb  : float [H, W, 3]       dL/d(rgb)  (caller-allocated)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void bilateral_grid_backward_hwc_kernel(
    const float* __restrict__ grid,
    const float* __restrict__ rgb,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_grid,
    float* __restrict__ grad_rgb,
    const int L, const int H, const int W,
    const int h, const int w)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= h * w) return;

    const int wi = idx % w;
    const int hi = idx / w;
    const int rgb_base = idx * 3;

    const float sr = rgb[rgb_base + 0];
    const float sg = rgb[rgb_base + 1];
    const float sb = rgb[rgb_base + 2];
    const float dr_g = grad_out[rgb_base + 0];
    const float dg_g = grad_out[rgb_base + 1];
    const float db_g = grad_out[rgb_base + 2];
    const bool finite = isfinite(sr) && isfinite(sg) && isfinite(sb)
                     && isfinite(dr_g) && isfinite(dg_g) && isfinite(db_g);

    const float r = finite ? sr : 0.0f;
    const float g = finite ? sg : 0.0f;
    const float b = finite ? sb : 0.0f;
    const float go_r = finite ? dr_g : 0.0f;
    const float go_g = finite ? dg_g : 0.0f;
    const float go_b = finite ? db_g : 0.0f;

    const float x = (w > 1) ? (static_cast<float>(wi) / (w - 1) * (W - 1)) : 0.0f;
    const float y = (h > 1) ? (static_cast<float>(hi) / (h - 1) * (H - 1)) : 0.0f;
    const float lum = fminf(1.0f, fmaxf(0.0f, kC2G_r * r + kC2G_g * g + kC2G_b * b));
    const float z = lum * (L - 1);

    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    int z0 = static_cast<int>(floorf(z));
    const int x1 = min(x0 + 1, W - 1);
    const int y1 = min(y0 + 1, H - 1);
    int z1 = z0 + 1;
    z0 = min(max(z0, 0), L - 1);
    z1 = min(max(z1, 0), L - 1);
    const float fx = x - x0;
    const float fy = y - y0;
    const float fz = z - z0;
    const int z_int = static_cast<int>(floorf(z));
    const float dfz = static_cast<float>(L - 1);
    const bool z_off_grid = (z_int != z_int);  // never true; used to skip when z exactly on slice
    const float lum_grad_sign = (z0 != static_cast<int>(floorf(z)) && z1 != static_cast<int>(floorf(z))) ? 1.0f : 0.0f;

    float g_r_acc = 0.0f, g_g_acc = 0.0f, g_b_acc = 0.0f;
    float gz_grad = 0.0f;

    // Per-channel + per-corner
    for (int ci = 0; ci < 12; ++ci) {
        const int di = ci >> 2;       // 0/1/2 = R/G/B output
        const int si = ci & 3;        // 0/1/2/3 = R/G/B/bias input
        const float gout = (di == 0) ? go_r : (di == 1) ? go_g : go_b;
        const float rin = (si == 0) ? r : (si == 1) ? g : (si == 2) ? b : 1.0f;

        const int base = ci * L * H * W;

        // 8 corner coords (xi, yi, zi)
        const float corners_x[8] = {
            static_cast<float>(x0), static_cast<float>(x1),
            static_cast<float>(x0), static_cast<float>(x1),
            static_cast<float>(x0), static_cast<float>(x1),
            static_cast<float>(x0), static_cast<float>(x1)
        };
        const float corners_y[8] = {
            static_cast<float>(y0), static_cast<float>(y0),
            static_cast<float>(y1), static_cast<float>(y1),
            static_cast<float>(y0), static_cast<float>(y0),
            static_cast<float>(y1), static_cast<float>(y1)
        };
        const int corners_z[8] = { z0, z0, z0, z0, z1, z1, z1, z1 };

        // Trilinear weights
        const float wx[2] = { 1.0f - fx, fx };
        const float wy[2] = { 1.0f - fy, fy };
        const float wz[2] = { 1.0f - fz, fz };

        // Pre-loaded grid values (8 corners) — will be used twice below
        float vs[8];
        for (int c = 0; c < 8; ++c) {
            vs[c] = grid[base + (corners_z[c] * H + static_cast<int>(corners_y[c])) * W
                          + static_cast<int>(corners_x[c])];
        }

        // Compute weights (8 weights per channel — same for all 12 channels of a pixel
        // because (x0,x1),(y0,y1),(z0,z1) are fixed per pixel)
        float wts[8];
        const float dwdz[8] = {
            -wz[0] * wy[0] * wx[0], -wz[0] * wy[0] * wx[1],
            -wz[0] * wy[1] * wx[0], -wz[0] * wy[1] * wx[1],
             wz[1] * wy[0] * wx[0],  wz[1] * wy[0] * wx[1],
             wz[1] * wy[1] * wx[0],  wz[1] * wy[1] * wx[1],
        };
        const float w00 = wy[0] * wx[0], w01 = wy[0] * wx[1];
        const float w10 = wy[1] * wx[0], w11 = wy[1] * wx[1];
        wts[0] = wz[0] * w00; wts[1] = wz[0] * w01;
        wts[2] = wz[0] * w10; wts[3] = wz[0] * w11;
        wts[4] = wz[1] * w00; wts[5] = wz[1] * w01;
        wts[6] = wz[1] * w10; wts[7] = wz[1] * w11;

        // Trilerp accumulator (for chain-rule on lum/z)
        float trilerp = 0.0f;

        for (int c = 0; c < 8; ++c) {
            const float wt = wts[c];
            const int dz = corners_z[c];
            const int dy = static_cast<int>(corners_y[c]);
            const int dx = static_cast<int>(corners_x[c]);

            // Atomic add into the grid grad: idx within the 12,L,H,W slab
            atomicAdd(&grad_grid[base + (dz * H + dy) * W + dx],
                      wt * rin * gout);

            // If si < 3 (R/G/B), propagate gradient back to rgb input
            if (si < 3) {
                float& g_dst = (si == 0) ? g_r_acc : (si == 1) ? g_g_acc : g_b_acc;
                g_dst += vs[c] * wt * gout;
            }

            trilerp += vs[c] * rin * gout;
        }

        // Chain rule through luminance → z axis
        // d(luma)/d(R/G/B) and d(luma)/d(z), but here we accumulate per output-di channel.
        // The LFS impl computes gz_grad for each ci and aggregates across the per-channel
        // gout, weighted by sum_channel (== trilerp for each ci). We replicate that:
        //   gz_grad += dwdz[c] * trilerp_v * gout_di
        // (For our flattened loop, weighted by each ci's trilerp.)
        // The dwdz contribution for this ci: sum_c dwdz[c] * v[c] * rin * gout
        // We rebuild dwdz_v below efficiently.
        float dwdz_v = 0.0f;
        for (int c = 0; c < 8; ++c) {
            dwdz_v += dwdz[c] * vs[c];
        }
        gz_grad += dwdz_v * rin * gout * dfz * lum_grad_sign;
        // (dfz = L-1, included for completeness; matches LFS kernel.)
    }

    // Luma gradient back to R/G/B
    if (lum_grad_sign > 0.0f) {
        g_r_acc += kC2G_r * gz_grad;
        g_g_acc += kC2G_g * gz_grad;
        g_b_acc += kC2G_b * gz_grad;
    }

    grad_rgb[rgb_base + 0] = g_r_acc;
    grad_rgb[rgb_base + 1] = g_g_acc;
    grad_rgb[rgb_base + 2] = g_b_acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// C++ binding wrappers (called by bilateral_grid.cpp)
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor BilateralGridForwardCUDA(
    torch::Tensor grid,
    torch::Tensor rgb)
{
    TORCH_CHECK(grid.is_cuda() && rgb.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(grid.dtype() == torch::kFloat32 && rgb.dtype() == torch::kFloat32,
                "Inputs must be float32");
    TORCH_CHECK(grid.dim() == 4, "grid must be [12, L, H, W]");
    TORCH_CHECK(rgb.dim() == 3 && rgb.size(2) == 3, "rgb must be [h, w, 3]");
    TORCH_CHECK(grid.is_contiguous() && rgb.is_contiguous(),
                "Inputs must be contiguous");

    const int L = grid.size(1);
    const int H = grid.size(2);
    const int W = grid.size(3);
    const int h = rgb.size(0);
    const int w = rgb.size(1);

    auto opts = grid.options();
    auto out = torch::empty({h, w, 3}, opts);

    const int blocks = (h * w + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bilateral_grid_forward_hwc_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        grid.data_ptr<float>(),
        rgb.data_ptr<float>(),
        out.data_ptr<float>(),
        L, H, W, h, w);
    return out;
}

torch::Tensor BilateralGridBackwardCUDA(
    torch::Tensor grid,
    torch::Tensor rgb,
    torch::Tensor grad_out,
    torch::Tensor grad_grid)
{
    TORCH_CHECK(grid.is_cuda() && rgb.is_cuda() && grad_out.is_cuda() && grad_grid.is_cuda(),
                "Inputs must be CUDA tensors");
    TORCH_CHECK(grid.dtype() == torch::kFloat32 && rgb.dtype() == torch::kFloat32
                && grad_out.dtype() == torch::kFloat32 && grad_grid.dtype() == torch::kFloat32,
                "Inputs must be float32");
    TORCH_CHECK(grid.dim() == 4, "grid must be [12, L, H, W]");
    TORCH_CHECK(rgb.dim() == 3 && rgb.size(2) == 3, "rgb must be [h, w, 3]");
    TORCH_CHECK(grad_out.sizes() == rgb.sizes(), "grad_out must match rgb");
    TORCH_CHECK(grad_grid.sizes() == grid.sizes(), "grad_grid must match grid");

    const int L = grid.size(1);
    const int H = grid.size(2);
    const int W = grid.size(3);
    const int h = rgb.size(0);
    const int w = rgb.size(1);

    auto opts = rgb.options();
    auto grad_rgb = torch::empty({h, w, 3}, opts);

    const int blocks = (h * w + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bilateral_grid_backward_hwc_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        grid.data_ptr<float>(),
        rgb.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        grad_grid.data_ptr<float>(),
        grad_rgb.data_ptr<float>(),
        L, H, W, h, w);
    return grad_rgb;
}
