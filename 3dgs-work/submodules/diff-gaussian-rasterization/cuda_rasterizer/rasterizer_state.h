/*
 * P1-7+ RasterizerState (OPTIMIZATIONS.md §13.6)
 *
 * Persistent per-Gaussian state attached to the rasterizer, separate from
 * the per-call functional buffers (geomBuffer/binningBuffer/imgBuffer).
 * Holds the `_error_score_max`-style buffer: one float per Gaussian,
 * accumulated across pixels during the post-blend kernel via atomicAdd.
 *
 * Lifetime contract:
 *   - Constructed once on the Python side (held by `rasterizer` module or
 *     GaussianModel), `resize_if_needed(P)` is called before each forward
 *     when P may have grown (densification). Memory is **only** allocated
 *     on growth — never freed except on destruction. This matches the
 *     PyTorch caching-allocator warmup pattern and prevents per-step
 *     fragmentation (per P0-TightAABB iter-610 OOM lessons).
 *   - `zero_current_stream()` is called once per step on the active CUDA
 *     stream (via `at::cuda::getCurrentCUDAStream()`); it uses
 *     `cudaMemsetAsync` to zero the buffer asynchronously, ordering after
 *     all prior writes on the same stream and before the post-blend
 *     kernel launch.
 *   - The `error_buffer_ptr()` is passed as a raw `float*` into the
 *     post-blend kernel. The Python wrapper does NOT own a copy; the
 *     state struct is the sole owner.
 *
 * Why this exists instead of per-call `torch::empty({P})` in
 * rasterize_points.cu: a persistent lifetime makes the buffer's growth
 * pattern (P grows by clone, never shrinks within an MCMC cycle) explicit
 * and lets `resize_if_needed` be the single growth point. Future LFS ports
 * (MRNF confidence buffer, IGS+ edge cache) can follow the same pattern.
 */
#ifndef CUDA_RASTERIZER_STATE_H_INCLUDED
#define CUDA_RASTERIZER_STATE_H_INCLUDED

#include <cstddef>
#include <cstdint>

// P1-7+ (OPTIMIZATIONS.md §13.6): opaque int64 handle to a C++ RasterizerState.
// 0 = no state (legacy path). Lifetime is managed by the Python side via
// make_rasterizer_state() and free_rasterizer_state() in ext.cpp.
using RasterizerStateHandle = int64_t;

namespace CudaRasterizer
{
	class RasterizerState
	{
	public:
		RasterizerState();
		~RasterizerState();

		// Disable copy (would double-free CUDA memory). Move is fine.
		RasterizerState(const RasterizerState&) = delete;
		RasterizerState& operator=(const RasterizerState&) = delete;
		RasterizerState(RasterizerState&& other) noexcept;
		RasterizerState& operator=(RasterizerState&& other) noexcept;

		// Grow the error buffer to hold at least P floats. No-op if capacity
		// already >= P. Never shrinks. Throws std::bad_alloc on CUDA OOM.
		void resize_if_needed(int P);

		// Zero the error buffer asynchronously on the CURRENT stream.
		// MUST be called after the last consumer (MCMC sampling) finishes
		// reading and before the next post-blend kernel writes. Uses
		// at::cuda::getCurrentCUDAStream() so it orders correctly with the
		// main training forward/backward stream.
		void zero_current_stream();

		// Raw device pointer (size = capacity()). Returned ptr is valid
		// until next resize_if_needed() that grows or until destruction.
		float* error_buffer_ptr() const { return buffer_; }

		// Current allocation count (NOT P at last forward — capacity may be
		// larger after a grow).
		int capacity() const { return capacity_; }

	private:
		float* buffer_ = nullptr;
		int capacity_ = 0;
	};
}

#endif