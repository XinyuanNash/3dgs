/*
 * P1-7+ RasterizerState implementation.
 * See rasterizer_state.h for lifetime contract.
 */
#include "rasterizer_state.h"
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <stdexcept>
#include <cstdio>

namespace CudaRasterizer
{
	RasterizerState::RasterizerState() = default;

	RasterizerState::~RasterizerState()
	{
		if (buffer_ != nullptr)
		{
			cudaError_t err = cudaFree(buffer_);
			if (err != cudaSuccess)
			{
				// Don't throw from dtor; log only.
				std::fprintf(stderr,
					"[RasterizerState] cudaFree failed: %s\n",
					cudaGetErrorString(err));
			}
			buffer_ = nullptr;
			capacity_ = 0;
		}
	}

	RasterizerState::RasterizerState(RasterizerState&& other) noexcept
		: buffer_(other.buffer_), capacity_(other.capacity_)
	{
		other.buffer_ = nullptr;
		other.capacity_ = 0;
	}

	RasterizerState& RasterizerState::operator=(RasterizerState&& other) noexcept
	{
		if (this != &other)
		{
			if (buffer_ != nullptr)
			{
				cudaFree(buffer_);
			}
			buffer_ = other.buffer_;
			capacity_ = other.capacity_;
			other.buffer_ = nullptr;
			other.capacity_ = 0;
		}
		return *this;
	}

	void RasterizerState::resize_if_needed(int P)
	{
		if (P <= capacity_)
		{
			return;  // already large enough (typical case after first grow)
		}

		// Free the old buffer if any. cudaFree(nullptr) is a no-op so the
		// initial path (buffer_ == nullptr) is safe.
		if (buffer_ != nullptr)
		{
			cudaError_t err = cudaFree(buffer_);
			if (err != cudaSuccess)
			{
				throw std::runtime_error(
					std::string("[RasterizerState] cudaFree failed: ") +
					cudaGetErrorString(err));
			}
			buffer_ = nullptr;
			capacity_ = 0;
		}

		// Allocate the new buffer on the CURRENT device. P can be up to
		// ~1M floats (~4 MB) at 18K hard gate — well within budget, but
		// guard for OOM.
		float* new_buf = nullptr;
		cudaError_t err = cudaMalloc(
			reinterpret_cast<void**>(&new_buf),
			static_cast<size_t>(P) * sizeof(float));
		if (err != cudaSuccess)
		{
			throw std::bad_alloc();
		}

		buffer_ = new_buf;
		capacity_ = P;
	}

	void RasterizerState::zero_current_stream()
	{
		if (buffer_ == nullptr || capacity_ == 0)
		{
			return;
		}
		cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
		cudaError_t err = cudaMemsetAsync(
			buffer_, 0,
			static_cast<size_t>(capacity_) * sizeof(float),
			stream);
		if (err != cudaSuccess)
		{
			throw std::runtime_error(
				std::string("[RasterizerState] cudaMemsetAsync failed: ") +
				cudaGetErrorString(err));
		}
	}
}