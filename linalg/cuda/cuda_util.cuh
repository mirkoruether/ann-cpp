#ifndef LINALG_CUDA_UTIL_CUH
#define LINALG_CUDA_UTIL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <functional>
#include "mat_arr.h"

namespace linalg { namespace cuda
{
	inline void cuda_sync()
	{
		const cudaError_t err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			throw std::runtime_error("error. Code: " + err);
		}
	}

	__device__ inline float mat_mul_case0_helper(const float* a_mat, const float* b_mat,
	                                             const unsigned i, const unsigned k,
	                                             const unsigned l, const unsigned m, const unsigned n)
	{
		float temp_sum = 0.0;
		const float* a_row = a_mat + i * m;
		const float* b_col = b_mat + k;
		for (unsigned j = 0; j < m; j++)
		{
			temp_sum += a_row[j] * b_col[j * n];
		}
		return temp_sum;
	}

	__device__ inline float mat_mul_case1_helper(const float* a_mat, const float* b_mat,
	                                             const unsigned i, const unsigned k,
	                                             const unsigned l, const unsigned m, const unsigned n)
	{
		float temp_sum = 0.0;
		const float* a_col = a_mat + i;
		const float* b_col = b_mat + k;
		for (unsigned j = 0; j < m; j++)
		{
			temp_sum += a_col[j * l] * b_col[j * n];
		}
		return temp_sum;
	}

	__device__ inline float mat_mul_case2_helper(const float* a_mat, const float* b_mat,
	                                             const unsigned i, const unsigned k,
	                                             const unsigned l, const unsigned m, const unsigned n)
	{
		float temp_sum = 0.0;
		const float* a_row = a_mat + i * m;
		const float* b_row = b_mat + k * m;
		for (unsigned j = 0; j < m; j++)
		{
			temp_sum += a_row[j] * b_row[j];
		}
		return temp_sum;
	}

	__device__ inline float mat_mul_case3_helper(const float* a_mat, const float* b_mat,
	                                             const unsigned i, const unsigned k,
	                                             const unsigned l, const unsigned m, const unsigned n)
	{
		float temp_sum = 0.0;
		const float* a_col = a_mat + i;
		const float* b_row = b_mat + k * m;
		for (unsigned j = 0; j < m; j++)
		{
			temp_sum += a_col[j * l] * b_row[j];
		}
		return temp_sum;
	}

	__device__ inline dim3 current_pos_cubic()
	{
		return dim3(blockIdx.x * blockDim.x + threadIdx.x,
		            blockIdx.y * blockDim.y + threadIdx.y,
		            blockIdx.z * blockDim.z + threadIdx.z);
	}

	__device__ inline dim3 current_pos_quadratic()
	{
		return dim3(blockIdx.x * blockDim.x + threadIdx.x,
		            blockIdx.y * blockDim.y + threadIdx.y,
		            0);
	}

	__device__ inline unsigned current_pos_linear()
	{
		return blockIdx.x * blockDim.x + threadIdx.x;
	}

	__device__ inline unsigned index_cubic(dim3 pos, dim3 size)
	{
		return pos.x * size.y * size.z + pos.y * size.z + pos.z;
	}

	__device__ inline unsigned index_cubic_transposed(dim3 pos, dim3 size)
	{
		return pos.x * size.y * size.z + pos.z * size.z + pos.y;
	}

	__device__ inline unsigned index_quadratic(dim3 pos, dim3 size)
	{
		return pos.x * size.y + pos.y;
	}

	__device__ inline bool check_pos_cubic(dim3 pos, dim3 size)
	{
		return pos.x < size.x && pos.y < size.y && pos.z < size.z;
	}

	__device__ inline bool check_pos_quadratic(dim3 pos, dim3 size)
	{
		return pos.x < size.x && pos.y < size.y;
	}

	inline void prepare_launch_cubic(const linalg::mat_arr& target,
	                                 std::function<void(dim3, dim3, dim3)> kernel_launch)
	{
		const dim3 size(target.count, target.rows, target.cols);
		const dim3 threads_per_block(std::min(16u, size.x), std::min(8u, size.y), std::min(8u, size.z));
		const dim3 blocks_per_grid(unsigned(ceil(double(size.x) / double(threads_per_block.x))),
		                           unsigned(ceil(double(size.y) / double(threads_per_block.y))),
		                           unsigned(ceil(double(size.z) / double(threads_per_block.z))));

		kernel_launch(std::move(size), std::move(threads_per_block), std::move(blocks_per_grid));

#ifdef LINALG_AUTOSYNC
		cuda_sync();
#endif
	}

	inline void prepare_launch_quadratic(const linalg::mat_arr& target,
	                                     std::function<void(dim3, dim3, dim3)> kernel_launch)
	{
		const dim3 size(target.count, target.rows * target.cols, 1);
		const dim3 threads_per_block(std::min(16u, size.x), std::min(64u, size.y), 1);
		const dim3 blocks_per_grid(unsigned(ceil(double(size.x) / double(threads_per_block.x))),
		                           unsigned(ceil(double(size.y) / double(threads_per_block.y))),
		                           1);

		kernel_launch(std::move(size), std::move(threads_per_block), std::move(blocks_per_grid));

#ifdef LINALG_AUTOSYNC
		cuda_sync();
#endif
	}

	inline void prepare_launch_linear(const linalg::mat_arr& target,
	                                  std::function<void(unsigned, unsigned, unsigned)> kernel_launch)
	{
		const unsigned size = target.size();
		const unsigned threads_per_block = std::min(1024u, size);
		const unsigned blocks_per_grid = unsigned(ceil(double(size) / double(threads_per_block)));

		kernel_launch(std::move(size), std::move(threads_per_block), std::move(blocks_per_grid));

#ifdef LINALG_AUTOSYNC
		cuda_sync();
#endif
	}
}}
#endif
