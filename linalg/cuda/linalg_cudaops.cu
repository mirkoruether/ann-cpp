#include "linalg_cudaops.cuh"
#include "linalg_cudaops_t.cuh"
#include "cuda_util.cuh"

using namespace linalg::cuda;

namespace linalg { namespace cuda
{
	template <typename Fc, bool add>
	__global__
	void _matrix_mul_kernel(const float* a, const float* b,
	                        bool a_isarray, bool b_isarray,
	                        float* c, unsigned m, dim3 size, Fc f)
	{
		const dim3 pos = current_pos_cubic();
		if (check_pos_cubic(pos, size))
		{
			const unsigned l = size.y;
			const unsigned n = size.z;
			const float* a_mat = a_isarray ? a + (pos.x * l * m) : a;
			const float* b_mat = b_isarray ? b + (pos.x * m * n) : b;

			const float result = f(a_mat, b_mat, pos.y, pos.z, l, m, n);
			if (add)
			{
				c[index_cubic(pos, size)] += result;
			}
			else
			{
				c[index_cubic(pos, size)] = result;
			}
		}
	}

	template <typename Fc, bool add>
	void _matrix_mul_launch(const mat_arr& A, const mat_arr& B, mat_arr* C, Fc f)
	{
		const unsigned m = A.cols;
		if (B.rows != m)
		{
			throw std::runtime_error("matrices cannot be multiplied");
		}

		prepare_launch_cubic(*C, [=](dim3 size, dim3 threads, dim3 blocks)
		{
			_matrix_mul_kernel<Fc, add>
				<< < blocks, threads >> >(A.dev_start(),
				                          B.dev_start(),
				                          A.count > 1, B.count > 1,
				                          C->dev_start(), m, size, f);
		});
	}

	struct mat_mul_case0_helper_struct
	{
		__device__

		float operator()(const float* a_mat, const float* b_mat,
		                 const unsigned i, const unsigned k,
		                 const unsigned l, const unsigned m, const unsigned n) const
		{
			return mat_mul_case0_helper(a_mat, b_mat, i, k, l, m, n);
		}
	};

	struct mat_mul_case1_helper_struct
	{
		__device__

		float operator()(const float* a_mat, const float* b_mat,
		                 const unsigned i, const unsigned k,
		                 const unsigned l, const unsigned m, const unsigned n) const
		{
			return mat_mul_case1_helper(a_mat, b_mat, i, k, l, m, n);
		}
	};

	struct mat_mul_case2_helper_struct
	{
		__device__

		float operator()(const float* a_mat, const float* b_mat,
		                 const unsigned i, const unsigned k,
		                 const unsigned l, const unsigned m, const unsigned n) const
		{
			return mat_mul_case2_helper(a_mat, b_mat, i, k, l, m, n);
		}
	};

	struct mat_mul_case3_helper_struct
	{
		__device__

		float operator()(const float* a_mat, const float* b_mat,
		                 const unsigned i, const unsigned k,
		                 const unsigned l, const unsigned m, const unsigned n) const
		{
			return mat_mul_case3_helper(a_mat, b_mat, i, k, l, m, n);
		}
	};

	template <bool add>
	mat_arr _matrix_mul(const mat_arr& A, const mat_arr& B, mat_arr* C, mat_tr tr)
	{
		const bool tr_A = (tr == transpose_A || tr == transpose_both);
		const bool tr_B = (tr == transpose_B || tr == transpose_both);
		return create_c(C, std::max(A.count, B.count),
		                tr_A ? A.cols : A.rows,
		                tr_B ? B.rows : B.cols,
		                [=](mat_arr* C_nonnull)
		                {
			                switch (tr)
			                {
			                case transpose_no:
				                _matrix_mul_launch<mat_mul_case0_helper_struct, add>(A, B, C_nonnull,
				                                                                     mat_mul_case0_helper_struct());
				                break;
			                case transpose_A:
				                _matrix_mul_launch<mat_mul_case1_helper_struct, add>(A, B, C_nonnull,
				                                                                     mat_mul_case1_helper_struct());
				                break;
			                case transpose_B:
				                _matrix_mul_launch<mat_mul_case2_helper_struct, add>(A, B, C_nonnull,
				                                                                     mat_mul_case2_helper_struct());
				                break;
			                case transpose_both:
				                _matrix_mul_launch<mat_mul_case3_helper_struct, add>(A, B, C_nonnull,
				                                                                     mat_mul_case3_helper_struct());
				                break;
			                }
		                });
	}

	mat_arr cuda_matrix_mul(const mat_arr& A, const mat_arr& B, mat_arr* C, mat_tr tr)
	{
		return _matrix_mul<false>(A, B, C, tr);
	}

	mat_arr cuda_matrix_mul_add(const mat_arr& A, const mat_arr& B, mat_arr* C, mat_tr tr)
	{
		return _matrix_mul<true>(A, B, C, tr);
	}

	struct _cuda_e_by_e_add_kernel
	{
		__device__ float operator()(float a, float b) const
		{
			return a + b;
		}
	};

	mat_arr cuda_element_wise_add(const mat_arr& A, const mat_arr& B, mat_arr* C, mat_tr tr)
	{
		return cuda_element_by_element_operation(A, B, C, _cuda_e_by_e_add_kernel(), tr);
	}

	struct _cuda_e_by_e_sub_kernel
	{
		__device__ float operator()(float a, float b) const
		{
			return a - b;
		}
	};

	mat_arr cuda_element_wise_sub(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		return cuda_element_by_element_operation(A, B, C, _cuda_e_by_e_sub_kernel(), tr);
	}

	struct _cuda_e_by_e_mul_kernel
	{
		__device__ float operator()(float a, float b) const
		{
			return a * b;
		}
	};

	mat_arr cuda_element_wise_mul(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		return cuda_element_by_element_operation(A, B, C, _cuda_e_by_e_mul_kernel(), tr);
	}

	struct _cuda_e_by_e_div_kernel
	{
		__device__ float operator()(float a, float b) const
		{
			return a / b;
		}
	};

	mat_arr cuda_element_wise_div(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		return cuda_element_by_element_operation(A, B, C, _cuda_e_by_e_div_kernel(), tr);
	}
}}
