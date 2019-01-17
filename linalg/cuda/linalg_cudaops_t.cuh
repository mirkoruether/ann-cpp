#ifndef LINALG_CUDAOPS_T_CUH
#define LINALG_CUDAOPS_T_CUH
#include "mat_arr.h"
#include "cuda_util.cuh"

using namespace linalg;

namespace linalg { namespace cuda
{
	template <typename Fc>
	mat_arr cuda_element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C,
	                                          const Fc& f, mat_tr tr = transpose_no);

	template <typename Fc>
	mat_arr cuda_element_wise_operation(const mat_arr& A, mat_arr* C, const Fc& f);

	template <typename Fc>
	mat_arr create_c(mat_arr* C, unsigned count, unsigned rows, unsigned cols, Fc f)
	{
		if (C == nullptr)
		{
			mat_arr temp_C(count, rows, cols);
			f(&temp_C);
			return temp_C;
		}

		if (C->count != count || C->rows != rows || C->cols != cols)
		{
			throw std::runtime_error("C has wrong dimensions");
		}

		f(C);
		return *C;
	}

	template <typename Fc>
	__global__
	void _e_by_e_operation_cubic_kernel(const float* a, const float* b,
	                                    bool a_isarray, bool b_isarray,
	                                    bool transpose_a, bool transpose_b,
	                                    float* c, dim3 size, Fc f)
	{
		const dim3 pos = current_pos_cubic();
		if (check_pos_cubic(pos, size))
		{
			const unsigned mat_offset = pos.x * size.y * size.z;
			const unsigned i = pos.y * size.z + pos.z;
			const unsigned i_t = pos.z * size.z + pos.y;

			const unsigned a_index = transpose_a ? i_t : i;
			const unsigned b_index = transpose_b ? i_t : i;

			c[mat_offset + i] = f(a[a_isarray ? mat_offset + a_index : a_index],
			                      b[b_isarray ? mat_offset + b_index : b_index]);
		}
	}

	template <typename Fc>
	__global__
	void _e_by_e_operation_quadratic_kernel(const float* a, const float* b,
	                                        bool a_isarray, bool b_isarray,
	                                        float* c, dim3 size, Fc f)
	{
		const dim3 pos = current_pos_quadratic();
		if (check_pos_quadratic(pos, size))
		{
			const unsigned mat_offset = pos.x * size.y;
			const unsigned index = pos.y;
			c[mat_offset + index] = f(a[a_isarray ? mat_offset + index : index],
			                          b[b_isarray ? mat_offset + index : index]);
		}
	}

	template <typename Fc>
	__global__
	void _e_by_e_operation_linear_kernel(const float* a, const float* b,
	                                     float* c, unsigned size, Fc f)
	{
		const unsigned pos = current_pos_linear();
		if (pos < size)
		{
			c[pos] = f(a[pos], b[pos]);
		}
	}

	inline void _e_by_e_size_check(const unsigned count_a, const unsigned rows_a, const unsigned cols_a,
	                               const unsigned count_b, const unsigned rows_b, const unsigned cols_b,
	                               const unsigned count_c, const unsigned rows_c, const unsigned cols_c)
	{
		if (count_a != 1 && count_b != 1 && count_a != count_b)
		{
			throw std::runtime_error("Wrong input array sizes");
		}

		if (count_c != std::max(count_a, count_b))
		{
			throw std::runtime_error("Wrong output array sizes");
		}

		if (rows_a != rows_b || rows_b != rows_c)
		{
			throw std::runtime_error("Row count does not fit");
		}

		if (cols_a != cols_b || cols_b != cols_c)
		{
			throw std::runtime_error("Column count does not fit");
		}
	}

	template <typename Fc>
	void _cuda_element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C,
	                                        const Fc& f, mat_tr tr)
	{
		const bool transpose_a = tr == transpose_A || tr == transpose_both;
		const bool transpose_b = tr == transpose_B || tr == transpose_both;

		_e_by_e_size_check(A.count, transpose_a ? A.cols : A.rows, transpose_a ? A.rows : A.cols,
		                   B.count, transpose_b ? B.cols : B.rows, transpose_b ? B.rows : B.cols,
		                   C->count, C->rows, C->cols);

		const bool a_isarray = A.count > 1;
		const bool b_isarray = B.count > 1;

		if (tr == transpose_no)
		{
			if (!a_isarray && !b_isarray)
			{
				prepare_launch_linear(*C, [&](unsigned size, unsigned threads, unsigned blocks)
				{
					_e_by_e_operation_linear_kernel<Fc>
						<< <blocks, threads >> >(A.dev_start(), B.dev_start(),
						                         C->dev_start(), size, f);
				});
			}
			else
			{
				prepare_launch_quadratic(*C, [&](dim3 size, dim3 threads, dim3 blocks)
				{
					_e_by_e_operation_quadratic_kernel<Fc>
						<< <blocks, threads >> >(A.dev_start(), B.dev_start(),
						                         a_isarray, b_isarray,
						                         C->dev_start(), size, f);
				});
			}
		}
		else
		{
			prepare_launch_cubic(*C, [=](dim3 size, dim3 threads, dim3 blocks)
			{
				_e_by_e_operation_cubic_kernel<Fc>
					<< < blocks, threads >> >(A.dev_start(), B.dev_start(),
					                          a_isarray, b_isarray,
					                          transpose_a, transpose_b,
					                          C->dev_start(), size, f);
			});
		}
	}

	template <typename Fc>
	mat_arr cuda_element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C,
	                                          const Fc& f, mat_tr tr)
	{
		const bool tr_A = (tr == transpose_A || tr == transpose_both);
		return create_c(C, std::max(A.count, B.count),
		                tr_A ? A.cols : A.rows,
		                tr_A ? A.rows : A.cols,
		                [=](mat_arr* C_nonnull)
		                {
			                _cuda_element_by_element_operation(A, B, C_nonnull, f, tr);
		                });
	}

	template <typename Fc>
	__global__ void _element_wise_operation_kernel(const float* a, float* c, unsigned size, const Fc& f)
	{
		const unsigned pos = current_pos_linear();
		if (pos < size)
		{
			c[pos] = f(a[pos]);
		}
	}

	template <typename Fc>
	mat_arr cuda_element_wise_operation(const mat_arr& A, mat_arr* C, const Fc& f)
	{
		return create_c(C, A.count, A.rows, A.cols,
		                [=](mat_arr* C_nonnull)
		                {
			                prepare_launch_linear(*C_nonnull, [&](unsigned size, unsigned threads, unsigned blocks)
			                {
				                _element_wise_operation_kernel
					                << <blocks, threads >> >(A.dev_start(),
					                                         C_nonnull->dev_start(),
					                                         size, f);
			                });
		                });
	}
}}
#endif
