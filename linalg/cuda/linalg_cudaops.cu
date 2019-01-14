#include "linalg_cudaops.cuh"
#include "cuda_util.cuh"

using namespace linalg::cuda;

namespace linalg
{
	namespace cuda
	{
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
			if(B.rows != m)
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
			return create_c(C, std::max(A.count, B.count), A.rows, B.cols, [=](mat_arr* C_nonnull)
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
	}
}
