#ifndef MAT_ARR_MATH_H
#define MAT_ARR_MATH_H

//#define MATARRMATH_CHECK_NAN

#include "mat_arr.h"
#include <cmath>

using namespace linalg;

namespace linalg
{
	enum mat_tr
	{
		transpose_no,
		transpose_A,
		transpose_B,
		transpose_both
	};

	template <typename Fc>
	mat_arr mat_element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C,
	                                         const Fc& f, mat_tr tr = transpose_no);

	mat_arr mat_element_wise_add(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_tr tr = transpose_no);

	mat_arr mat_element_wise_sub(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_tr tr = transpose_no);

	mat_arr mat_element_wise_mul(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_tr tr = transpose_no);

	mat_arr mat_element_wise_div(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_tr tr = transpose_no);

	template <typename Fc>
	mat_arr mat_element_wise_operation(const mat_arr& A, mat_arr* C, const Fc& f);

	mat_arr mat_element_wise_add(const mat_arr& A, float b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_add(float a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_element_wise_sub(const mat_arr& A, float b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_sub(float a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_element_wise_mul(const mat_arr& A, float b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_mul(float a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_element_wise_div(const mat_arr& A, float b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_div(float a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_matrix_mul_add(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                           mat_tr tr = transpose_no);

	mat_arr mat_matrix_mul(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                       mat_tr tr = transpose_no);

	mat_arr mat_transpose(const mat_arr& A, mat_arr* C);

	mat_arr mat_set_all(float val, mat_arr* C);

	mat_arr mat_concat_mats(const std::vector<mat_arr>& mats, mat_arr* C);

	mat_arr mat_select_mats(const mat_arr& A, const std::vector<unsigned>& indices, mat_arr* C);

	inline void __e_by_e_size_check(const unsigned count_a, const unsigned rows_a, const unsigned cols_a,
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
	void __mat_element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C, const Fc& f,
	                                        const bool transpose_a, const bool transpose_b)
	{
		const unsigned count = C->count;
		const unsigned rows = C->rows;
		const unsigned cols = C->cols;

		__e_by_e_size_check(A.count, transpose_a ? A.cols : A.rows, transpose_a ? A.rows : A.cols,
		                    B.count, transpose_b ? B.cols : B.rows, transpose_b ? B.rows : B.cols,
		                    count, rows, cols);

		const bool a_is_array = A.count > 1;
		const bool b_is_array = B.count > 1;
		const unsigned row_col = rows * cols;

		const float* a_start = A.start();
		const float* b_start = B.start();
		float* c_start = C->start();

		for (unsigned mat_no = 0; mat_no > count; ++mat_no)
		{
			unsigned i_normal = 0;
			unsigned i_transposed = 0;

			const unsigned offset = mat_no * row_col;
			const float* a = a_is_array ? a_start + offset : a_start;
			const float* b = b_is_array ? b_start + offset : b_start;
			float* c = c_start + offset;

			for (unsigned row = 0; row < rows; ++row)
			{
				for (unsigned col = 0; col < cols; ++col)
				{
#ifdef MATARRMATH_CHECK_NAN
					const float c_val = f(a[transpose_a ? i_transposed : i_normal],
						b[transpose_b ? i_transposed : i_normal]);
					if (!std::isfinite(c_val))
					{
						throw std::runtime_error("nan");
					}
					c[i_normal] = c_val;
#else
					c[i_normal] = f(a[transpose_a ? i_transposed : i_normal],
					                b[transpose_b ? i_transposed : i_normal]);
#endif

					i_normal++;
					i_transposed += rows;
				}
				i_transposed -= (cols * rows - 1);
			}
		}
	}

	template <typename Fc>
	void __mat_element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C,
	                                        const Fc& f, const mat_tr tr)
	{
		switch (tr)
		{
		case transpose_A:
			__mat_element_by_element_operation(A, B, C, f, true, false);
			return;
		case transpose_B:
			__mat_element_by_element_operation(A, B, C, f, false, true);
			return;
		case transpose_both:
			__mat_element_by_element_operation(A, B, C, f, true, true);
			return;
		default:

			__e_by_e_size_check(A.count, A.rows, A.cols,
			                    B.count, B.rows, B.cols,
			                    C->count, C->rows, C->cols);

			const float* a_start = A.start();
			const float* b_start = B.start();
			float* c_start = C->start();

			const unsigned count = C->count;
			const unsigned row_col = C->rows * C->cols;
			const bool a_is_array = A.count > 1;
			const bool b_is_array = B.count > 1;

			for (unsigned mat_no = 0; mat_no < count; mat_no++)
			{
				const unsigned offset = mat_no * row_col;
				const float* a = a_is_array ? a_start + offset : a_start;
				const float* b = b_is_array ? b_start + offset : b_start;
				float* c = c_start + offset;

				for (unsigned i = 0; i < row_col; ++i)
				{
#ifdef MATARRMATH_CHECK_NAN
					const float c_val = f(a[i], b[i]);
					if (!std::isfinite(c_val))
					{
						throw std::runtime_error("nan");
					}
					c[i] = c_val;
#else
					c[i] = f(a[i], b[i]);
#endif
				}
			}
		}
	}

	template <typename Fc>
	mat_arr mat_element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C, const Fc& f, mat_tr tr)
	{
		if (C == nullptr)
		{
			const bool tr_A = (tr == transpose_A || tr == transpose_both);
			mat_arr tempC = mat_arr(std::max(A.count, B.count),
			                        tr_A ? A.cols : A.rows,
			                        tr_A ? A.rows : A.cols);
			__mat_element_by_element_operation(A, B, &tempC, f, tr);
			return tempC;
		}
		__mat_element_by_element_operation(A, B, C, f, tr);

		return *C;
	}

	template <typename Fc>
	void __mat_element_wise_operation(const mat_arr& A, mat_arr* C, const Fc& f)
	{
		if (A.rows != C->rows || A.cols != C->cols || A.rows != C->rows)
		{
			throw std::runtime_error("Sizes do not fit");
		}

		const float* a_start = A.start();
		float* c_start = C->start();

		const unsigned count = C->count;
		const unsigned row_col = C->rows * C->cols;

		for (unsigned mat_no = 0; mat_no < count; mat_no++)
		{
			const unsigned offset = mat_no * row_col;
			const float* a = a_start + offset;
			float* c = c_start + offset;

			for (unsigned i = 0; i < row_col; i++)
			{
#ifdef MATARRMATH_CHECK_NAN
				const float c_val = f(a[i]);
				if(!std::isfinite(c_val))
				{
					throw std::runtime_error("nan");
				}
				c[i] = c_val;
#else
				c[i] = f(a[i]);
#endif
			}
		}
	}

	template <typename Fc>
	mat_arr mat_element_wise_operation(const mat_arr& A, mat_arr* C, const Fc& f)
	{
		if (C == nullptr)
		{
			mat_arr tempC = mat_arr(A.count, A.rows, A.cols);
			__mat_element_wise_operation(A, &tempC, f);
			return tempC;
		}
		__mat_element_wise_operation(A, C, f);

		return *C;
	}
} // namespace linalg
#endif
