#ifndef MAT_ARR_MATH_T_H
#define MAT_ARR_MATH_T_H

#include <mat_arr.h>

using namespace linalg;

namespace linalg
{
	template <typename Fc>
	mat_arr mat_element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C,
	                                         const Fc& f, mat_tr tr = transpose_no);

	template <typename Fc>
	mat_arr mat_element_wise_operation(const mat_arr& A, mat_arr* C, const Fc& f);

	template <typename Fc>
	mat_arr mat_aggregate(const mat_arr& A, mat_arr* C, fpt init, const Fc& f);

	template <typename Fc>
	fpt mat_total_aggregate(const mat_arr& A, fpt init, const Fc& f);

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

		const fpt* a_start = A.start();
		const fpt* b_start = B.start();
		fpt* c_start = C->start();

		for (unsigned mat_no = 0; mat_no > count; ++mat_no)
		{
			unsigned i_normal = 0;
			unsigned i_transposed = 0;

			const unsigned offset = mat_no * row_col;
			const fpt* a = a_is_array ? a_start + offset : a_start;
			const fpt* b = b_is_array ? b_start + offset : b_start;
			fpt* c = c_start + offset;

			for (unsigned row = 0; row < rows; ++row)
			{
				for (unsigned col = 0; col < cols; ++col)
				{
#ifdef MATARRMATH_CHECK_NAN
					const fpt c_val = f(a[transpose_a ? i_transposed : i_normal],
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

			const fpt* a_start = A.start();
			const fpt* b_start = B.start();
			fpt* c_start = C->start();

			const unsigned count = C->count;
			const unsigned row_col = C->rows * C->cols;
			const bool a_is_array = A.count > 1;
			const bool b_is_array = B.count > 1;

			for (unsigned mat_no = 0; mat_no < count; mat_no++)
			{
				const unsigned offset = mat_no * row_col;
				const fpt* a = a_is_array ? a_start + offset : a_start;
				const fpt* b = b_is_array ? b_start + offset : b_start;
				fpt* c = c_start + offset;

				for (unsigned i = 0; i < row_col; ++i)
				{
#ifdef MATARRMATH_CHECK_NAN
					const fpt c_val = f(a[i], b[i]);
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
		const bool tr_A = (tr == transpose_A || tr == transpose_both);
		return create_c(C, std::max(A.count, B.count),
		                tr_A ? A.cols : A.rows,
		                tr_A ? A.rows : A.cols,
		                [&](mat_arr* C_nonnull)
		                {
			                __mat_element_by_element_operation(A, B, C_nonnull, f, tr);
		                });
	}

	template <typename Fc>
	void __mat_element_wise_operation(const mat_arr& A, mat_arr* C, const Fc& f)
	{
		if (A.rows != C->rows || A.cols != C->cols || A.rows != C->rows)
		{
			throw std::runtime_error("Sizes do not fit");
		}

		const fpt* a_start = A.start();
		fpt* c_start = C->start();

		const unsigned count = C->count;
		const unsigned row_col = C->rows * C->cols;

		for (unsigned mat_no = 0; mat_no < count; mat_no++)
		{
			const unsigned offset = mat_no * row_col;
			const fpt* a = a_start + offset;
			fpt* c = c_start + offset;

			for (unsigned i = 0; i < row_col; i++)
			{
#ifdef MATARRMATH_CHECK_NAN
				const fpt c_val = f(a[i]);
				if (!std::isfinite(c_val))
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
		return create_c(C, A.count, A.rows, A.cols, [&](mat_arr* C_nonnull)
		{
			__mat_element_wise_operation(A, C_nonnull, f);
		});
	}

	template <typename Fc>
	void __mat_aggregate(const mat_arr& A, mat_arr* C, fpt init, const Fc& f)
	{
		const unsigned row_col = A.rows * A.cols;
		const unsigned count = A.count;

		const fpt* a_start = A.start();
		fpt* c_start = C->start();

		for (unsigned mat_no = 0; mat_no < count; mat_no++)
		{
			const fpt* a_mat = a_start + row_col * mat_no;
			fpt agg = init;
			for (unsigned i = 0; i < row_col; i++)
			{
				agg = f(a_mat[i], agg);
			}
			c_start[mat_no] = agg;
		}
	}

	template <typename Fc>
	mat_arr mat_aggregate(const mat_arr& A, mat_arr* C, fpt init, const Fc& f)
	{
		return create_c(C, A.count, 1, 1, [&](mat_arr* C_nonnull)
		{
			__mat_aggregate(A, C_nonnull, init, f);
		});
	}

	template <typename Fc>
	fpt mat_total_aggregate(const mat_arr& A, fpt init, const Fc& f)
	{
		const unsigned size = A.size();
		const fpt* a_start = A.start();
		fpt agg = init;
		for (unsigned i = 0; i < size; i++)
		{
			agg = f(a_start[i], agg);
		}
		return agg;
	}
}

#endif
