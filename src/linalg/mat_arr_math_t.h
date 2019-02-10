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

	enum _input_transpose_type
	{
		normal,
		transposed,
		scalar
	};

	template <_input_transpose_type t_a, _input_transpose_type t_b>
	bool __e_by_e_check_dim(unsigned dim_a, unsigned dim_a_tr, unsigned dim_b, unsigned dim_b_tr, unsigned dim_c)
	{
		if (t_a == scalar)
		{
			if (t_b == scalar)
			{
				return dim_c == 1;
			}
			else
			{
				return dim_c == t_b == transposed ? dim_b_tr : dim_b;
			}
		}
		else
		{
			if (t_b == scalar)
			{
				return dim_c == t_a == transposed ? dim_a_tr : dim_a;
			}
			else
			{
				const unsigned real_dim_a = t_a == transposed ? dim_a_tr : dim_a;
				const unsigned real_dim_b = t_b == transposed ? dim_b_tr : dim_b;
				return real_dim_a == real_dim_b && dim_c == real_dim_a;
			}
		}
	}

	template <_input_transpose_type t_a, _input_transpose_type t_b>
	void __e_by_e_size_check(const mat_arr& A, const mat_arr& B, const mat_arr& C)
	{
		if (A.count != 1 && B.count != 1 && A.count != B.count)
		{
			throw std::runtime_error("Wrong input array sizes");
		}

		if (C.count != std::max(A.count, B.count))
		{
			throw std::runtime_error("Wrong output array sizes");
		}

		if (!__e_by_e_check_dim<t_a, t_b>(A.rows, A.cols, B.rows, B.cols, C.rows))
		{
			throw std::runtime_error("Row count does not fit");
		}

		if (!__e_by_e_check_dim<t_a, t_b>(A.cols, A.rows, B.cols, B.rows, C.cols))
		{
			throw std::runtime_error("Column count does not fit");
		}
	}

	template <bool tr_loop, typename Ker>
	void __mat_e_by_e_loop(unsigned rows, unsigned cols, Ker k)
	{
		if (tr_loop)
		{
			unsigned i_normal = 0;
			unsigned i_transposed = 0;

			for (unsigned row = 0; row < rows; ++row)
			{
				for (unsigned col = 0; col < cols; ++col)
				{
					k(i_normal, i_transposed);

					i_normal++;
					i_transposed += rows;
				}
				i_transposed -= (cols * rows - 1);
			}
		}
		else
		{
			const unsigned rows_col = rows * cols;
			for (unsigned i = 0; i < rows_col; i++)
			{
				k(i, std::numeric_limits<unsigned>::max());
			}
		}
	}

	template <typename Fc>
	void __mat_e_by_e_set_c(fpt a, fpt b, fpt* c, Fc f)
	{
		const fpt c_val = f(a, b);
#ifdef MATARRMATH_CHECK_NAN
		if (!std::isfinite(c_val))
		{
			throw std::runtime_error("nan");
		}
#endif
		*c = c_val;
	}

	template <_input_transpose_type t>
	const fpt* __mat_e_by_e_get_mat(const fpt* array_start, unsigned mat_no, unsigned offset, bool is_array)
	{
		if (t == scalar)
		{
			return is_array ? array_start + mat_no : array_start;
		}
		else
		{
			return is_array ? array_start + offset : array_start;
		}
	}

	template <typename Fc, _input_transpose_type t_a, _input_transpose_type t_b>
	void __mat_element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C, const Fc& f)
	{
		const unsigned count = C->count;
		const unsigned rows = C->rows;
		const unsigned cols = C->cols;

		__e_by_e_size_check<t_a, t_b>(A, B, *C);

		const bool a_is_array = A.count > 1;
		const bool b_is_array = B.count > 1;
		const unsigned row_col = rows * cols;

		const fpt* a_start = A.start();
		const fpt* b_start = B.start();
		fpt* c_start = C->start();

		for (unsigned mat_no = 0; mat_no < count; ++mat_no)
		{
			const unsigned offset = mat_no * row_col;
			const fpt* a = __mat_e_by_e_get_mat<t_a>(a_start, mat_no, offset, a_is_array);
			const fpt* b = __mat_e_by_e_get_mat<t_b>(b_start, mat_no, offset, b_is_array);
			fpt* c = c_start + offset;

			__mat_e_by_e_loop<t_a == transposed || t_b == transposed>
				(rows, cols, [&](const unsigned& i_normal, const unsigned& i_transposed)
				{
					__mat_e_by_e_set_c(t_a == scalar ? *a : t_a == normal ? a[i_normal] : a[i_transposed],
					                   t_b == scalar ? *b : t_b == normal ? b[i_normal] : b[i_transposed],
					                   c + i_normal, f);
				});
		}
	}

	template <typename Fc>
	mat_arr mat_element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C, const Fc& f, mat_tr tr)
	{
		const bool tr_a = (tr == transpose_A || tr == transpose_both);
		const bool tr_b = (tr == transpose_B || tr == transpose_both);

		const bool sc_a = A.cols == 1 && A.rows == 1;
		const bool sc_b = B.cols == 1 && B.rows == 1;

		const _input_transpose_type t_a = sc_a ? scalar : tr_a ? transposed : normal;
		const _input_transpose_type t_b = sc_b ? scalar : tr_b ? transposed : normal;

		return create_c(C, std::max(A.count, B.count),
		                std::max(tr_a ? A.cols : A.rows, tr_b ? B.cols : B.rows),
		                std::max(tr_a ? A.rows : A.cols, tr_b ? B.rows : B.cols),
		                [&](mat_arr* C_nonnull)
		                {
			                if (t_a == normal && t_b == normal)
				                __mat_element_by_element_operation<Fc, normal, normal>(A, B, C, f);
			                else if (t_a == transposed && t_b == normal)
				                __mat_element_by_element_operation<Fc, transposed, normal>(A, B, C, f);
			                else if (t_a == scalar && t_b == normal)
				                __mat_element_by_element_operation<Fc, scalar, normal>(A, B, C, f);
			                else if (t_a == normal && t_b == transposed)
				                __mat_element_by_element_operation<Fc, normal, transposed>(A, B, C, f);
			                else if (t_a == transposed && t_b == transposed)
				                __mat_element_by_element_operation<Fc, transposed, transposed>(A, B, C, f);
			                else if (t_a == scalar && t_b == transposed)
				                __mat_element_by_element_operation<Fc, scalar, transposed>(A, B, C, f);
			                else if (t_a == normal && t_b == scalar)
				                __mat_element_by_element_operation<Fc, normal, scalar>(A, B, C, f);
			                else if (t_a == transposed && t_b == scalar)
				                __mat_element_by_element_operation<Fc, transposed, scalar>(A, B, C, f);
			                else if (t_a == scalar && t_b == scalar)
				                __mat_element_by_element_operation<Fc, scalar, scalar>(A, B, C, f);
			                else
				                throw std::logic_error("unreachable");
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
