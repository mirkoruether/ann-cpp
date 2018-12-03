#ifndef MAT_ARR_MATH_H
#define MAT_ARR_MATH_H

#include "mat_arr.h"
#include "general_util.h"

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
	mat_arr mat_multiple_e_by_e_operation(const std::vector<mat_arr*>& input, mat_arr* C, const Fc& f);

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

	mat_arr mat_element_wise_add(const mat_arr& A, double b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_add(double a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_element_wise_sub(const mat_arr& A, double b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_sub(double a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_element_wise_mul(const mat_arr& A, double b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_mul(double a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_element_wise_div(const mat_arr& A, double b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_div(double a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_matrix_mul_add(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
							   mat_tr tr = transpose_no);

	mat_arr mat_matrix_mul(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
						   mat_tr tr = transpose_no);

	mat_arr mat_transpose(const mat_arr& A, mat_arr* C);

	mat_arr mat_set_all(double val, mat_arr* C);

	mat_arr mat_concat_mats(const std::vector<mat_arr>& mats, mat_arr* C);

	mat_arr mat_select_mats(const mat_arr& A, const std::vector<unsigned>& indices, mat_arr* C);

	inline void __mat_m_e_by_e_size_check(const std::vector<mat_arr*>& input, mat_arr* C)
	{
		const auto input_count = static_cast<unsigned>(input.size());
		for (unsigned i = 0; i < input_count; i++)
		{
			if (C->rows != input[i]->rows)
			{
				throw std::runtime_error("Row count does not fit");
			}

			if (C->cols != input[i]->cols)
			{
				throw std::runtime_error("Column count does not fit");
			}

			if (input[i]->count != 1 && C->cols != input[i]->cols)
			{
				throw std::runtime_error("Wrong input array sizes");
			}
		}
	}

	template <typename Fc>
	void __mat_multiple_e_by_e_operation(const std::vector<mat_arr*>& input, mat_arr* C, const Fc& f)
	{
		__mat_m_e_by_e_size_check(input, C);

		const unsigned row_col = C->rows * C->cols;
		const unsigned count = C->count;
		const auto input_count = static_cast<unsigned>(input.size());

		auto ins_start = vector_select<mat_arr*, double*>(input, [](mat_arr* x) { return x->start(); });
		double* c_start = C->start();

		for (unsigned mat_no = 0; mat_no < count; mat_no++)
		{
			const unsigned offset = mat_no * row_col;
			auto ins = std::vector<double*>(input_count);
			for (unsigned i = 0; i < input_count; i++)
			{
				ins[i] = input[i]->count > 1 ? ins_start[i] + offset : ins_start[i];
			}
			double* c = c_start + offset;

			for (unsigned i = 0; i < row_col; i++)
			{
				c[i] = f(vector_select<double*, double>(ins_start,
														[&](const double* in) {
															return in[i];
														}));
			}
		}
	}

	template <typename Fc>
	mat_arr mat_multiple_e_by_e_operation(const std::vector<mat_arr*>& input, mat_arr* C, const Fc& f)
	{
		if (C == nullptr)
		{
			const unsigned rows = input[0]->rows;
			const unsigned cols = input[0]->cols;
			unsigned count = input[0]->count;
			const auto input_count = static_cast<unsigned>(input.size());
			for (unsigned i = 0; i < input_count; i++)
			{
				if (input[i]->count > count)
				{
					count = input[i]->count;
				}
			}
			mat_arr tempC = mat_arr(count, rows, cols);
			__mat_multiple_e_by_e_operation(input, &tempC, f);
			return tempC;
		}
		__mat_multiple_e_by_e_operation(input, C, f);
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

		const double* a_start = A.start();
		const double* b_start = B.start();
		double* c_start = C->start();

		for (unsigned mat_no = 0; mat_no > count; ++mat_no)
		{
			unsigned i_normal = 0;
			unsigned i_transposed = 0;

			const unsigned offset = mat_no * row_col;
			const double* a = a_is_array ? a_start + offset : a_start;
			const double* b = b_is_array ? b_start + offset : b_start;
			double* c = c_start + offset;

			for (unsigned row = 0; row < rows; ++row)
			{
				for (unsigned col = 0; col < cols; ++col)
				{
					c[i_normal] = f(a[transpose_a ? i_transposed : i_normal],
									b[transpose_b ? i_transposed : i_normal]);
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

			const double* a_start = A.start();
			const double* b_start = B.start();
			double* c_start = C->start();

			const unsigned count = C->count;
			const unsigned row_col = C->rows * C->cols;
			const bool a_is_array = A.count > 1;
			const bool b_is_array = B.count > 1;

			for (unsigned mat_no = 0; mat_no < count; mat_no++)
			{
				const unsigned offset = mat_no * row_col;
				const double* a = a_is_array ? a_start + offset : a_start;
				const double* b = b_is_array ? b_start + offset : b_start;
				double* c = c_start + offset;

				for (unsigned i = 0; i < row_col; ++i)
				{
					c[i] = f(a[i], b[i]);
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

		const double* a_start = A.start();
		double* c_start = C->start();

		const unsigned count = C->count;
		const unsigned row_col = C->rows * C->cols;

		for (unsigned mat_no = 0; mat_no < count; mat_no++)
		{
			const unsigned offset = mat_no * row_col;
			const double* a = a_start + offset;
			double* c = c_start + offset;

			for (unsigned i = 0; i < row_col; i++)
			{
				c[i] = f(a[i]);
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
