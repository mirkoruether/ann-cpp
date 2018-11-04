#define MAT_ARR_MATH_SIZE_CHECK 1

#include "mat_arr_math.h"

using namespace linalg;

namespace linalg
{
#pragma region element_by_element_internal
	inline void __e_by_e_size_check(const unsigned count_a, const unsigned rows_a, const unsigned cols_a,
	                                const unsigned count_b, const unsigned rows_b, const unsigned cols_b,
	                                const unsigned count_c, const unsigned rows_c, const unsigned cols_c)
	{
		if (count_a != 1 && count_b != 1 && count_a != count_b)
		{
			throw runtime_error("Wrong input array sizes");
		}

		if (count_c != max(count_a, count_b))
		{
			throw runtime_error("Wrong output array sizes");
		}

		if (rows_a != rows_b || rows_b != rows_c)
		{
			throw runtime_error("Row count does not fit");
		}

		if (cols_a != cols_b || cols_b != cols_c)
		{
			throw runtime_error("Column count does not fit");
		}
	}

	void __mat_element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C,
	                                        const function<double(double, double)>& f,
	                                        const bool transpose_a, const bool transpose_b)
	{
		const unsigned count = C->count;
		const unsigned rows = C->rows;
		const unsigned cols = C->cols;

#ifdef MAT_ARR_MATH_SIZE_CHECK
		__e_by_e_size_check(A.count, transpose_a ? A.cols : A.rows, transpose_a ? A.rows : A.cols,
		                    B.count, transpose_b ? B.cols : B.rows, transpose_b ? B.rows : B.cols,
		                    count, rows, cols);
#endif

		const double* a = A.start();
		const double* b = B.start();
		double* c = C->start();

		const unsigned size_A = A.size();
		const unsigned size_B = B.size();
		const unsigned rows_count = rows * count;

		unsigned i_normal = 0;
		unsigned i_transposed = 0;
		for (unsigned i = 0; i < rows_count; ++i)
		{
			for (unsigned col = 0; col < cols; ++col)
			{
				*(c + i_normal) = f(*(a + ((transpose_a ? i_transposed : i_normal) % size_A)),
				                    *(b + ((transpose_b ? i_transposed : i_normal) % size_B)));
				i_normal++;
				i_transposed += rows;
			}
			i_transposed -= (cols * rows - 1);
		}
	}

	void __mat_element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C,
	                                        const function<double(double, double)>& f, const mat_tr tr)
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
#ifdef MAT_ARR_MATH_SIZE_CHECK
			__e_by_e_size_check(A.count, A.rows, A.cols,
			                    B.count, B.rows, B.cols,
			                    C->count, C->rows, C->cols);
#endif
			const double* a = A.start();
			const double* b = B.start();
			double* c = C->start();

			const unsigned size_A = A.size();
			const unsigned size_B = B.size();
			const unsigned size_C = C->size();

			for (unsigned i = 0; i < size_C; ++i)
			{
				*(c + i) = f(*(a + (i % size_A)),
				             *(b + (i % size_B)));
			}
		}
	}
#pragma endregion

	mat_arr mat_element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C,
	                                         const function<double(double, double)>& f, mat_tr tr)
	{
		if (C == nullptr)
		{
			const bool tr_A = (tr == transpose_A || tr == transpose_both);
			mat_arr tempC = mat_arr(max(A.count, B.count),
			                        tr_A ? A.cols : A.rows,
			                        tr_A ? A.rows : A.cols);
			__mat_element_by_element_operation(A, B, &tempC, f, tr);
			return tempC;
		}
		__mat_element_by_element_operation(A, B, C, f, tr);
		return *C;
	}

	mat_arr mat_element_wise_add(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		return mat_element_by_element_operation(A, B, C, [](double a, double b) { return a + b; }, tr);
	}

	mat_arr mat_element_wise_sub(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		return mat_element_by_element_operation(A, B, C, [](double a, double b) { return a - b; }, tr);
	}

	mat_arr mat_element_wise_mul(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		return mat_element_by_element_operation(A, B, C, [](double a, double b) { return a * b; }, tr);
	}

	mat_arr mat_element_wise_div(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		return mat_element_by_element_operation(A, B, C, [](double a, double b) { return a / b; }, tr);
	}

#pragma region element_wise_internal
	void __mat_element_wise_operation(const mat_arr& A, mat_arr* C,
	                                  const function<double(double)>& f)
	{
#ifdef MAT_ARR_MATH_SIZE_CHECK
		if (A.rows != C->rows
			|| A.cols != C->cols
			|| A.rows != C->rows)
		{
			throw runtime_error("Sizes do not fit");
		}
#endif

		const double* a = A.start();
		double* c = C->start();

		const unsigned size = C->size();
		for (unsigned i = 0; i < size; i++)
		{
			*(c + i) = f(*(a + i));
		}
	}
#pragma endregion

	mat_arr mat_element_wise_operation(const mat_arr& A, mat_arr* C, const function<double(double)>& f)
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

	mat_arr mat_element_wise_add(const mat_arr& A, double b, mat_arr* C)
	{
		return mat_element_wise_operation(A, C, [b](double a) { return a + b; });
	}

	mat_arr mat_element_wise_add(double a, const mat_arr& B, mat_arr* C)
	{
		return mat_element_wise_operation(B, C, [a](double b) { return a + b; });
	}

	mat_arr mat_element_wise_sub(const mat_arr& A, double b, mat_arr* C)
	{
		return mat_element_wise_operation(A, C, [b](double a) { return a - b; });
	}

	mat_arr mat_element_wise_sub(double a, const mat_arr& B, mat_arr* C)
	{
		return mat_element_wise_operation(B, C, [a](double b) { return a - b; });
	}

	mat_arr mat_element_wise_mul(const mat_arr& A, double b, mat_arr* C)
	{
		return mat_element_wise_operation(A, C, [b](double a) { return a * b; });
	}

	mat_arr mat_element_wise_mul(double a, const mat_arr& B, mat_arr* C)
	{
		return mat_element_wise_operation(B, C, [a](double b) { return a * b; });
	}

	mat_arr mat_element_wise_div(const mat_arr& A, double b, mat_arr* C)
	{
		return mat_element_wise_operation(A, C, [b](double a) { return a / b; });
	}

	mat_arr mat_element_wise_div(double a, const mat_arr& B, mat_arr* C)
	{
		return mat_element_wise_operation(B, C, [a](double b) { return a / b; });
	}

#pragma region matrix_mul_internal
	inline void __matrix_mul_size_check(const unsigned count_a, const unsigned rows_a, const unsigned cols_a,
	                                    const unsigned count_b, const unsigned rows_b, const unsigned cols_b,
	                                    const unsigned count_c, const unsigned rows_c, const unsigned cols_c)
	{
		if (count_a != 1 && count_b != 1 && count_a != count_b)
		{
			throw runtime_error("Wrong input array sizes");
		}

		if (count_c != max(count_a, count_b))
		{
			throw runtime_error("Wrong output array sizes");
		}

		if (cols_a != rows_b)
		{
			throw runtime_error("A and B cannot be multiplied");
		}

		if (rows_a != rows_c || cols_b != cols_c)
		{
			throw runtime_error("C has wrong size");
		}
	}

	void __mat_matrix_mul_add_case0(const mat_arr& A, const mat_arr& B, mat_arr* C)
	{
		const unsigned count = A.count;
		const unsigned l = A.rows;
		const unsigned m = A.cols;
		const unsigned n = B.cols;


		for (unsigned matNo = 0; matNo < count; matNo++)
		{
			const double* a = A.start() + (matNo * l * m) % A.size();
			const double* b = B.start() + (matNo * m * n) % B.size();
			double* c = C->start() + (matNo * l * n);

			for (unsigned i = 0; i < l; i++)
			{
				const double* a_row = a + (i * m);
				double* c_row = c + (i * n);
				const double* b_element = b;
				for (unsigned k = 0; k < m; k++)
				{
					const double a_element_value = *(a_row + k);
					double* c_element = c_row;
					for (unsigned j = 0; j < n; j++)
					{
						*c_element += a_element_value * *b_element;
						b_element++;
						c_element++;
					}
				}
			}
		}
	}

	void __mat_matrix_mul_add_case1(const mat_arr& A, const mat_arr& B, mat_arr* C)
	{
		throw runtime_error("Not supported yet");
	}

	void __mat_matrix_mul_add_case2(const mat_arr& A, const mat_arr& B, mat_arr* C)
	{
		throw runtime_error("Not supported yet");
	}

	void __mat_matrix_mul_add_case3(const mat_arr& A, const mat_arr& B, mat_arr* C)
	{
		throw runtime_error("Not supported yet");
	}

	void __mat_matrix_mul_add(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
#ifdef MAT_ARR_MATH_SIZE_CHECK
		const bool transpose_a = tr == transpose_A || tr == transpose_both;
		const bool transpose_b = tr == transpose_B || tr == transpose_both;
		__matrix_mul_size_check(A.count, transpose_a ? A.cols : A.rows, transpose_a ? A.rows : A.cols,
		                        B.count, transpose_b ? B.cols : B.rows, transpose_b ? B.rows : B.cols,
		                        C->count, C->rows, C->cols);

		if (A.start() == C->start() || B.start() == C->start())
		{
			throw runtime_error("Matrix mul in place not possible");
		}
#endif

		switch (tr)
		{
		case transpose_no:
			__mat_matrix_mul_add_case0(A, B, C);
			return;
		case transpose_A:
			__mat_matrix_mul_add_case1(A, B, C);
			return;
		case transpose_B:
			__mat_matrix_mul_add_case2(A, B, C);
			return;
		case transpose_both:
			__mat_matrix_mul_add_case3(A, B, C);
			return;
		}
	}

	void __mat_matrix_mul(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		unsigned size = C->size();
		double* c = C->start();
		for (unsigned i = 0; i < size; i++)
		{
			*(c + i) = 0.0;
		}

		__mat_matrix_mul_add(A, B, C, tr);
	}
#pragma endregion

	mat_arr mat_matrix_mul_add(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		if (C == nullptr)
		{
			mat_arr tempC = mat_arr(max(A.count, B.count),
			                        (tr == transpose_A || tr == transpose_both) ? A.cols : A.rows,
			                        (tr == transpose_B || tr == transpose_both) ? B.rows : B.cols);
			__mat_matrix_mul_add(A, B, &tempC, tr);
			return tempC;
		}
		__mat_matrix_mul_add(A, B, C, tr);
		return *C;
	}

	mat_arr mat_matrix_mul(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		if (C == nullptr)
		{
			mat_arr tempC = mat_arr(max(A.count, B.count),
			                        (tr == transpose_A || tr == transpose_both) ? A.cols : A.rows,
			                        (tr == transpose_B || tr == transpose_both) ? B.rows : B.cols);
			__mat_matrix_mul_add(A, B, &tempC, tr);
			return tempC;
		}
		__mat_matrix_mul(A, B, C, tr);
		return *C;
	}

#pragma region transpose_internal
	void __transpose_size_check(const unsigned count_a, const unsigned rows_a, const unsigned cols_a,
	                            const unsigned count_c, const unsigned rows_c, const unsigned cols_c)
	{
		if (count_a != count_c)
		{
			throw runtime_error("Wrong array sizes");
		}

		if (rows_a != cols_c || cols_a != rows_c)
		{
			throw runtime_error("Wrong matrix dimensions");
		}
	}

	void __mat_transpose(const mat_arr& A, mat_arr* C)
	{
#ifdef MAT_ARR_MATH_SIZE_CHECK
		__transpose_size_check(A.count, A.rows, A.cols,
		                       C->count, C->rows, C->cols);
#endif

		const double* a = A.start();
		double* c = C->start();

		const unsigned rows = C->rows;
		const unsigned rows_count = C->count * C->rows;
		const unsigned cols = C->cols;

		int i_normal = 0;
		int i_transposed = 0;
		for (unsigned i = 0; i < rows_count; ++i)
		{
			for (unsigned col = 0; col < cols; ++col)
			{
				*(c + i_normal) = *(a + i_transposed);
				i_normal++;
				i_transposed += rows;
			}
			i_transposed -= (cols * rows - 1);
		}
	}
#pragma endregion

	mat_arr mat_transpose(const mat_arr& A, mat_arr* C)
	{
		if (C == nullptr)
		{
			mat_arr tempC = mat_arr(A.count, A.cols, A.rows);
			__mat_transpose(A, C);
			return tempC;
		}
		__mat_transpose(A, C);
		return *C;
	}

	mat_arr mat_set_all(double val, mat_arr* C)
	{
		if (C == nullptr)
		{
			throw runtime_error("C is nullptr");
		}
		const unsigned size = C->size();
		double* c = C->start();
		for (unsigned i = 0; i < size; i++)
		{
			*(c + i) = val;
		}
		return *C;
	}
}
