#define MAT_ARR_MATH_SIZE_CHECK 1

#include "mat_arr_math.h"

using namespace linalg;

namespace linalg
{
	inline void element_by_element_operation_size_check(unsigned count_a, unsigned rows_a, unsigned cols_a,
	                                                    unsigned count_b, unsigned rows_b, unsigned cols_b,
	                                                    unsigned count_c, unsigned rows_c, unsigned cols_c)
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

	void element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C,
	                                  const function<double(double, double)>& f,
	                                  const bool transpose_a, const bool transpose_b)
	{
		const unsigned count = C->count;
		const unsigned rows = C->rows;
		const unsigned cols = C->cols;

#ifdef MAT_ARR_MATH_SIZE_CHECK
		element_by_element_operation_size_check(A.count, transpose_a ? A.cols : A.rows, transpose_a ? A.rows : A.cols,
		                                        B.count, transpose_b ? B.cols : B.rows, transpose_b ? B.rows : B.cols,
		                                        count, rows, cols);
#endif

		const double* a = A.start();
		const double* b = B.start();
		double* c = C->start();

		const unsigned size_A = A.size();
		const unsigned size_B = B.size();

		unsigned i_normal = 0;
		unsigned i_transposed = 0;
		for (unsigned index = 0; index < count; ++index)
		{
			for (unsigned row = 0; row < rows; ++row)
			{
				for (unsigned col = 0; col < cols; ++col)
				{
					*(c + i_normal) = f(*(a + ((transpose_a ? i_transposed : i_normal) % size_A)),
					                    *(b + ((transpose_b ? i_transposed : i_normal) % size_B)));
					i_normal++;
					i_transposed += cols;
				}
				i_transposed -= (cols * rows - 1);
			}
		}
	}

	void element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C,
	                                  const function<double(double, double)>& f, const mat_transpose tr)
	{
		switch (tr)
		{
		case transpose_no:
			element_by_element_operation(A, B, C, f, false, false);
			break;
		case transpose_A:
			element_by_element_operation(A, B, C, f, true, false);
			break;
		case transpose_B:
			element_by_element_operation(A, B, C, f, false, true);
			break;
		case transpose_both:
			element_by_element_operation(A, B, C, f, true, true);
			break;
		}
	}

	void element_wise_operation(const mat_arr& A, mat_arr* C,
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
}
