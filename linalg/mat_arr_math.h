#ifndef MAT_ARR_MATH_H
#define MAT_ARR_MATH_H

#include "mat_arr.h"
#include <functional>

using namespace linalg;

namespace linalg
{
	enum mat_transpose
	{
		transpose_no,
		transpose_A,
		transpose_B,
		transpose_both
	};

	void element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C,
	                                  const function<double(double, double)>& f, mat_transpose tr);

	void element_wise_operation(const mat_arr& A, mat_arr* C,
	                            const function<double(double)>& f);
}
#endif
