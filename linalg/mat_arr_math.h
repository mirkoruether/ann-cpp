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

	mat_arr mat_element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C,
	                                         const function<double(double, double)>& f, mat_transpose tr);

	mat_arr mat_element_wise_add(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_transpose tr = transpose_no);

	mat_arr mat_element_wise_sub(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_transpose tr = transpose_no);

	mat_arr mat_element_wise_mul(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_transpose tr = transpose_no);

	mat_arr mat_element_wise_div(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_transpose tr = transpose_no);

	mat_arr mat_element_wise_operation(const mat_arr& A, mat_arr* C,
	                                   const function<double(double)>& f);

	mat_arr mat_element_wise_add(const mat_arr& A, double b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_add(double a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_element_wise_sub(const mat_arr& A, double b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_sub(double a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_element_wise_mul(const mat_arr& A, double b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_mul(double a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_element_wise_div(const mat_arr& A, double b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_div(double a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_matrix_mul(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                       mat_transpose tr = transpose_no);
}
#endif
