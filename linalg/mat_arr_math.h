#ifndef MAT_ARR_MATH_H
#define MAT_ARR_MATH_H

#include "mat_arr.h"
#include <functional>
#include <array>

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

	mat_arr mat_multiple_e_by_e_operation(const vector<mat_arr*>& input, mat_arr* C,
	                                      const function<double(vector<double>)>& f);

	mat_arr mat_element_by_element_operation(const mat_arr& A, const mat_arr& B, mat_arr* C,
	                                         const function<double(double, double)>& f,
	                                         mat_tr tr = transpose_no);

	mat_arr mat_element_wise_add(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_tr tr = transpose_no);

	mat_arr mat_element_wise_sub(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_tr tr = transpose_no);

	mat_arr mat_element_wise_mul(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_tr tr = transpose_no);

	mat_arr mat_element_wise_div(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_tr tr = transpose_no);

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

	mat_arr mat_matrix_mul_add(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                           mat_tr tr = transpose_no);

	mat_arr mat_matrix_mul(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                       mat_tr tr = transpose_no);

	mat_arr mat_transpose(const mat_arr& A, mat_arr* C);

	mat_arr mat_set_all(double val, mat_arr* C);

	mat_arr mat_concat_mats(const vector<mat_arr>& mats, mat_arr* C);

	mat_arr mat_select_mats(const mat_arr& A, const vector<unsigned>& indices, mat_arr* C);
}
#endif
