#ifndef MAT_ARR_MATH_H
#define MAT_ARR_MATH_H

//#define MATARRMATH_CHECK_NAN

#include "mat_arr.h"
#include <random>

using namespace linalg;

namespace linalg
{
	mat_arr mat_element_wise_add(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_tr tr = transpose_no);

	mat_arr mat_element_wise_sub(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_tr tr = transpose_no);

	mat_arr mat_element_wise_mul(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_tr tr = transpose_no);

	mat_arr mat_element_wise_div(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_tr tr = transpose_no);

	mat_arr mat_element_wise_max(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_tr tr = transpose_no);

	mat_arr mat_element_wise_min(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                             mat_tr tr = transpose_no);

	mat_arr mat_element_wise_add(const mat_arr& A, fpt b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_add(fpt a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_element_wise_sub(const mat_arr& A, fpt b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_sub(fpt a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_element_wise_mul(const mat_arr& A, fpt b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_mul(fpt a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_element_wise_div(const mat_arr& A, fpt b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_div(fpt a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_element_wise_max(const mat_arr& A, fpt b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_max(fpt a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_element_wise_min(const mat_arr& A, fpt b, mat_arr* C = nullptr);

	mat_arr mat_element_wise_min(fpt a, const mat_arr& B, mat_arr* C = nullptr);

	mat_arr mat_matrix_mul_add(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                           mat_tr tr = transpose_no);

	mat_arr mat_matrix_mul(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                       mat_tr tr = transpose_no);

	mat_arr mat_transpose(const mat_arr& A, mat_arr* C);

	mat_arr mat_set_all(fpt val, mat_arr* C);

	mat_arr mat_concat_mats(const std::vector<mat_arr>& mats, mat_arr* C);

	mat_arr mat_select_mats(const mat_arr& A, const std::vector<unsigned>& indices, mat_arr* C);

	mat_arr mat_random_gaussian(fpt mean, fpt sigma, std::mt19937* rnd, mat_arr* C);

	mat_arr mat_copy(const mat_arr& A, mat_arr* C = nullptr);

	mat_arr mat_max(const mat_arr& A, mat_arr* C = nullptr);

	mat_arr mat_min(const mat_arr& A, mat_arr* C = nullptr);

	mat_arr mat_sum(const mat_arr& A, mat_arr* C = nullptr);

	mat_arr mat_product(const mat_arr& A, mat_arr* C = nullptr);

	fpt mat_total_max(const mat_arr& A);

	fpt mat_total_min(const mat_arr& A);

	fpt mat_total_sum(const mat_arr& A);

	fpt mat_total_product(const mat_arr& A);
} // namespace linalg
#endif
