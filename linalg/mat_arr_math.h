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

	mat_arr mat_random_gaussian(float mean, float sigma, std::random_device* rnd, mat_arr* C);
} // namespace linalg
#endif
