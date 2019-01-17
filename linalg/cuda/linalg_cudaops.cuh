#ifndef LINALG_CUDAOPS_CUH
#define LINALG_CUDAOPS_CUH
#include "mat_arr.h"

using namespace linalg;

namespace linalg { namespace cuda
{
	mat_arr cuda_matrix_mul(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                        mat_tr tr = transpose_no);

	mat_arr cuda_matrix_mul_add(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                            mat_tr tr = transpose_no);

	mat_arr cuda_element_wise_add(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
	                              mat_tr tr = transpose_no);

	mat_arr cuda_element_wise_sub(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
		mat_tr tr = transpose_no);

	mat_arr cuda_element_wise_mul(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
		mat_tr tr = transpose_no);

	mat_arr cuda_element_wise_div(const mat_arr& A, const mat_arr& B, mat_arr* C = nullptr,
		mat_tr tr = transpose_no);
}}
#endif
