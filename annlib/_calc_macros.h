#ifndef _CALC_MACROS_H
#define _CALC_MACROS_H

#ifdef ANNLIB_USE_CUDA
	#include "cuda/linalg_cudaops.cuh"

	#define M_ADD linalg::cuda::cuda_element_wise_add
	#define M_SUB linalg::cuda::cuda_element_wise_sub
	#define M_MUL linalg::cuda::cuda_element_wise_mul
	#define M_DIV linalg::cuda::cuda_element_wise_div

	#define M_MATMUL linalg::cuda::cuda_matrix_mul
	#define M_MATMUL_ADD linalg::cuda::cuda_matrix_mul_add

	#define M_SET_ALL linalg::cuda::cuda_set_all
	#define M_SELECT linalg::cuda::cuda_select_mats
#else
	#define M_ADD linalg::mat_element_wise_add
	#define M_SUB linalg::mat_element_wise_sub
	#define M_MUL linalg::mat_element_wise_mul
	#define M_DIV linalg::mat_element_wise_div

	#define M_MATMUL linalg::mat_matrix_mul
	#define M_MATMUL_ADD linalg::mat_matrix_mul_add

	#define M_SET_ALL linalg::mat_set_all
	#define M_SELECT linalg::mat_select_mats
#endif

#endif