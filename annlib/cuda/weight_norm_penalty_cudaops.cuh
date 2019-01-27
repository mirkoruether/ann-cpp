#ifndef WEIGHT_NORM_PENALTY_CUDAOPS_CUH
#define WEIGHT_NORM_PENALTY_CUDAOPS_CUH
#include "mat_arr.h"

using namespace linalg;

namespace annlib { namespace cuda
{
	void cuda_add_L1_regularization(const mat_arr& weights_noarr,
	                                float regularization_parameter,
	                                mat_arr* gradient_noarr);

	void cuda_add_L2_regularization(const mat_arr& weights_noarr,
	                                float regularization_parameter,
	                                mat_arr* gradient_noarr);
}}

#endif
