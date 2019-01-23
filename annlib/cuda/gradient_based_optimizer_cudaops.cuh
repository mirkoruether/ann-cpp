#ifndef GRADIENT_BASED_OPTIMIZER_CUDAOPS_CUH
#define GRADIENT_BASED_OPTIMIZER_CUDAOPS_CUH
#include "mat_arr.h"

using namespace linalg;

namespace annlib { namespace cuda
{
	void cuda_momentum_sgd_update_velocities(float alpha, float learning_rate,
	                                         const mat_arr& gradient_noarr,
	                                         mat_arr* velocities_noarr);

	void cuda_ordinary_sgd_update(float learning_rate,
	                              const mat_arr& gradient_noarr,
	                              mat_arr* target_noarr);
}}

#endif
