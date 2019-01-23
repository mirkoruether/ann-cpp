#ifndef COST_FUNCTION_CUDAOPS_CUH
#define COST_FUNCTION_CUDAOPS_CUH
#include "mat_arr.h"

using namespace linalg;

namespace annlib { namespace cuda
{
	void cuda_cross_entropy_cost_gradient(const mat_arr& net_output_rv,
	                                      const mat_arr& solution_rv,
	                                      mat_arr* gradient_rv);
}}

#endif
