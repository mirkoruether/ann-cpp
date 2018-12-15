#ifndef SGD_TRAINER_CUDAOPS_CUH
#define SGD_TRAINER_CUDAOPS_CUH
#include "mat_arr.h"

using namespace linalg;

namespace annlib { namespace cuda
{
	void cuda_weight_input(const mat_arr& input_rv,
	                       const mat_arr& weights_noarr,
	                       const mat_arr& biases_rv_noarr,
	                       mat_arr* weighted_inputs_rv);
}}
#endif
