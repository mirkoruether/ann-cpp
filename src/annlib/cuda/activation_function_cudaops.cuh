#ifndef ACTIVATION_FUNCTION_CUDAOPS_CUH
#define ACTIVATION_FUNCTION_CUDAOPS_CUH
#include "mat_arr.h"

using namespace linalg;

namespace annlib { namespace cuda
{
	void cuda_sigmoid_apply(const mat_arr& in, mat_arr* target);

	void cuda_sigmoid_apply_derivative(const mat_arr& in, mat_arr* target);

	void cuda_relu_apply(const mat_arr& in, mat_arr* target);

	void cuda_relu_apply_derivative(const mat_arr& in, mat_arr* target);
}}

#endif
