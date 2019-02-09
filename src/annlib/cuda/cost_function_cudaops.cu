#include "cost_function_cudaops.cuh"

#include "cuda/linalg_cudaops_t.cuh"

using namespace linalg::cuda;

struct cross_entropy_gradient
{
	__device__ float operator()(float a, float y) const
	{
		return (1.0f - y) / (1.0f - a) - y / a;
	}
};

void annlib::cuda::cuda_cross_entropy_cost_gradient(const mat_arr& net_output_rv,
                                                    const mat_arr& solution_rv,
                                                    mat_arr* gradient_rv)
{
	cuda_element_by_element_operation(net_output_rv, solution_rv, gradient_rv,
	                                  cross_entropy_gradient());
}
