#include "gradient_based_optimizer_cudaops.cuh"

#include "cuda/linalg_cudaops_t.cuh"

struct momentum_update
{
	float alpha, learning_rate;

	momentum_update(float alpha, float learning_rate)
		: alpha(alpha),
		  learning_rate(learning_rate)
	{
	}

	__device__ float operator()(float v, float grad) const
	{
		return alpha * v - learning_rate * grad;
	}
};

void annlib::cuda::cuda_momentum_sgd_update_velocities(float alpha, float learning_rate,
                                                       const mat_arr& gradient_noarr,
                                                       mat_arr* velocities_noarr)
{
	linalg::cuda::cuda_element_by_element_operation(*velocities_noarr, gradient_noarr, velocities_noarr,
	                                                momentum_update(alpha, learning_rate), transpose_no);
}

struct ordinary_sgd_update
{
	float learning_rate;

	ordinary_sgd_update(float learning_rate)
		: learning_rate(learning_rate)
	{
	}

	__device__ float operator()(float target, float grad) const
	{
		return target - learning_rate * grad;
	}
};

void annlib::cuda::cuda_ordinary_sgd_update(float learning_rate,
                                            const mat_arr& gradient_noarr,
                                            mat_arr* target_noarr)
{
	linalg::cuda::cuda_element_by_element_operation(*target_noarr, gradient_noarr, target_noarr,
	                                                ordinary_sgd_update(learning_rate), transpose_no);
}
