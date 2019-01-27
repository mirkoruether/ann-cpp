#include "weight_norm_penalty_cudaops.cuh"

#include "cuda/linalg_cudaops_t.cuh"

struct L1_reg
{
	float regularization_parameter;

	explicit L1_reg(float regularization_parameter)
		: regularization_parameter(regularization_parameter)
	{
	}

	__device__ float operator()(float grad, float weight) const
	{
		return weight > 0 ? grad + regularization_parameter : grad - regularization_parameter;
	}
};

void annlib::cuda::cuda_add_L1_regularization(const mat_arr& weights_noarr,
                                              float regularization_parameter,
                                              mat_arr* gradient_noarr)
{
	linalg::cuda::cuda_element_by_element_operation(*gradient_noarr, weights_noarr, gradient_noarr,
	                                                L1_reg(regularization_parameter), transpose_no);
}

struct L2_reg
{
	float regularization_parameter;

	explicit L2_reg(float regularization_parameter)
		: regularization_parameter(regularization_parameter)
	{
	}

	__device__ float operator()(float grad, float weight) const
	{
		return grad + weight * regularization_parameter;
	}
};

void annlib::cuda::cuda_add_L2_regularization(const mat_arr& weights_noarr, float regularization_parameter,
                                              mat_arr* gradient_noarr)
{
	linalg::cuda::cuda_element_by_element_operation(*gradient_noarr, weights_noarr, gradient_noarr,
	                                                L2_reg(regularization_parameter), transpose_no);
}
