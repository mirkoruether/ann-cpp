#include "activation_function_cudaops.cuh"

#include "cuda/linalg_cudaops_t.cuh"

using namespace linalg::cuda;

struct sigmoid
{
	__device__ float operator()(const float f) const
	{
		return 1.0f / (1.0f + std::exp(-f));
	}
};

void annlib::cuda::cuda_sigmoid_apply(const mat_arr& in, mat_arr* target)
{
	cuda_element_wise_operation(in, target, sigmoid());
}

struct sigmoid_derivative
{
	__device__ float operator()(const float f) const
	{
		const float e_abs = std::abs(f);
		if (e_abs > 5.0f)
			return 1.0f / std::exp(e_abs);
		const float v = std::exp(f) + 1.0f;
		return std::exp(f) / (v * v);
	}
};

void annlib::cuda::cuda_sigmoid_apply_derivative(const mat_arr& in, mat_arr* target)
{
	cuda_element_wise_operation(in, target, sigmoid_derivative());
}

struct relu
{
	__device__ float operator()(const float f) const
	{
		return std::max(0.0f, f);
	}
};

void annlib::cuda::cuda_relu_apply(const mat_arr& in, mat_arr* target)
{
	cuda_element_wise_operation(in, target, relu());
}

struct relu_derivative
{
	__device__ float operator()(const float f) const
	{
		return f > 0 ? 1.0f : 0.0f;
	}
};

void annlib::cuda::cuda_relu_apply_derivative(const mat_arr& in, mat_arr* target)
{
	cuda_element_wise_operation(in, target, relu_derivative());
}
