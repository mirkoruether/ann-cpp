#include "activation_function_cudaops.cuh"

#include "cuda/cuda_util.cuh"

using namespace linalg::cuda;

struct sigmoid
{
	__device__ float operator()(const float f) const
	{
		return 1.0f / (1.0f + std::exp(-f));
	}
};

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

template <typename Fc>
__global__ void apply_function_kernel(const float* in, float* target, unsigned size, const Fc& f)
{
	const unsigned pos = current_pos_linear();
	if (pos < size)
	{
		target[pos] = f(in[pos]);
	}
}

void annlib::cuda::cuda_sigmoid_apply(const mat_arr& in, mat_arr* target)
{
	prepare_launch_linear(*target, [&](unsigned size, unsigned threads, unsigned blocks)
	{
		apply_function_kernel << <blocks, threads >> >(in.dev_start(),
		                                               target->dev_start(),
		                                               size,
		                                               sigmoid());
	});
}

void annlib::cuda::cuda_sigmoid_apply_derivative(const mat_arr& in, mat_arr* target)
{
	prepare_launch_linear(*target, [&](unsigned size, unsigned threads, unsigned blocks)
	{
		apply_function_kernel << <blocks, threads >> >(in.dev_start(),
		                                               target->dev_start(),
		                                               size,
		                                               sigmoid_derivative());
	});
}
