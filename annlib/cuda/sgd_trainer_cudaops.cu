#include "sgd_trainer_cudaops.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <functional>
#include "cuda/cuda_util.cuh"

using namespace linalg::cuda;

__global__ void weight_input_kernel(const float* in, const float* weights,
                                    const float* biases, float* out,
                                    unsigned input_size, dim3 size)
{
	const dim3 pos = current_pos_cubic();
	if (check_pos_cubic(pos, size))
	{
		const float* in_mat = in + (pos.x * input_size);
		out[index_cubic(pos, size)] = mat_mul_case0_helper(in_mat, weights, pos.y, pos.z, 1, input_size, size.z)
			+ biases[pos.z];
	}
}

void annlib::cuda::cuda_weight_input(const mat_arr& input_rv,
                                     const mat_arr& weights_noarr,
                                     const mat_arr& biases_rv_noarr,
                                     mat_arr* output_rv)
{
	prepare_launch_cubic(*output_rv, [&](dim3 size, dim3 threads, dim3 blocks)
	{
		weight_input_kernel <<< blocks, threads >>>(input_rv.dev_start(),
		                                            weights_noarr.dev_start(),
		                                            biases_rv_noarr.dev_start(),
		                                            output_rv->dev_start(),
		                                            input_rv.cols, size);
	});
}

__global__ void backprop_error_kernel(const float* error_next_layer,
                                      const float* weights_next_layer,
                                      const float* act_df,
                                      float* error,
                                      unsigned next_layer_size, dim3 size)
{
	const dim3 pos = current_pos_cubic();
	if (check_pos_cubic(pos, size))
	{
		const float* error_next_layer_mat = error_next_layer + (pos.x * next_layer_size);
		const unsigned out_index = index_cubic(pos, size);
		error[out_index] = mat_mul_case2_helper(error_next_layer_mat, weights_next_layer,
		                                        pos.y, pos.z, 1, next_layer_size, size.z)
			* act_df[out_index];
	}
}

void annlib::cuda::cuda_backprop_error(const mat_arr& error_next_layer_rv,
                                       const mat_arr& weights_next_layer_noarr,
                                       const mat_arr& act_df_rv,
                                       mat_arr* error_rv)
{
	prepare_launch_cubic(*error_rv, [&](dim3 size, dim3 threads, dim3 blocks)
	{
		backprop_error_kernel << < blocks, threads >> >(error_next_layer_rv.dev_start(),
		                                                weights_next_layer_noarr.dev_start(),
		                                                act_df_rv.dev_start(),
		                                                error_rv->dev_start(),
		                                                error_next_layer_rv.cols, size);
	});
}

__global__ void calculate_gradient_weight_kernel(const float* prev_act, const float* error, float* grad,
                                                 unsigned prev_act_mat_size, unsigned error_mat_size,
                                                 unsigned mini_batch_size, dim3 size)
{
	const dim3 pos = current_pos_cubic();
	if (check_pos_cubic(pos, size))
	{
		float temp = 0.0f;
		for (unsigned i = 0; i < mini_batch_size; i++)
		{
			const float* prev_act_mat = prev_act + i * prev_act_mat_size;
			const float* error_mat = error + i * error_mat_size;
			temp += mat_mul_case1_helper(prev_act_mat, error_mat, pos.y, pos.z, prev_act_mat_size, 1, error_mat_size);
		}
		grad[index_cubic(pos, size)] = temp / static_cast<float>(mini_batch_size);
	}
}

void annlib::cuda::cuda_calculate_gradient_weight(const mat_arr& previous_activation_rv,
                                                  const mat_arr& error_rv,
                                                  mat_arr* gradient_weight_noarr)
{
	prepare_launch_cubic(*gradient_weight_noarr, [&](dim3 size, dim3 threads, dim3 blocks)
	{
		calculate_gradient_weight_kernel << < blocks, threads >> >(previous_activation_rv.dev_start(),
		                                                           error_rv.dev_start(),
		                                                           gradient_weight_noarr->dev_start(),
		                                                           previous_activation_rv.cols, error_rv.cols,
		                                                           previous_activation_rv.count, size);
	});
}

__global__ void calculate_gradient_bias_kernel(const float* error, float* grad, unsigned mini_batch_size, unsigned size)
{
	const unsigned pos = current_pos_linear();
	if (pos < size)
	{
		float temp = 0.0f;
		for (unsigned i = 0; i < mini_batch_size; i++)
		{
			const float* error_mat = error + i * size;
			temp += error_mat[pos];
		}
		grad[pos] = temp / static_cast<float>(mini_batch_size);
	}
}

void annlib::cuda::cuda_calculate_gradient_bias(const mat_arr& error_rv,
                                                mat_arr* gradient_bias_noarr)
{
	prepare_launch_linear(*gradient_bias_noarr, [&](unsigned size, unsigned threads, unsigned blocks)
	{
		calculate_gradient_bias_kernel << < blocks, threads >> >(error_rv.dev_start(),
		                                                         gradient_bias_noarr->dev_start(),
		                                                         error_rv.count, size);
	});
}
