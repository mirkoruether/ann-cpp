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
