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

	void cuda_backprop_error(const mat_arr& error_next_layer_rv,
	                         const mat_arr& weights_next_layer_noarr,
	                         const mat_arr& act_df_rv,
	                         mat_arr* error_rv);

	void cuda_calculate_gradient_weight(const mat_arr& previous_activation_rv,
	                                    const mat_arr& error_rv,
	                                    mat_arr* gradient_weight_noarr);

	void cuda_calculate_gradient_bias(const mat_arr& error_rv,
	                                  mat_arr* gradient_bias_noarr);
}}
#endif
