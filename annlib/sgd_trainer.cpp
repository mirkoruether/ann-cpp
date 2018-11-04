#include "sgd_trainer.h"
#include "mat_arr.h"
#include "mat_arr_math.h"

using namespace std;
using namespace linalg;
using namespace annlib;

void sgd_trainer::feed_forward_detailed(const mat_arr& input,
                                        vector<mat_arr>* weighted_inputs_rv,
                                        vector<mat_arr>* activations_rv) const
{
	const size_t layer_count = weights_noarr.size();

	const mat_arr* layerInput = &input;
	for (unsigned layerNo = 0; layerNo < layer_count; layerNo++)
	{
		mat_matrix_mul(*layerInput,
		               weights_noarr[layerNo],
		               &weighted_inputs_rv->operator[](layerNo));

		mat_element_wise_add(weighted_inputs_rv->operator[](layerNo),
		                     biases_noarr_rv[layerNo],
		                     &weighted_inputs_rv->operator[](layerNo));

		mat_element_wise_operation(weighted_inputs_rv->operator[](layerNo),
		                           &activations_rv->operator[](layerNo),
		                           activation_f->f);

		layerInput = &activations_rv->operator[](layerNo);
	}
}

void sgd_trainer::calculate_error(const mat_arr& net_output_rv, const mat_arr& solution_rv,
                                  const vector<mat_arr>& weighted_inputs_rv,
                                  const vector<mat_arr>& activations_rv,
                                  vector<mat_arr>* errors_rv) const
{
	const size_t layer_count = weights_noarr.size();

	cost_f->calculate_output_layer_error(net_output_rv,
	                                     solution_rv,
	                                     weighted_inputs_rv[layer_count - 1],
	                                     activation_f->df,
	                                     &errors_rv->operator[](layer_count - 1));

	for (int layer_no = layer_count - 2; layer_no >= 0; --layer_no)
	{
		mat_matrix_mul(errors_rv->operator[](layer_no + 1),
		               weights_noarr[layer_no + 1],
		               &errors_rv->operator[](layer_no),
		               transpose_B);

		const function<double(double)>& actv_df = activation_f->df;

		mat_element_by_element_operation(errors_rv->operator[](layer_no),
		                                 weighted_inputs_rv[layer_no],
		                                 &errors_rv->operator[](layer_no),
		                                 [actv_df](double mat_mul_result, double wi)
		                                 {
			                                 return mat_mul_result * actv_df(wi);
		                                 });
	}
}

void sgd_trainer::calculate_gradient_weights(const mat_arr& input_rv,
                                             const vector<mat_arr>& activations_rv,
                                             const vector<mat_arr>& errors_rv,
                                             vector<mat_arr>* gradient_weights_noarr) const
{
	const size_t layer_count = weights_noarr.size();
	const size_t batch_entry_count = input_rv.count;

	for (unsigned layer_no = 0; layer_no < layer_count; layer_no++)
	{
		const mat_arr& lhs = layer_no == 0 ? input_rv : activations_rv[layer_no - 1];
		const mat_arr& rhs = errors_rv[layer_no];
		mat_arr* grad_noarr = &gradient_weights_noarr->operator[](layer_no);

		for (unsigned batch_entry_no = 0; batch_entry_no < batch_entry_count; batch_entry_no++)
		{
			mat_matrix_mul_add(lhs.get_mat(batch_entry_no),
			                   rhs.get_mat(batch_entry_no),
			                   grad_noarr,
			                   transpose_A);
		}

		mat_element_wise_div(*grad_noarr, batch_entry_count, grad_noarr);

		if (weight_norm_penalty != nullptr)
		{
			weight_norm_penalty->add_penalty_to_gradient(weights_noarr[layer_no], grad_noarr);
		}
	}
}

void sgd_trainer::calculate_gradient_biases(const vector<mat_arr>& errors_rv,
                                            vector<mat_arr>* gradient_biases_noarr_rv) const
{
	const size_t layer_count = weights_noarr.size();
	const size_t batch_entry_count = errors_rv[0].count;

	for (unsigned layer_no = 0; layer_no < layer_count; layer_no++)
	{
		const mat_arr& layer_error = errors_rv[layer_no];
		mat_arr* grad_noarr_rv = &gradient_biases_noarr_rv->operator[](layer_no);

		for (unsigned batch_entry_no = 0; batch_entry_no < batch_entry_count; batch_entry_no++)
		{
			mat_element_wise_add(*grad_noarr_rv,
			                     layer_error.get_mat(batch_entry_no),
			                     grad_noarr_rv);
		}

		mat_element_wise_div(*grad_noarr_rv, batch_entry_count, grad_noarr_rv);
	}
}
