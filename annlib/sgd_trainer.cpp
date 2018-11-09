#include "sgd_trainer.h"
#include "mat_arr.h"
#include "mat_arr_math.h"
#include <random>
#include "general_util.h"

using namespace std;
using namespace linalg;
using namespace annlib;

void sgd_trainer::build_batch(const training_data& training_data, mat_arr* input_rv, mat_arr* solution_rv,
                              const function<double()>& random) const
{
	vector<unsigned> batch_indices(mini_batch_size);
	for (unsigned i = 0; i < mini_batch_size; i++)
	{
		batch_indices[i] = random();
	}

	mat_select_mats(training_data.input, batch_indices, input_rv);
	mat_select_mats(training_data.solution, batch_indices, solution_rv);
}

vector<unsigned> sgd_trainer::sizes() const
{
	vector<unsigned> sizes(weights_noarr.size() + 1);
	sizes[0] = weights_noarr[0].cols;
	for (unsigned i = 1; i < sizes.size(); i++)
	{
		sizes[i] = biases_noarr_rv[i - 1].cols;
	}
	return sizes;
}

void sgd_trainer::train_epochs(const training_data& training_data, unsigned epoch_count)
{
	const unsigned training_size = training_data.input.count;
	const unsigned batch_count = epoch_count * (training_size / mini_batch_size);

	training_buffer buffer(sizes(), mini_batch_size);

	random_device rd;
	mt19937 rng(rd());
	const uniform_int_distribution<unsigned> randomNumber(0, static_cast<unsigned>(training_size - 1));

	for (unsigned batch_no = 0; batch_no < batch_count; batch_no++)
	{
		build_batch(training_data, &buffer.input_rv, &buffer.solution_rv,
		            [&]() { return randomNumber(rng); });

		feed_forward_detailed(buffer.input_rv,
		                      &buffer.weighted_inputs_rv, &buffer.activations_rv);

		calculate_error(buffer.activations_rv.back(), buffer.solution_rv, buffer.weighted_inputs_rv,
		                &buffer.errors_rv);

		calculate_gradient_weights(buffer.input_rv, buffer.activations_rv, buffer.errors_rv,
		                           &buffer.gradient_weights_noarr);

		calculate_gradient_biases(buffer.errors_rv,
		                          &buffer.gradient_biases_rv_noarr);

		optimizer->next_mini_batch();
		optimizer->adjust_weights(buffer.gradient_weights_noarr, &weights_noarr);
		optimizer->adjust_biases(buffer.gradient_biases_rv_noarr, &biases_noarr_rv);
	}
}

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

void sgd_trainer::calculate_error(const mat_arr& net_output_rv,
                                  const mat_arr& solution_rv,
                                  const vector<mat_arr>& weighted_inputs_rv,
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
		mat_set_all(0, grad_noarr);

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

training_buffer::training_buffer(vector<unsigned> sizes, unsigned mini_batch_size)
	: input_rv(mini_batch_size, 1, sizes.front()),
	  solution_rv(mini_batch_size, 1, sizes.back())
{
	const unsigned layer_count = sizes.size() - 1;
	for (unsigned i = 0; i < layer_count; i++)
	{
		const unsigned layer_size = sizes[i + 1];
		weighted_inputs_rv.emplace_back(mat_arr(mini_batch_size, 1, layer_size));
		activations_rv.emplace_back(mat_arr(mini_batch_size, 1, layer_size));
		errors_rv.emplace_back(mat_arr(mini_batch_size, 1, layer_size));

		gradient_biases_rv_noarr.emplace_back(mat_arr(1, 1, layer_size));
		gradient_weights_noarr.emplace_back(mat_arr(1, sizes[i], layer_size));
	}
}


vector<mat_arr*> training_buffer::all()
{
	vector<mat_arr*> result;
	add_pointers(weighted_inputs_rv, &result);
	add_pointers(activations_rv, &result);
	add_pointers(errors_rv, &result);
	add_pointers(gradient_biases_rv_noarr, &result);
	add_pointers(gradient_weights_noarr, &result);
	result.emplace_back(&input_rv);
	result.emplace_back(&solution_rv);
	return result;
}

void training_buffer::clear()
{
	for (auto ma : all())
	{
		mat_set_all(0, ma);
	}
}
