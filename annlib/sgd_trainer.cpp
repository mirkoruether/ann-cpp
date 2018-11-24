#include "sgd_trainer.h"
#include "mat_arr.h"
#include "mat_arr_math.h"
#include <random>
#include "general_util.h"

using namespace std;
using namespace linalg;
using namespace annlib;

sgd_trainer::sgd_trainer()
	: mini_batch_size(16),
	  activation_f(make_shared<logistic_activation_function>(1.0)),
	  cost_f(make_shared<cross_entropy_costs>()),
	  weight_norm_penalty(nullptr),
	  optimizer(make_shared<ordinary_sgd>(1.0)),
	  net_init(make_shared<normalized_gaussian_net_init>())
{
}

vector<unsigned> sgd_trainer::sizes() const
{
	const unsigned layer_count = get_layer_count();
	vector<unsigned> sizes(layer_count + 1);
	sizes[0] = weights_noarr[0].rows;
	for (unsigned i = 0; i < layer_count; i++)
	{
		sizes[i + 1] = biases_noarr_rv[i].cols;
	}
	return sizes;
}

unsigned sgd_trainer::get_layer_count() const
{
	return static_cast<unsigned>(weights_noarr.size());
}

void sgd_trainer::init(vector<unsigned>& sizes)
{
	const auto layer_count = static_cast<unsigned>(sizes.size() - 1);
	weights_noarr.clear();
	biases_noarr_rv.clear();

	for (unsigned i = 0; i < layer_count; i++)
	{
		weights_noarr.emplace_back(1, sizes[i], sizes[i + 1]);
		biases_noarr_rv.emplace_back(1, 1, sizes[i + 1]);

		net_init->init_weights(&weights_noarr[i]);
		net_init->init_biases(&biases_noarr_rv[i]);
	}

	optimizer->init(sizes);
}

void sgd_trainer::train_epochs(const training_data& training_data, const double epoch_count)
{
	const auto batch_count = static_cast<unsigned>((epoch_count / mini_batch_size) * training_data.input.count);

	training_buffer buffer(sizes(), mini_batch_size);
	mini_batch_builder mb_builder(training_data);

	for (unsigned batch_no = 0; batch_no < batch_count; batch_no++)
	{
		mb_builder.build_mini_batch(&buffer.input_rv, &buffer.solution_rv);

		feed_forward_detailed(buffer.input_rv,
		                      &buffer.weighted_inputs_rv, &buffer.activations_rv);

		calculate_error(buffer.activations_rv.back(), buffer.solution_rv, buffer.weighted_inputs_rv,
		                &buffer.errors_rv);

		optimizer->next_mini_batch();

		const unsigned layer_count = get_layer_count();
		for (unsigned layer_no = 0; layer_no < layer_count; layer_no++)
		{
			adjust_weights(layer_no, &buffer);
			adjust_biases(layer_no, &buffer);
		}
	}
}

neural_network sgd_trainer::to_neural_network(bool copy_parameters)
{
	if (!copy_parameters)
	{
		return neural_network(weights_noarr, biases_noarr_rv, activation_f->f);
	}

	vector<mat_arr> weights_copy_noarr;
	vector<mat_arr> biases_copy_noarr_rv;

	for (unsigned i = 0; i < biases_noarr_rv.size(); i++)
	{
		weights_copy_noarr.emplace_back(weights_noarr[i].duplicate());
		biases_copy_noarr_rv.emplace_back(biases_noarr_rv[i].duplicate());
	}
	return neural_network(move(weights_copy_noarr), move(biases_copy_noarr_rv), activation_f->f);
}

void sgd_trainer::feed_forward_detailed(const mat_arr& input,
                                        vector<mat_arr>* weighted_inputs_rv,
                                        vector<mat_arr>* activations_rv) const
{
	const unsigned layer_count = get_layer_count();

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
	const unsigned layer_count = get_layer_count();

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

void sgd_trainer::calculate_gradient_weight(const mat_arr& previous_activation_rv,
                                            const mat_arr& error_rv,
                                            mat_arr* gradient_weight_noarr) const
{
	const unsigned batch_entry_count = previous_activation_rv.count;
	mat_set_all(0, gradient_weight_noarr);

	for (unsigned batch_entry_no = 0; batch_entry_no < batch_entry_count; batch_entry_no++)
	{
		mat_matrix_mul_add(previous_activation_rv.get_mat(batch_entry_no),
		                   error_rv.get_mat(batch_entry_no),
		                   gradient_weight_noarr,
		                   transpose_A);
	}

	mat_element_wise_div(*gradient_weight_noarr, batch_entry_count, gradient_weight_noarr);
}

void sgd_trainer::calculate_gradient_bias(const mat_arr& error_rv,
                                          mat_arr* gradient_bias_noarr_rv) const
{
	const unsigned batch_entry_count = error_rv.count;
	mat_set_all(0, gradient_bias_noarr_rv);

	for (unsigned batch_entry_no = 0; batch_entry_no < batch_entry_count; batch_entry_no++)
	{
		mat_element_wise_add(*gradient_bias_noarr_rv,
		                     error_rv.get_mat(batch_entry_no),
		                     gradient_bias_noarr_rv);
	}

	mat_element_wise_div(*gradient_bias_noarr_rv, batch_entry_count, gradient_bias_noarr_rv);
}

void sgd_trainer::adjust_weights(unsigned layer_no, training_buffer* buffer)
{
	const mat_arr& previous_activation_rv = layer_no == 0
		                                        ? buffer->input_rv
		                                        : buffer->activations_rv[layer_no - 1];

	calculate_gradient_weight(previous_activation_rv, buffer->errors_rv[layer_no],
	                          &buffer->gradient_weights_noarr[layer_no]);

	if (weight_norm_penalty != nullptr)
	{
		weight_norm_penalty->add_penalty_to_gradient(weights_noarr[layer_no],
		                                             &buffer->gradient_weights_noarr[layer_no]);
	}

	optimizer->adjust(buffer->gradient_weights_noarr[layer_no],
	                  &weights_noarr[layer_no],
	                  weights, layer_no);
}

void sgd_trainer::adjust_biases(unsigned layer_no, training_buffer* buffer)
{
	calculate_gradient_bias(buffer->errors_rv[layer_no],
	                        &buffer->gradient_biases_rv_noarr[layer_no]);

	optimizer->adjust(buffer->gradient_biases_rv_noarr[layer_no],
	                  &biases_noarr_rv[layer_no], biases, layer_no);
}

training_buffer::training_buffer(vector<unsigned> sizes, unsigned mini_batch_size)
	: input_rv(mini_batch_size, 1, sizes.front()),
	  solution_rv(mini_batch_size, 1, sizes.back())
{
	const auto layer_count = static_cast<unsigned>(sizes.size() - 1);
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
	add_pointers(&weighted_inputs_rv, &result);
	add_pointers(&activations_rv, &result);
	add_pointers(&errors_rv, &result);
	add_pointers(&gradient_biases_rv_noarr, &result);
	add_pointers(&gradient_weights_noarr, &result);
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

mini_batch_builder::mini_batch_builder(training_data data)
	: data(move(data)),
	  distribution(0, data.input.count - 1)
{
}

void mini_batch_builder::build_mini_batch(mat_arr* input_rv, mat_arr* solution_rv)
{
	const unsigned mini_batch_size = input_rv->count;
	vector<unsigned> batch_indices(mini_batch_size);
	for (unsigned i = 0; i < mini_batch_size; i++)
	{
		batch_indices[i] = distribution(rng);
	}

	mat_select_mats(data.input, batch_indices, input_rv);
	mat_select_mats(data.solution, batch_indices, solution_rv);
}
