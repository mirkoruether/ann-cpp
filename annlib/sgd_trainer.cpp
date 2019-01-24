#include "sgd_trainer.h"
#include "mat_arr.h"
#include "mat_arr_math.h"
#include <future>
#include <random>

#include "_calc_macros.h"

#ifdef ANNLIB_USE_CUDA
#include "sgd_trainer_cudaops.cuh"
#include "cuda/linalg_cudaops.cuh"
#endif

using namespace linalg;
using namespace annlib;

sgd_trainer::sgd_trainer()
	: mini_batch_size(8),
	  activation_f(std::make_shared<logistic_activation_function>()),
	  cost_f(std::make_shared<cross_entropy_costs>()),
	  weight_norm_penalty(nullptr),
	  optimizer(std::make_shared<ordinary_sgd>(1.0f)),
	  net_init(std::make_shared<normalized_gaussian_net_init>())
{
}

std::vector<unsigned> sgd_trainer::sizes() const
{
	const unsigned layer_count = get_layer_count();
	std::vector<unsigned> sizes(layer_count + 1);
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

void sgd_trainer::init(std::vector<unsigned>& sizes)
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

void sgd_trainer::train_epochs(const training_data& training_data, const double epoch_count, bool print)
{
	const auto batch_count = static_cast<unsigned>((epoch_count / mini_batch_size) * training_data.input.count);

#if _OPENMP
	const unsigned part_count = std::thread::hardware_concurrency();
#else
	const unsigned part_count = 1;
#endif


	training_buffer buffer(sizes(), mini_batch_size, part_count);
	mini_batch_builder mb_builder(training_data);

	for (unsigned batch_no = 0; batch_no < batch_count; batch_no++)
	{
		if (print && batch_no % 100 == 0)
		{
			std::cout << "\r" << batch_no << "/" << batch_count
				<< " [" << unsigned(100.0 * (batch_no + 1) / batch_count) << "%]";
		}

		mb_builder.build_mini_batch(&buffer.input_rv, &buffer.solution_rv);

		do_feed_forward_and_backprop(&buffer);

		do_adjustments(&buffer);
	}

	if (print)
	{
		std::cout << "\r" << batch_count << "/" << batch_count << " [100%]" << std::endl;
	}
}

neural_network sgd_trainer::to_neural_network(bool copy_parameters)
{
	if (!copy_parameters)
	{
		return neural_network(weights_noarr, biases_noarr_rv,
		                      [=](float f)
		                      {
			                      return activation_f->apply(f);
		                      });
	}

	std::vector<mat_arr> weights_copy_noarr;
	std::vector<mat_arr> biases_copy_noarr_rv;

	for (unsigned i = 0; i < biases_noarr_rv.size(); i++)
	{
		weights_copy_noarr.emplace_back(weights_noarr[i].duplicate());
		biases_copy_noarr_rv.emplace_back(biases_noarr_rv[i].duplicate());
	}
	return neural_network(move(weights_copy_noarr), std::move(biases_copy_noarr_rv),
	                      [=](float f)
	                      {
		                      return activation_f->apply(f);
	                      });
}

void sgd_trainer::feed_forward_detailed(const mat_arr& input,
                                        std::vector<mat_arr>* weighted_inputs_rv,
                                        std::vector<mat_arr>* activations_rv,
                                        std::vector<mat_arr>* activations_dfs_rv) const
{
	const unsigned layer_count = get_layer_count();

	const mat_arr* layerInput = &input;
	for (unsigned layerNo = 0; layerNo < layer_count; layerNo++)
	{
#ifdef ANNLIB_USE_CUDA
		annlib::cuda::cuda_weight_input(*layerInput,
		                                weights_noarr[layerNo],
		                                biases_noarr_rv[layerNo],
		                                &weighted_inputs_rv->operator[](layerNo));
#else
		mat_matrix_mul(*layerInput,
		               weights_noarr[layerNo],
		               &weighted_inputs_rv->operator[](layerNo));

		mat_element_wise_add(weighted_inputs_rv->operator[](layerNo),
		                     biases_noarr_rv[layerNo],
		                     &weighted_inputs_rv->operator[](layerNo));
#endif
		activation_f->apply(weighted_inputs_rv->operator[](layerNo),
		                    &activations_rv->operator[](layerNo));

		activation_f->apply_derivative(weighted_inputs_rv->operator[](layerNo),
		                               &activations_dfs_rv->operator[](layerNo));

		layerInput = &activations_rv->operator[](layerNo);
	}
}

void sgd_trainer::calculate_error(const mat_arr& net_output_rv,
                                  const mat_arr& solution_rv,
                                  const std::vector<mat_arr>& activation_dfs_rv,
                                  std::vector<mat_arr>* errors_rv) const
{
	const unsigned layer_count = get_layer_count();

	mat_arr* error_last_layer = &errors_rv->operator[](layer_count - 1);
	cost_f->calculate_gradient(net_output_rv, solution_rv, error_last_layer);

	M_MUL(*error_last_layer, activation_dfs_rv[layer_count - 1], error_last_layer);

	for (int layer_no = layer_count - 2; layer_no >= 0; --layer_no)
	{
#ifdef ANNLIB_USE_CUDA
		cuda::cuda_backprop_error(errors_rv->operator[](layer_no + 1),
		                          weights_noarr[layer_no + 1],
		                          activation_dfs_rv[layer_no],
		                          &errors_rv->operator[](layer_no));
#else
		mat_matrix_mul(errors_rv->operator[](layer_no + 1),
		               weights_noarr[layer_no + 1],
		               &errors_rv->operator[](layer_no),
		               transpose_B);

		mat_element_wise_mul(errors_rv->operator[](layer_no),
		                     activation_dfs_rv[layer_no],
		                     &errors_rv->operator[](layer_no));
#endif
	}
}

void sgd_trainer::calculate_gradient_weight(const mat_arr& previous_activation_rv,
                                            const mat_arr& error_rv,
                                            mat_arr* gradient_weight_noarr) const
{
	const unsigned batch_entry_count = previous_activation_rv.count;
	M_SET_ALL(0.0f, gradient_weight_noarr);

	for (unsigned batch_entry_no = 0; batch_entry_no < batch_entry_count; batch_entry_no++)
	{
		M_MATMUL_ADD(previous_activation_rv.get_mat(batch_entry_no),
		             error_rv.get_mat(batch_entry_no),
		             gradient_weight_noarr,
		             transpose_A);
	}

	M_DIV(*gradient_weight_noarr, static_cast<float>(batch_entry_count), gradient_weight_noarr);
}

void sgd_trainer::calculate_gradient_bias(const mat_arr& error_rv,
                                          mat_arr* gradient_bias_noarr_rv) const
{
	const unsigned batch_entry_count = error_rv.count;
	M_SET_ALL(0.0f, gradient_bias_noarr_rv);

	for (unsigned batch_entry_no = 0; batch_entry_no < batch_entry_count; batch_entry_no++)
	{
		M_ADD(*gradient_bias_noarr_rv,
		      error_rv.get_mat(batch_entry_no),
		      gradient_bias_noarr_rv);
	}

	M_DIV(*gradient_bias_noarr_rv,
	      static_cast<float>(batch_entry_count),
	      gradient_bias_noarr_rv);
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

void sgd_trainer::do_feed_forward_and_backprop(training_buffer* buffer) const
{
	const int part_count = static_cast<int>(buffer->part_count());

#pragma omp parallel for
	for (int part_no = 0; part_no < part_count; part_no++)
	{
		partial_training_buffer* part_buf = &buffer->partial_buffers[part_no];
		feed_forward_detailed(part_buf->input_rv,
		                      &part_buf->weighted_inputs_rv,
		                      &part_buf->activations_rv,
		                      &part_buf->activation_dfs_rv);

		calculate_error(part_buf->activations_rv.back(),
		                part_buf->solution_rv,
		                part_buf->activation_dfs_rv,
		                &part_buf->errors_rv);
	}
}

void sgd_trainer::do_adjustments(training_buffer* buffer)
{
	optimizer->next_mini_batch();

	const int iterations = 2 * static_cast<int>(get_layer_count());

#pragma omp parallel for
	for (int i = 0; i < iterations; i++)
	{
		unsigned layer_no = i / 2;
		if (i % 2 == 0)
		{
			adjust_weights(layer_no, buffer);
		}
		else
		{
			adjust_biases(layer_no, buffer);
		}
	}
}

mini_batch_builder::mini_batch_builder(training_data data)
	: data(std::move(data)),
	  distribution(0, data.input.count - 1)
{
	rng.seed(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));
}

void mini_batch_builder::build_mini_batch(mat_arr* input_rv, mat_arr* solution_rv)
{
	const unsigned mini_batch_size = input_rv->count;
	std::vector<unsigned> batch_indices(mini_batch_size);
	for (unsigned i = 0; i < mini_batch_size; i++)
	{
		batch_indices[i] = distribution(rng);
	}

	mat_select_mats(data.input, batch_indices, input_rv);
	mat_select_mats(data.solution, batch_indices, solution_rv);
}
