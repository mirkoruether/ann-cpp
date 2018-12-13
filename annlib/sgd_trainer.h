#ifndef SGD_TRAINER_H
#define SGD_TRAINER_H

#include <vector>
#include "mat_arr.h"
#include "cost_function.h"
#include "activation_function.h"
#include "weight_norm_penalty.h"
#include "training_data.h"
#include "gradient_based_optimizer.h"
#include <random>
#include "net_init.h"
#include "neural_network.h"
#include "training_buffer.h"

using namespace linalg;

namespace annlib
{
	class training_buffer;

	class sgd_trainer
	{
	public:
		sgd_trainer();

		unsigned mini_batch_size;
		std::shared_ptr<activation_function> activation_f;
		std::shared_ptr<cost_function> cost_f;
		std::shared_ptr<weight_norm_penalty> weight_norm_penalty;
		std::shared_ptr<gradient_based_optimizer> optimizer;
		std::shared_ptr<net_init> net_init;

		std::vector<unsigned> sizes() const;

		unsigned get_layer_count() const;

		void init(std::vector<unsigned>& sizes);

		void train_epochs(const training_data& training_data, double epoch_count);

		neural_network to_neural_network(bool copy_parameters = false);

	private:
		std::vector<mat_arr> weights_noarr;
		std::vector<mat_arr> biases_noarr_rv;

		void feed_forward_detailed(const mat_arr& input,
		                           std::vector<mat_arr>* weighted_inputs_rv,
		                           std::vector<mat_arr>* activations_rv) const;

		void calculate_error(const mat_arr& net_output_rv,
		                     const mat_arr& solution_rv,
		                     const std::vector<mat_arr>& weighted_inputs_rv,
		                     std::vector<mat_arr>* errors_rv) const;

		void calculate_gradient_weight(const mat_arr& previous_activation_rv,
		                               const mat_arr& error_rv,
		                               mat_arr* gradient_weight_noarr) const;

		void calculate_gradient_bias(const mat_arr& error_rv,
		                             mat_arr* gradient_bias_noarr_rv) const;

		void adjust_weights(unsigned layer_no, training_buffer* buffer);
		void adjust_biases(unsigned layer_no, training_buffer* buffer);

		void do_feed_forward_and_backprop(training_buffer* buffer) const;
		void do_adjustments(training_buffer* buffer);
	};

	class mini_batch_builder
	{
	public:
		const training_data& data;
		explicit mini_batch_builder(training_data data);

		void build_mini_batch(mat_arr* input_rv, mat_arr* solution_rv);
	private:
		std::uniform_int_distribution<unsigned> distribution;
		std::mt19937 rng;
	};
}
#endif
