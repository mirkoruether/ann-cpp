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
#include "training_buffer.h"
#include "network_layer.h"

using namespace linalg;

namespace annlib
{
	class training_buffer;

	class sgd_trainer
	{
	public:
		sgd_trainer();

		unsigned mini_batch_size;
		std::shared_ptr<cost_function> cost_f;

		unsigned get_layer_count() const;

		std::vector<unsigned> get_sizes() const;

		unsigned get_input_size() const;

		unsigned get_output_size() const;

		void add_layer(std::shared_ptr<network_layer> layer);

		void init();

		void train_epochs(const training_data& training_data, gradient_based_optimizer* opt,
		                  double epoch_count, bool print = false);

		mat_arr feed_forward(const mat_arr& in) const;

	private:
		std::vector<std::shared_ptr<network_layer>> layers;

		void do_feed_forward_and_backprop(training_buffer* buffer) const;
		void do_adjustments(gradient_based_optimizer* opt, training_buffer* buffer);
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
