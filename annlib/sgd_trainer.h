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
#include "neural_network.h"
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
		std::shared_ptr<gradient_based_optimizer> optimizer;

		unsigned get_layer_count() const;

		sgd_trainer add_layer(network_layer* layer);

		void init();

		void train_epochs(const training_data& training_data, double epoch_count, bool print = false);

		neural_network to_neural_network(bool copy_parameters = false);

	private:
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
