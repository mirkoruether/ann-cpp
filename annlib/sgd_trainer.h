#ifndef SGD_TRAINER_H
#define SGD_TRAINER_H

#include <vector>
#include "mat_arr.h"
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
		unsigned get_layer_count() const;

		std::vector<unsigned> get_sizes() const;

		unsigned get_input_size() const;

		unsigned get_output_size() const;

		void add_layer(std::shared_ptr<network_layer> layer);

		template <typename Ty, typename... Tys>
		void add_new_layer(Tys&&... args)
		{
			add_layer(std::make_shared<Ty>(std::forward<Tys>(args)...));
		}

		void init();

		void train_epochs(const training_data& training_data, gradient_based_optimizer* opt,
		                  unsigned mini_batch_size, double epoch_count, bool print = false);

		mat_arr feed_forward(const mat_arr& in) const;

		fpt calculate_costs(const mat_arr& net_output, const mat_arr& solution) const;

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
