#ifndef SGD_TRAINER_H
#define SGD_TRAINER_H

#include <vector>
#include "mat_arr.h"
#include "training_data.h"
#include "gradient_based_optimizer.h"
#include "mini_batch_builder.h"
#include <random>
#include "training_buffer.h"
#include "network_layer.h"

using namespace linalg;

namespace annlib
{
	struct training_status
	{
		unsigned batch_no;
		unsigned batch_count;
		training_buffer* buf;
	};

	class sgd_trainer
	{
	public:
		unsigned get_layer_count() const;

		std::vector<unsigned> get_sizes() const;

		unsigned get_input_size() const;

		unsigned get_output_size() const;

		void add_layer(std::shared_ptr<network_layer> layer);

		template <typename Ty, typename... Tys>
		void add_new_layer(Tys&& ... args)
		{
			add_layer(std::make_shared<Ty>(std::forward<Tys>(args)...));
		}

		void init();

		void train_epochs(gradient_based_optimizer* opt, const mini_batch_builder& mini_batch_builder, double epoch_count,
		                  const std::function<void(training_status)>* logger = nullptr, unsigned log_interval = 100);

		mat_arr feed_forward(const mat_arr& in) const;

		fpt calculate_costs(const mat_arr& net_output, const mat_arr& solution) const;

		network_layer* get_layer(unsigned index);
	private:
		std::vector<std::shared_ptr<network_layer>> layers;

		void do_feed_forward_and_backprop(training_buffer* buffer) const;

		void do_adjustments(gradient_based_optimizer* opt, training_buffer* buffer);
	};
}
#endif
