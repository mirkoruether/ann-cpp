#ifndef TRAINING_BUFFER_H
#define TRAINING_BUFFER_H

#include "mat_arr.h"

using namespace linalg;

namespace annlib
{
	class partial_training_buffer;

	class training_buffer
	{
	public:
		training_buffer(std::vector<unsigned> sizes, unsigned mini_batch_size, unsigned part_count);

		mat_arr input_rv;
		mat_arr solution_rv;

		std::vector<mat_arr> weighted_inputs_rv;
		std::vector<mat_arr> activations_rv;
		std::vector<mat_arr> activation_dfs_rv;
		std::vector<mat_arr> errors_rv;

		std::vector<mat_arr> gradient_biases_rv_noarr;
		std::vector<mat_arr> gradient_weights_noarr;

		std::vector<partial_training_buffer> partial_buffers;

		std::vector<mat_arr*> all();
		void clear();

		unsigned layer_count() const;
		unsigned part_count() const;
	};

	class partial_training_buffer
	{
	public:
		partial_training_buffer(training_buffer* buf,
		                        unsigned start, unsigned count);

		mat_arr input_rv;
		mat_arr solution_rv;

		std::vector<mat_arr> weighted_inputs_rv;
		std::vector<mat_arr> activations_rv;
		std::vector<mat_arr> activation_dfs_rv;
		std::vector<mat_arr> errors_rv;
	};
}
#endif
