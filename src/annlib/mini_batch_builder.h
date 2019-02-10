#ifndef ANN_CPP_MINI_BATCH_BUILDER_H
#define ANN_CPP_MINI_BATCH_BUILDER_H

#include "training_data.h"
#include <random>

using namespace annlib;
using namespace linalg;

namespace annlib
{
	class mini_batch_builder
	{
	protected:
		mutable std::uniform_int_distribution<unsigned> distribution;
		mutable std::mt19937 rng;

		virtual void build_mini_batch_internal(mat_arr* input_rv, mat_arr* solution_rv, std::vector<unsigned> batch_indices) const = 0;

	public:
		virtual ~mini_batch_builder() = default;

		const unsigned mini_batch_size;
		const unsigned training_data_size;

		mini_batch_builder(const unsigned int mini_batch_size, const unsigned int training_data_size);

		virtual void build_mini_batch(mat_arr* input_rv, mat_arr* solution_rv) const;
	};

	class default_mini_batch_builder : public mini_batch_builder
	{
	public:
		const training_data& data;

		explicit default_mini_batch_builder(unsigned mini_batch_size, training_data data);

	protected:
		void build_mini_batch_internal(mat_arr* input_rv, mat_arr* solution_rv, std::vector<unsigned> batch_indices) const override;
	};
}

#endif //ANN_CPP_MINI_BATCH_BUILDER_H
