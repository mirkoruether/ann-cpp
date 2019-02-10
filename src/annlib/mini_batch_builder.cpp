#include <algorithm>
#include <chrono>
#include "mini_batch_builder.h"
#include "mat_arr_math.h"

mini_batch_builder::mini_batch_builder(const unsigned int mini_batch_size, const unsigned int training_data_size)
	: distribution(0, training_data_size - 1),
	  rng(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count())),
	  mini_batch_size(mini_batch_size),
	  training_data_size(training_data_size)
{
}

void mini_batch_builder::build_mini_batch(mat_arr* input_rv, mat_arr* solution_rv) const
{
	if (input_rv == nullptr || solution_rv == nullptr)
	{
		throw std::runtime_error("Nullptr");
	}

	if (input_rv->count != mini_batch_size || solution_rv->count != mini_batch_size)
	{
		throw std::runtime_error("Wrong count");
	}

	std::vector<unsigned> batch_indices(mini_batch_size);
	for (unsigned i = 0; i < mini_batch_size; i++)
	{
		batch_indices[i] = distribution(rng);
	}

	build_mini_batch_internal(input_rv, solution_rv, batch_indices);
}

default_mini_batch_builder::default_mini_batch_builder(unsigned mini_batch_size, training_data data)
	: mini_batch_builder(mini_batch_size, data.entry_count()),
	  data(std::move(data))
{
}

void default_mini_batch_builder::build_mini_batch_internal(mat_arr* input_rv, mat_arr* solution_rv, std::vector<unsigned> batch_indices) const
{
	mat_select_mats(data.input, batch_indices, input_rv);
	mat_select_mats(data.solution, batch_indices, solution_rv);
}



