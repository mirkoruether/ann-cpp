#include "training_buffer.h"
#include "mat_arr_math.h"
#include "general_util.h"

using namespace annlib;

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
