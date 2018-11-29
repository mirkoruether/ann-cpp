#include "neural_network.h"
#include "mat_arr_math.h"

void neural_network::feed_forward_internal(const mat_arr& input, mat_arr* result, std::vector<mat_arr>* buffer) const
{
	const size_t layer_count = biases_noarr_rv.size();
	const mat_arr* in = &input;

	for (unsigned i = 0; i < layer_count; i++)
	{
		mat_arr* out = i == layer_count - 1 ? result : &buffer->operator[](i);
		mat_matrix_mul(*in, weights_noarr[i], out);
		mat_element_by_element_operation(*out, biases_noarr_rv[i], out,
		                                 [&](const double mat_mul_result, const double bias)
		                                 {
			                                 return activation_function(mat_mul_result + bias);
		                                 });
		in = out;
	}
}

void neural_network::feed_forward_result_nonnull(const mat_arr& input, mat_arr* result,
                                                 std::vector<mat_arr>* buffer) const
{
	if (buffer == nullptr)
	{
		auto temp_buf = build_buffer(input.count);
		feed_forward_internal(input, result, &temp_buf);
	}
	else
	{
		feed_forward_internal(input, result, buffer);
	}
}

neural_network::neural_network(std::vector<mat_arr> weights_noarr,
                               std::vector<mat_arr> biases_noarr_rv,
                               std::function<double(double)> activation_function)
	: weights_noarr(move(weights_noarr)),
	  biases_noarr_rv(move(biases_noarr_rv)),
	  activation_function(move(activation_function))
{
}

std::vector<mat_arr> neural_network::build_buffer(unsigned count) const
{
	std::vector<mat_arr> buf;
	const size_t layer_count = biases_noarr_rv.size();
	for (unsigned i = 0; i < layer_count - 1; i++)
	{
		buf.emplace_back(count, 1, biases_noarr_rv[i].cols);
	}
	return buf;
}

mat_arr neural_network::feed_forward(const mat_arr& input, mat_arr* result, std::vector<mat_arr>* buffer) const
{
	const size_t layer_count = biases_noarr_rv.size();
	if (result == nullptr)
	{
		mat_arr temp_result(input.count, 1, biases_noarr_rv[layer_count - 1].cols);
		feed_forward_result_nonnull(input, &temp_result, buffer);
		return temp_result;
	}
	else
	{
		feed_forward_result_nonnull(input, result, buffer);
		return *result;
	}
}
