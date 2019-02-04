#include "output_layer.h"

#include "mat_arr_math.h"
#include "mat_arr_math_t.h"

output_layer::output_layer(unsigned input_size, unsigned output_size)
	: network_layer(input_size, output_size)
{
}

void output_layer::backprop(const mat_arr& error, mat_arr* error_prev, layer_buffer* buf) const
{
	calculate_error_prev_layer(buf->out, error, error_prev);
}

logistic_act_cross_entropy_costs::logistic_act_cross_entropy_costs(unsigned size)
	: output_layer(size, size),
	  act(std::make_shared<logistic_activation_function>())
{
}

void logistic_act_cross_entropy_costs::feed_forward(const mat_arr& in, mat_arr* out) const
{
	act->apply(in, out);
}

fpt logistic_act_cross_entropy_costs::calculate_costs(const mat_arr& net_output, const mat_arr& solution) const
{
	const size_t size = net_output.size();
	if (solution.size() != size)
	{
		throw std::runtime_error("sizes differ");
	}

	const fpt* no_element = net_output.start();
	const fpt* s_element = solution.start();

	fpt result = 0;
	for (unsigned i = 0; i < size; i++)
	{
		const fpt a = *no_element;
		const fpt y = *s_element;
		result += y * std::log(a) + (1 - y) * std::log(1 - a);

		no_element++;
		s_element++;
	}
	return -result;
}

void logistic_act_cross_entropy_costs::calculate_error_prev_layer(const mat_arr& net_output,
                                                                  const mat_arr& solution,
                                                                  mat_arr* error_prev) const
{
	mat_element_wise_sub(net_output, solution, error_prev);
}

softmax_act_cross_entropy_costs::softmax_act_cross_entropy_costs(unsigned size)
	: output_layer(size, size)
{
}

void softmax_act_cross_entropy_costs::feed_forward(const mat_arr& in, mat_arr* out) const
{
	const unsigned count = in.count;
	for (unsigned i = 0; i < count; i++)
	{
		const mat_arr in_mat = in.get_mat(i);
		mat_arr out_mat = out->get_mat(i);

		const fpt max = mat_max(in_mat);
		mat_element_wise_operation(in_mat, &out_mat, [=](fpt z)
		{
			return std::exp(z - max);
		});

		const fpt den = mat_sum(out_mat);
		if (std::isfinite(den))
		{
			mat_element_wise_div(out_mat, den, &out_mat);
		}
		mat_element_wise_operation(out_mat, &out_mat, [=](fpt o)
		{
			const fpt res = o / den;
			return std::isfinite(res) ? res : 0.9999f;
		});

		out_mat.assert_only_real();
	}
}

fpt softmax_act_cross_entropy_costs::calculate_costs(const mat_arr& net_output, const mat_arr& solution) const
{
	return 0.0f;
}

void softmax_act_cross_entropy_costs::calculate_error_prev_layer(const mat_arr& net_output,
                                                                 const mat_arr& solution,
                                                                 mat_arr* error_prev) const
{
	mat_element_wise_sub(net_output, solution, error_prev);
}
