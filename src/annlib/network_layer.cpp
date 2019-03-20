#include "network_layer.h"

#include "mat_arr_math.h"

annlib::network_layer::network_layer(unsigned input_size, unsigned output_size)
	: input_size(input_size), output_size(output_size)
{
}

void annlib::network_layer::init(std::mt19937* rnd)
{
}

void annlib::network_layer::prepare_buffer(layer_buffer* buf)
{
}

void annlib::network_layer::feed_forward_detailed(const mat_arr& in, mat_arr* out, layer_buffer* buf) const
{
	feed_forward(in, out);
}

void annlib::network_layer::calculate_gradients(const mat_arr& error, layer_buffer* buf)
{
}

annlib::fully_connected_layer::fully_connected_layer(unsigned input_size, unsigned output_size,
                                                     std::shared_ptr<weight_norm_penalty> wnp)
	: network_layer(input_size, output_size),
	  weights_noarr(1, input_size, output_size),
	  biases_noarr(1, 1, output_size),
	  wnp(std::move(wnp))
{
}

void annlib::fully_connected_layer::feed_forward(const mat_arr& in, mat_arr* out) const
{
	mat_matrix_mul(in, weights_noarr, out);
	mat_element_wise_add(*out, biases_noarr, out);
}

void annlib::fully_connected_layer::init(std::mt19937* rnd)
{
	mat_random_gaussian(0.0f, 1.0f, rnd, &biases_noarr);
	const auto factor = static_cast<fpt>(2.0 / std::sqrt(1.0 * input_size));
	mat_random_gaussian(0.0f, factor, rnd, &weights_noarr);
}

void annlib::fully_connected_layer::prepare_buffer(layer_buffer* buf)
{
	buf->add_opt_target(&biases_noarr);
	buf->add_opt_target(&weights_noarr);
}

void annlib::fully_connected_layer::backprop(const mat_arr& error, mat_arr* error_prev, layer_buffer* buf) const
{
	mat_matrix_mul(error, weights_noarr, error_prev, transpose_B);
}

void calculate_gradient_weight_cpu(const mat_arr& previous_activation_rv,
                                   const mat_arr& error_rv,
                                   mat_arr* gradient_weight_noarr)
{
	const unsigned batch_entry_count = previous_activation_rv.count;
	mat_set_all(0.0f, gradient_weight_noarr);

	for (unsigned batch_entry_no = 0; batch_entry_no < batch_entry_count; batch_entry_no++)
	{
		mat_matrix_mul_add(previous_activation_rv.get_mat(batch_entry_no),
		                   error_rv.get_mat(batch_entry_no),
		                   gradient_weight_noarr,
		                   transpose_A);
	}

	mat_element_wise_div(*gradient_weight_noarr, static_cast<fpt>(batch_entry_count), gradient_weight_noarr);
}

void calculate_gradient_bias_cpu(const mat_arr& error_rv,
                                 mat_arr* gradient_bias_noarr_rv)
{
	const unsigned batch_entry_count = error_rv.count;
	mat_set_all(0.0f, gradient_bias_noarr_rv);

	for (unsigned batch_entry_no = 0; batch_entry_no < batch_entry_count; batch_entry_no++)
	{
		mat_element_wise_add(*gradient_bias_noarr_rv,
		                     error_rv.get_mat(batch_entry_no),
		                     gradient_bias_noarr_rv);
	}

	mat_element_wise_div(*gradient_bias_noarr_rv,
	                     static_cast<fpt>(batch_entry_count),
	                     gradient_bias_noarr_rv);
}

void annlib::fully_connected_layer::calculate_gradients(const mat_arr& error, layer_buffer* buf)
{
	mat_arr* grad_b = buf->get_grad_ptr(0);
	calculate_gradient_bias_cpu(error, grad_b);

	mat_arr* grad_w = buf->get_grad_ptr(1);
	calculate_gradient_weight_cpu(buf->in, error, grad_w);

	if (wnp != nullptr)
	{
		wnp->add_penalty_to_gradient(weights_noarr, grad_w);
	}
}

annlib::activation_layer::activation_layer(unsigned size, std::shared_ptr<activation_function> act)
	: network_layer(size, size),
	  act(std::move(act))
{
}

void annlib::activation_layer::prepare_buffer(layer_buffer* buf)
{
	buf->add_mini_batch_size("df", 1, output_size);
}

void annlib::activation_layer::feed_forward(const mat_arr& in, mat_arr* out) const
{
	act->apply(in, out);
}

void annlib::activation_layer::feed_forward_detailed(const mat_arr& in, mat_arr* out, layer_buffer* buf) const
{
	act->apply(in, out, buf->get_ptr("df"));
}

void annlib::activation_layer::backprop(const mat_arr& error, mat_arr* error_prev, layer_buffer* buf) const
{
	mat_element_wise_mul(error, buf->get_val("df"), error_prev);
}
