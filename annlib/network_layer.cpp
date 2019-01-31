#include "network_layer.h"
#include <utility>

#include "mat_arr_math.h"
#include "gradient_based_optimizer.h"

annlib::network_layer::network_layer(unsigned input_size, unsigned output_size)
	: input_size(input_size), output_size(output_size)
{
}

annlib::fully_connected_layer::fully_connected_layer(unsigned input_size, unsigned output_size,
                                                     std::shared_ptr<activation_function> act,
                                                     std::shared_ptr<weight_norm_penalty> wnp)
	: network_layer(input_size, output_size),
	  weights_noarr(1, input_size, output_size),
	  biases_noarr(1, 1, output_size),
	  act(std::move(act)),
	  wnp(std::move(wnp))
{
}

void annlib::fully_connected_layer::feed_forward(const mat_arr& in, mat_arr* out) const
{
	mat_matrix_mul(in, weights_noarr, out);
	mat_element_wise_add(*out, biases_noarr, out);
	act->apply(*out, out);
}

void annlib::fully_connected_layer::init(std::mt19937* rnd)
{
	mat_random_gaussian(0.0f, 1.0f, rnd, &biases_noarr);
	const auto factor = static_cast<float>(2.0 / std::sqrt(1.0 * input_size));
	mat_random_gaussian(0.0f, factor, rnd, &weights_noarr);
}

void annlib::fully_connected_layer::prepare_buffer(layer_buffer* buf, gradient_based_optimizer* opt) const
{
	buf->add_mini_batch_size("wi", 1, output_size);
	buf->add_mini_batch_size("out_df", 1, output_size);
	buf->add_mini_batch_size("err", 1, output_size);

	buf->add_single("grad_b", 1, output_size);
	buf->add_single("grad_w", input_size, output_size);

	opt->add_to_buffer("opt_b", buf, 1, output_size);
	opt->add_to_buffer("opt_w", buf, input_size, output_size);
}

void annlib::fully_connected_layer::feed_forward_detailed(const mat_arr& in,
                                                          mat_arr* out,
                                                          layer_buffer* buf) const
{
	mat_matrix_mul(in, weights_noarr, buf->get_ptr("wi"));
	mat_element_wise_add(buf->get_val("wi"), biases_noarr, buf->get_ptr("wi"));

	act->apply(buf->get_val("wi"), out);
	act->apply_derivative(buf->get_val("wi"), buf->get_ptr("out_df"));
}

void annlib::fully_connected_layer::prepare_optimization(const mat_arr& backprop_term, layer_buffer* buf) const
{
	mat_element_wise_mul(backprop_term, buf->get_val("out_df"), buf->get_ptr("err"));
}

void annlib::fully_connected_layer::backprop(mat_arr* backprop_term_prev, layer_buffer* buf) const
{
	mat_matrix_mul(buf->get_val("err"), weights_noarr, backprop_term_prev, transpose_B);
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

	mat_element_wise_div(*gradient_weight_noarr, static_cast<float>(batch_entry_count), gradient_weight_noarr);
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
	                     static_cast<float>(batch_entry_count),
	                     gradient_bias_noarr_rv);
}

void annlib::fully_connected_layer::optimize(annlib::gradient_based_optimizer* opt, layer_buffer* buf)
{
	//TODO Parallel 
	mat_arr* grad_w = buf->get_ptr("grad_w");
	calculate_gradient_weight_cpu(*buf->in, buf->get_val("err"), grad_w);

	if (wnp != nullptr)
	{
		wnp->add_penalty_to_gradient(weights_noarr, grad_w);
	}

	opt->adjust(*grad_w, &weights_noarr, "opt_w", buf);

	mat_arr* grad_b = buf->get_ptr("grad_b");
	calculate_gradient_bias_cpu(buf->get_val("err"), grad_b);
	opt->adjust(*grad_b, &biases_noarr, "opt_b", buf);
}
