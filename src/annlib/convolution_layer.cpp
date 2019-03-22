#include "mat_arr_math.h"
#include <functional>
#include "convolution_layer.h"

namespace annlib
{
	template <typename Fc>
	void convolve(unsigned mini_batch_size, conv_layer_hyperparameters p, Fc f)
	{
		convolve(mini_batch_size, 1, p.map_count, p.image_width, p.image_height, p.mask_width, p.mask_height, p.stride_length_x, p.stride_length_y,
		         f);
	}

	template <typename Fc>
	void iterate_feature_maps(unsigned mini_batch_size, conv_layer_hyperparameters p, Fc f)
	{
		iterate_image_set(mini_batch_size, p.map_count, p.feature_map_width(), p.feature_map_height(), f);
	}

	template <typename Fc>
	void convolve(unsigned mini_batch_size, pooling_layer_hyperparameters p, Fc f)
	{
		convolve(mini_batch_size, p.count, p.count, p.in_width, p.in_height, p.mask_width, p.mask_height, p.mask_width, p.mask_height, f);
	}
}

void convolution_layer::init(std::mt19937* rnd)
{
	mat_random_gaussian(0.0f, 1.0f, rnd, &mask_biases);
	const auto factor = static_cast<fpt>(2.0 / std::sqrt(1.0 * p.mask_size()));
	mat_random_gaussian(0.0f, factor, rnd, &mask_weights);
}

void convolution_layer::prepare_buffer(layer_buffer* buf)
{
	buf->add_opt_target(&mask_biases);
	buf->add_opt_target(&mask_weights);
}

void convolution_layer::feed_forward(const mat_arr& in, mat_arr* out) const
{
	if (in.cols != p.input_size())
	{
		throw std::runtime_error("Wrong input size");
	}

	if (out->cols != p.output_size())
	{
		throw std::runtime_error("Wrong output size");
	}

	const fpt* in_ptr = in.start();
	fpt* out_ptr = out->start();
	const fpt* mw_ptr = mask_weights.start();
	const fpt* mb_ptr = mask_biases.start();

	iterate_feature_maps(in.count, p, std::function<void(unsigned, unsigned)>
		([&](unsigned mask_no, unsigned i_out)
		 {
			 out_ptr[i_out] = mb_ptr[mask_no];
		 }));

	convolve(in.count, p, std::function<void(unsigned, unsigned, unsigned)>
		([&](unsigned i_in, unsigned i_out, unsigned i_mask)
		 {
			 out_ptr[i_out] += in_ptr[i_in] * mw_ptr[i_mask];
		 }));
}

void convolution_layer::feed_forward_detailed(const mat_arr& in, mat_arr* out, layer_buffer* buf) const
{
	network_layer::feed_forward_detailed(in, out, buf);
}

void convolution_layer::backprop(const mat_arr& error, mat_arr* error_prev, layer_buffer* buf) const
{
	throw std::runtime_error("Backpropagation through convolutional layer not supported yet");
}

void convolution_layer::calculate_gradients(const mat_arr& error, layer_buffer* buf)
{
	const mat_arr in = buf->in;
	mat_arr* mb_grad = buf->get_grad_ptr(0);
	mat_arr* mw_grad = buf->get_grad_ptr(1);
	mat_set_all(0.0f, mb_grad);
	mat_set_all(0.0f, mw_grad);

	const fpt* in_ptr = in.start();
	const fpt* err_ptr = error.start();
	fpt* mb_grad_ptr = mb_grad->start();
	fpt* mw_grad_ptr = mw_grad->start();

	iterate_feature_maps(in.count, p, std::function<void(unsigned, unsigned)>
		([&](unsigned mask_no, unsigned i_err)
		 {
			 mb_grad_ptr[mask_no] += err_ptr[i_err];
		 }));

	convolve(in.count, p, std::function<void(unsigned, unsigned, unsigned)>
		([&](unsigned i_in, unsigned i_err, unsigned i_mask)
		 {
			 mw_grad_ptr[i_mask] += in_ptr[i_in] * err_ptr[i_err];
		 }));

	const fpt mini_batch_size = static_cast<fpt>(in.count);
	mat_element_wise_div(*mb_grad, mini_batch_size, mb_grad);
	mat_element_wise_div(*mw_grad, mini_batch_size, mw_grad);
}

convolution_layer::convolution_layer(conv_layer_hyperparameters p)
	: network_layer(p.input_size(), p.output_size()), p(p),
	  mask_weights(p.map_count, p.mask_height, p.mask_width),
	  mask_biases(p.map_count, 1, 1)
{
}

void max_pooling_layer::prepare_buffer(layer_buffer* buf)
{
	buf->add_mini_batch_size("df", 1, p.in_size_total());
}

void max_pooling_layer::feed_forward(const mat_arr& in, mat_arr* out) const
{
	mat_set_all(0.0f, out);

	if (in.cols != p.in_size_total())
	{
		throw std::runtime_error("Wrong input size");
	}

	if (out->cols != p.out_size_total())
	{
		throw std::runtime_error("Wrong output size");
	}

	const fpt* in_ptr = in.start();
	fpt* out_ptr = out->start();

	convolve(in.count, p, std::function<void(unsigned, unsigned, unsigned)>
		([&](unsigned i_in, unsigned i_out, unsigned i_mask)
		 {
			 if (out_ptr[i_out] < in_ptr[i_in])
			 {
				 out_ptr[i_out] = in_ptr[i_in];
			 }
		 }));
}

void max_pooling_layer::feed_forward_detailed(const mat_arr& in, mat_arr* out, layer_buffer* buf) const
{
	feed_forward(in, out);

	const fpt* in_ptr = in.start();
	const fpt* out_ptr = out->start();
	fpt* df_ptr = buf->get_ptr("df")->start();

	convolve(in.count, p, std::function<void(unsigned, unsigned, unsigned)>
		([&](unsigned i_in, unsigned i_out, unsigned i_mask)
		 {
			 df_ptr[i_in] = out_ptr[i_out] == in_ptr[i_in] ? 1.0f : 0.0f;
		 }));
}

void max_pooling_layer::backprop(const mat_arr& error, mat_arr* error_prev, layer_buffer* buf) const
{
	const fpt* err_ptr = error.start();
	const fpt* df_ptr = buf->get_ptr("df")->start();
	fpt* res_ptr = error_prev->start();

	convolve(error.count, p, std::function<void(unsigned, unsigned, unsigned)>
		([&](unsigned i_in, unsigned i_out, unsigned i_mask)
		 {
			 res_ptr[i_in] = df_ptr[i_in] * err_ptr[i_out];
		 }));
}

max_pooling_layer::max_pooling_layer(pooling_layer_hyperparameters p)
	: network_layer(p.in_size_total(), p.out_size_total()), p(p)
{
}
