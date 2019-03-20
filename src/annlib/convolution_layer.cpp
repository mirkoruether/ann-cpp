#include "mat_arr_math.h"
#include "convolution_layer.h"

void convolution_layer::init(std::mt19937* rnd)
{
	mat_random_gaussian(0.0f, 1.0f, rnd, &mask_biases);
	const auto factor = static_cast<fpt>(2.0 / std::sqrt(1.0 * p.mask_size()));
	mat_random_gaussian(0.0f, factor, rnd, &mask_weights);
}

void convolution_layer::prepare_buffer(layer_buffer* buf, gradient_based_optimizer* opt) const
{
	buf->add_single("grad_mb", 1, output_size);
	buf->add_single("grad_mw", input_size, output_size);

	opt->add_to_buffer("opt_mb", buf, 1, output_size);
	opt->add_to_buffer("opt_mw", buf, input_size, output_size);
}

void feed_forward_mask_element(const fpt* in_start, fpt* out, const fpt weight, unsigned fm_width, unsigned fm_heigth, unsigned im_width,
                               unsigned stride_x, unsigned stride_y)
{
	for (unsigned fm_row = 0; fm_row < fm_heigth; fm_row++)
	{
		const unsigned im_row = fm_row * stride_y;
		const fpt* im_row_p = in_start + im_row * im_width;
		fpt* fm_row_p = out + fm_row * fm_width;
		for (unsigned fm_col = 0; fm_col < fm_width; fm_col++)
		{
			const unsigned im_col = fm_col * stride_x;
			fm_row_p[fm_col] += weight * im_row_p[im_col];
		}
	}
}

void feed_forward_single(const fpt* in, fpt* out, const fpt* mask_weights, const fpt mask_bias, const conv_layer_hyperparameters& p)
{
	for (unsigned i = 0; i < p.feature_map_size(); i++)
	{
		out[i] = mask_bias;
	}

	for (unsigned mask_row = 0; mask_row < p.mask_height; mask_row++)
	{
		for (unsigned mask_col = 0; mask_col < p.mask_width; mask_col++)
		{
			const fpt* im_start = in + mask_col + mask_row * p.image_width;
			const fpt weight = mask_weights[mask_row * p.mask_width + mask_col];
			feed_forward_mask_element(im_start, out, weight, p.feature_map_width(), p.feature_map_height(), p.image_width,
			                          p.stride_length_x, p.stride_length_y);
		}
	}
}

void convolution_layer::feed_forward(const mat_arr& in, mat_arr* out) const
{
	for (unsigned mini_batch_element = 0; mini_batch_element < in.count; mini_batch_element++)
	{
		for (unsigned mask_no = 0; mask_no < p.map_count; mask_no++)
		{
			const fpt* single_in = in.start() + mini_batch_element * in.rows * in.cols;
			fpt* single_out = out->start() + mask_no * p.feature_map_size();
			const fpt* mask_w = mask_weights.start() + mask_no * p.mask_size();
			const fpt mask_b = mask_biases[mask_no];
			return feed_forward_single(single_in, single_out, mask_w, mask_b, p);
		}
	}
}

void convolution_layer::feed_forward_detailed(const mat_arr& in, mat_arr* out, layer_buffer* buf) const
{
	network_layer::feed_forward_detailed(in, out, buf);
}

void convolution_layer::backprop(const mat_arr& error, mat_arr* error_prev, layer_buffer* buf) const
{
	throw std::runtime_error("Backpropagation through convolutional layer not supported yet");
}

void convolution_layer::optimize(const mat_arr& error, gradient_based_optimizer* opt, layer_buffer* buf)
{

}

convolution_layer::convolution_layer(conv_layer_hyperparameters p)
	: network_layer(p.input_size(), p.output_size()),
	  p(p),
	  mask_biases(p.map_count, 1, 1),
	  mask_weights(p.map_count, p.mask_height, p.mask_width)
{
}

