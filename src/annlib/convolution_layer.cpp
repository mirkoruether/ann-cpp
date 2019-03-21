#include "mat_arr_math.h"
#include "convolution_layer.h"

template <typename Fc>
void convolve(const unsigned mini_batch_size, const unsigned in_count, const unsigned out_count,
              const unsigned in_width, const unsigned in_height, const unsigned mask_width, const unsigned mask_height,
              const unsigned stride_x, const unsigned stride_y, Fc f)
{
	if (in_count != 1 && out_count != 1 && in_count != out_count)
	{
		throw std::runtime_error("Invalid counts");
	}

	const unsigned count = std::max(in_count, out_count);
	const unsigned in_size = in_width * in_height;
	const unsigned in_size_total = in_size * in_count;
	const unsigned mask_size = mask_width * mask_height;
	const unsigned out_width = (in_width - (mask_width - 1)) / stride_x;
	const unsigned out_height = (in_height - (mask_height - 1)) / stride_y;
	const unsigned out_size = out_width * out_height;
	const unsigned out_size_total = out_size * out_count;

	for (unsigned mb_el = 0; mb_el < mini_batch_size; mb_el++)
	{
		for (unsigned no = 0; no < count; no++)
		{
			const unsigned i_in_single = mb_el * in_size_total + (in_count == 1 ? 0 : no * in_size);
			const unsigned i_out_single = mb_el * out_size_total + (out_count == 1 ? 0 : no * out_size);
			const unsigned i_mask_single = no * mask_size;

			for (unsigned mask_row = 0; mask_row < mask_height; mask_row++)
			{
				for (unsigned mask_col = 0; mask_col < mask_width; mask_col++)
				{
					const unsigned i_mask = i_mask_single + mask_row * mask_width + mask_col;
					const unsigned i_in_start = i_in_single + mask_col + mask_row * in_width;
					for (unsigned out_row = 0; out_row < out_height; out_row++)
					{
						for (unsigned out_col = 0; out_col < out_width; out_col++)
						{
							const unsigned i_in = i_in_start + out_row * stride_y * in_width + out_col * stride_x;
							const unsigned i_out = i_out_single + out_row * out_width + out_col;
							f(i_in, i_out, i_mask);
						}
					}
				}
			}
		}
	}
}

template <typename Fc>
void iterate_image_set(unsigned mini_batch_size, unsigned count, unsigned width, unsigned height, Fc f)
{
	const unsigned size = width * height;
	const unsigned size_total = size * count;

	for (unsigned mb_el = 0; mb_el < mini_batch_size; mb_el++)
	{
		for (unsigned no = 0; no < count; no++)
		{
			unsigned i_single = mb_el * size_total + no * size;
			for (unsigned row = 0; row < width; row++)
			{
				for (unsigned col = 0; col < height; col++)
				{
					unsigned i = i_single + row * width + col;
					f(no, i);
				}
			}
		}
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

void feed_forward_mask_element(const fpt* in_start, fpt* out, const fpt weight, unsigned fm_width, unsigned fm_height, unsigned im_width,
                               unsigned stride_x, unsigned stride_y)
{
	for (unsigned fm_row = 0; fm_row < fm_height; fm_row++)
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
	if (in.cols != p.input_size())
	{
		throw std::runtime_error("Wrong input size");
	}

	if (out->cols != p.output_size())
	{
		throw std::runtime_error("Wrong output size");
	}

	for (unsigned mini_batch_element = 0; mini_batch_element < in.count; mini_batch_element++)
	{
		for (unsigned mask_no = 0; mask_no < p.map_count; mask_no++)
		{
			const fpt* single_in = in.start() + mini_batch_element * p.input_size();
			fpt* single_out = out->start() + mask_no * p.feature_map_size() + mini_batch_element * p.output_size();
			const fpt* mask_w = mask_weights.start() + mask_no * p.mask_size();
			const fpt mask_b = mask_biases[mask_no];
			feed_forward_single(single_in, single_out, mask_w, mask_b, p);
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

fpt calculate_gradient_mask_element(const fpt* in_start, const fpt* err, unsigned fm_width, unsigned fm_height, unsigned im_width,
                                    unsigned stride_x, unsigned stride_y)
{
	fpt result = 0.0f;
	for (unsigned fm_row = 0; fm_row < fm_height; fm_row++)
	{
		const unsigned im_row = fm_row * stride_y;
		const fpt* im_row_p = in_start + im_row * im_width;
		const fpt* fm_row_p = err + fm_row * fm_width;
		for (unsigned fm_col = 0; fm_col < fm_width; fm_col++)
		{
			const unsigned im_col = fm_col * stride_x;
			result += im_row_p[im_col] * fm_row_p[fm_col];
		}
	}
	return result;
}

void add_to_gradient_single(const fpt* in, const fpt* err, fpt* mb_grad, fpt* mw_grad,
                            const conv_layer_hyperparameters& p)
{
	*mb_grad = 0.0f;
	for (unsigned i = 0; i < p.feature_map_size(); i++)
	{
		*mb_grad += err[i];
	}

	for (unsigned mask_row = 0; mask_row < p.mask_height; mask_row++)
	{
		for (unsigned mask_col = 0; mask_col < p.mask_width; mask_col++)
		{
			const fpt* im_start = in + mask_col + mask_row * p.image_width;
			fpt g = calculate_gradient_mask_element(im_start, err, p.feature_map_width(), p.feature_map_height(),
			                                        p.image_width, p.stride_length_x, p.stride_length_y);
			mw_grad[mask_row * p.mask_width + mask_col] = g;
		}
	}
}

void convolution_layer::calculate_gradients(const mat_arr& error, layer_buffer* buf)
{
	const mat_arr in = buf->in;
	mat_arr* mb_grad = buf->get_grad_ptr(0);
	mat_arr* mw_grad = buf->get_grad_ptr(1);
	for (unsigned mini_batch_element = 0; mini_batch_element < in.count; mini_batch_element++)
	{
		for (unsigned mask_no = 0; mask_no < p.map_count; mask_no++)
		{
			const fpt* single_in = in.start() + mini_batch_element * p.input_size();
			const fpt* single_err = error.start() + mask_no * p.feature_map_size() + mini_batch_element * p.output_size();
			fpt* single_mb_grad = mb_grad->start() + mask_no;
			fpt* single_mw_grad = mw_grad->start() + mask_no * p.mask_size();
			add_to_gradient_single(single_in, single_err, single_mb_grad, single_mw_grad, p);
		}
	}

	const fpt mini_batch_count = static_cast<fpt>(in.count);
	mat_element_wise_div(*mb_grad, mini_batch_count, mb_grad);
	mat_element_wise_div(*mw_grad, mini_batch_count, mw_grad);
}

convolution_layer::convolution_layer(conv_layer_hyperparameters p)
	: network_layer(p.input_size(), p.output_size()), p(p),
	  mask_weights(p.map_count, p.mask_height, p.mask_width),
	  mask_biases(p.map_count, 1, 1)
{
}

