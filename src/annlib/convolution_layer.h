#ifndef ANN_CPP_CONVOLUTION_LAYER_H
#define ANN_CPP_CONVOLUTION_LAYER_H

#include "network_layer.h"

using namespace annlib;

namespace annlib
{
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
		const unsigned out_width = (in_width - mask_width) / stride_x + 1;
		const unsigned out_height = (in_height - mask_height) / stride_y + 1;
		const unsigned out_size = out_width * out_height;
		const unsigned out_size_total = out_size * out_count;

#pragma omp parallel for
		for (unsigned mb_el = 0; mb_el < mini_batch_size; mb_el++)
		{
#pragma omp parallel for
			for (unsigned no = 0; no < count; no++)
			{
				const unsigned i_in_single = mb_el * in_size_total + (in_count == 1 ? 0 : no * in_size);
				const unsigned i_out_single = mb_el * out_size_total + (out_count == 1 ? 0 : no * out_size);
				const unsigned i_mask_single = no * mask_size;

				for (unsigned mask_row = 0; mask_row < mask_height; mask_row++)
				{
					for (unsigned out_row = 0; out_row < out_height; out_row++)
					{
						for (unsigned mask_col = 0; mask_col < mask_width; mask_col++)
						{
							const unsigned i_mask = i_mask_single + mask_row * mask_width + mask_col;
							const unsigned i_in_start = i_in_single + mask_col + mask_row * in_width;

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

#pragma omp parallel for
		for (unsigned mb_el = 0; mb_el < mini_batch_size; mb_el++)
		{
#pragma omp parallel for
			for (unsigned no = 0; no < count; no++)
			{
				unsigned i_single = mb_el * size_total + no * size;
				for (unsigned row = 0; row < height; row++)
				{
					for (unsigned col = 0; col < width; col++)
					{
						unsigned i = i_single + row * width + col;
						f(no, i);
					}
				}
			}
		}
	}

	struct conv_layer_hyperparameters
	{
		unsigned map_count;

		unsigned image_width;
		unsigned image_height;

		unsigned mask_width;
		unsigned mask_height;

		unsigned stride_length_x;
		unsigned stride_length_y;

		unsigned input_size() const
		{
			return image_width * image_height;
		};

		unsigned mask_size() const
		{
			return mask_width * mask_height;
		};

		unsigned feature_map_width() const
		{
			return (image_width - mask_width) / stride_length_x + 1;
		};

		unsigned feature_map_height() const
		{
			return (image_height - mask_height) / stride_length_y + 1;
		};

		unsigned feature_map_size() const
		{
			return feature_map_width() * feature_map_height();
		};

		unsigned output_size() const
		{
			return feature_map_size() * map_count;
		};
	};

	class convolution_layer : public network_layer
	{
	public:
		const conv_layer_hyperparameters p;

		//private:
		mat_arr mask_weights;
		mat_arr mask_biases;

		//public:
		void init(std::mt19937* rnd) override;

		void prepare_buffer(layer_buffer* buf) override;

		void feed_forward(const mat_arr& in, mat_arr* out) const override;

		void feed_forward_detailed(const mat_arr& in, mat_arr* out, layer_buffer* buf) const override;

		void backprop(const mat_arr& error, mat_arr* error_prev, layer_buffer* buf) const override;

		void calculate_gradients(const mat_arr& error, layer_buffer* buf) override;

		explicit convolution_layer(conv_layer_hyperparameters p);

		~convolution_layer() override = default;
	};

	struct pooling_layer_hyperparameters
	{
		unsigned count;

		unsigned in_width;
		unsigned in_height;

		unsigned mask_width;
		unsigned mask_height;

		unsigned in_size() const
		{
			return in_width * in_height;
		};

		unsigned in_size_total() const
		{
			return in_size() * count;
		};

		unsigned mask_size() const
		{
			return mask_width * mask_height;
		};

		unsigned out_width() const
		{
			return in_width / mask_width;
		};

		unsigned out_height() const
		{
			return in_height / mask_height;
		};

		unsigned out_size() const
		{
			return out_width() * out_height();
		};

		unsigned out_size_total() const
		{
			return out_size() * count;
		}
	};

	class max_pooling_layer : public network_layer
	{
	public:
		const pooling_layer_hyperparameters p;

		void prepare_buffer(layer_buffer* buf) override;

		void feed_forward(const mat_arr& in, mat_arr* out) const override;

		void feed_forward_detailed(const mat_arr& in, mat_arr* out, layer_buffer* buf) const override;

		void backprop(const mat_arr& error, mat_arr* error_prev, layer_buffer* buf) const override;

		explicit max_pooling_layer(pooling_layer_hyperparameters p);

		~max_pooling_layer() override = default;
	};

	class average_pooling_layer : public network_layer
	{
	public:
		const pooling_layer_hyperparameters p;

		void feed_forward(const mat_arr& in, mat_arr* out) const override;

		void backprop(const mat_arr& error, mat_arr* error_prev, layer_buffer* buf) const override;

		explicit average_pooling_layer(pooling_layer_hyperparameters p);

		~average_pooling_layer() override = default;
	};
}

#endif //ANN_CPP_CONVOLUTION_LAYER_H
