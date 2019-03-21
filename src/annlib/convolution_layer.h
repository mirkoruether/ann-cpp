#ifndef ANN_CPP_CONVOLUTION_LAYER_H
#define ANN_CPP_CONVOLUTION_LAYER_H

#include "network_layer.h"

using namespace annlib;

namespace annlib
{
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
			return (image_width - (mask_width - 1)) / stride_length_x;
		};

		unsigned feature_map_height() const
		{
			return (image_height - (mask_height - 1)) / stride_length_y;
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

	private:
		mat_arr mask_weights;
		mat_arr mask_biases;

	public:
		void init(std::mt19937* rnd) override;

		void prepare_buffer(layer_buffer* buf) override;

		void feed_forward(const mat_arr& in, mat_arr* out) const override;

		void feed_forward_detailed(const mat_arr& in, mat_arr* out, layer_buffer* buf) const override;

		void backprop(const mat_arr& error, mat_arr* error_prev, layer_buffer* buf) const override;

		void calculate_gradients(const mat_arr& error, layer_buffer* buf) override;

		convolution_layer(conv_layer_hyperparameters p);

		~convolution_layer() override = default;
	};

	struct pooling_layer_hyperparameters
	{

	};

	class max_pooling_layer : public network_layer
	{

	};
}

#endif //ANN_CPP_CONVOLUTION_LAYER_H
