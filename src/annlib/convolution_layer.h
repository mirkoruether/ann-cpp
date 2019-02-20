#ifndef ANN_CPP_CONVOLUTION_LAYER_H
#define ANN_CPP_CONVOLUTION_LAYER_H

#include "network_layer.h"

using namespace annlib;

namespace annlib
{
	class conv_layer_hyperparameters
	{
		unsigned mask_width;
		unsigned mask_height;

		unsigned x_start;
		unsigned y_start;


	};

	class convolution_layer : public network_layer
	{
	private:
		mat_arr mask_weights;
		mat_arr mask_biases;


	public:
		void init(std::mt19937* rnd) override;

		void prepare_buffer(layer_buffer* buf, gradient_based_optimizer* opt) const override;

		void feed_forward(const mat_arr& in, mat_arr* out) const override;

		void feed_forward_detailed(const mat_arr& in, mat_arr* out, layer_buffer* buf) const override;

		void backprop(const mat_arr& error, mat_arr* error_prev, layer_buffer* buf) const override;

		void optimize(const mat_arr& error, gradient_based_optimizer* opt, layer_buffer* buf) override;

	protected:
		convolution_layer(unsigned input_size, unsigned output_size);

		~convolution_layer() override = default;
	};
}

#endif //ANN_CPP_CONVOLUTION_LAYER_H
