//
// Created by Mirko on 12.02.2019.
//

#include "convolution_layer.h"

void convolution_layer::init(std::mt19937* rnd)
{
}

void convolution_layer::prepare_buffer(layer_buffer* buf, gradient_based_optimizer* opt) const
{
	network_layer::prepare_buffer(buf, opt);
}

void convolution_layer::feed_forward(const mat_arr& in, mat_arr* out) const
{

}

void convolution_layer::feed_forward_detailed(const mat_arr& in, mat_arr* out, layer_buffer* buf) const
{
	network_layer::feed_forward_detailed(in, out, buf);
}

void convolution_layer::backprop(const mat_arr& error, mat_arr* error_prev, layer_buffer* buf) const
{

}

void convolution_layer::optimize(const mat_arr& error, gradient_based_optimizer* opt, layer_buffer* buf)
{
	network_layer::optimize(error, opt, buf);
}

convolution_layer::convolution_layer(unsigned input_size, unsigned output_size) : network_layer(input_size, output_size)
{

}

convolution_layer::~convolution_layer()
{

}
