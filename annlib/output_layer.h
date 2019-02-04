#ifndef OUTPUT_LAYER_H
#define OUTPUT_LAYER_H

#include "mat_arr.h"
#include "network_layer.h"
#include "activation_function.h"

using namespace linalg;
using namespace annlib;

namespace annlib
{
	class output_layer : public network_layer
	{
	public:
		output_layer(unsigned input_size, unsigned output_size);

		virtual ~output_layer() = default;

		void backprop(const mat_arr& error, mat_arr* error_prev, layer_buffer* buf) const override;

		virtual float calculate_costs(const mat_arr& net_output, const mat_arr& solution) const = 0;

		virtual void calculate_error_prev_layer(const mat_arr& net_output,
		                                        const mat_arr& solution,
		                                        mat_arr* error_prev) const = 0;
	};

	class logistic_act_cross_entropy_costs : public output_layer
	{
	private:
		std::shared_ptr<activation_function> act;;

	public:
		logistic_act_cross_entropy_costs(unsigned size);

		void feed_forward(const mat_arr& in, mat_arr* out) const override;

		float calculate_costs(const mat_arr& net_output, const mat_arr& solution) const override;

		void calculate_error_prev_layer(const mat_arr& net_output,
		                                const mat_arr& solution,
		                                mat_arr* error_prev) const
		override;
	};
}
#endif // OUTPUT_LAYER_H
