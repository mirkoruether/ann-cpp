#ifndef NETWORK_LAYER_H
#define NETWORK_LAYER_H
#include "mat_arr.h"
#include "training_buffer.h"

using namespace linalg;

namespace annlib
{
	class network_layer
	{
	protected:
		~network_layer() = default;

	public:
		virtual void feed_forward(const mat_arr& in, mat_arr* out) = 0;

		virtual void prepare_buffer(layer_buffer* buf) = 0;

		virtual void feed_forward_detailed(const mat_arr& in, mat_arr* out, layer_buffer* buf) = 0;

		virtual void backprop_error(const mat_arr& error_current, mat_arr* error_prev, layer_buffer* buf) = 0;

		virtual void optimize(const mat_arr& error_current, layer_buffer* buf) = 0;
	};

	class fully_connected_layer : network_layer
	{
		//TODO implement
	};
}
#endif // NETWORK_LAYER_H
