#ifndef NETWORK_LAYER_H
#define NETWORK_LAYER_H
#include "mat_arr.h"
#include "training_buffer.h"
#include "activation_function.h"
#include "gradient_based_optimizer.h"
#include "weight_norm_penalty.h"
#include <random>
#include <memory>

using namespace linalg;

namespace annlib
{
	class network_layer
	{
	protected:
		network_layer(unsigned input_size, unsigned output_size);
		~network_layer() = default;

	public:
		const unsigned input_size;
		const unsigned output_size;

		virtual void feed_forward(const mat_arr& in, mat_arr* out) const = 0;

		virtual void init(std::mt19937* rnd) = 0;

		virtual void prepare_buffer(layer_buffer* buf, gradient_based_optimizer* opt) const = 0;

		virtual void feed_forward_detailed(const mat_arr& in, mat_arr* out, layer_buffer* buf) const = 0;

		virtual void prepare_optimization(const mat_arr& backprop_term, layer_buffer* buf) const = 0;

		virtual void backprop(mat_arr* backprop_term_prev, layer_buffer* buf) const = 0;

		virtual void optimize(annlib::gradient_based_optimizer* opt, layer_buffer* buf) = 0;
	};

	class fully_connected_layer : public network_layer
	{
	private:
		mat_arr weights_noarr;
		mat_arr biases_noarr;

		std::shared_ptr<activation_function> act;
		std::shared_ptr<weight_norm_penalty> wnp;

	public:
		virtual ~fully_connected_layer() = default;

		fully_connected_layer(unsigned input_size, unsigned output_size,
		                      std::shared_ptr<activation_function> act,
		                      std::shared_ptr<weight_norm_penalty> wnp = nullptr);

		void feed_forward(const mat_arr& in, mat_arr* out) const override;

		void init(std::mt19937* rnd) override;

		void prepare_buffer(layer_buffer* buf, gradient_based_optimizer* opt) const override;

		void feed_forward_detailed(const mat_arr& in, mat_arr* out, layer_buffer* buf) const override;

		void prepare_optimization(const mat_arr& backprop_term, layer_buffer* buf) const override;

		void backprop(mat_arr* backprop_term_prev, layer_buffer* buf) const override;

		void optimize(annlib::gradient_based_optimizer* opt, layer_buffer* buf) override;
	};
}
#endif // NETWORK_LAYER_H
