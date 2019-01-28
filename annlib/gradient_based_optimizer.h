#ifndef GRADIENT_BASED_OPTIMIZER_H
#define GRADIENT_BASED_OPTIMIZER_H

#include "mat_arr.h"
#include "training_buffer.h"
using namespace linalg;

namespace annlib
{
	class gradient_based_optimizer
	{
	public:
		virtual ~gradient_based_optimizer() = default;

		virtual void add_to_buffer(std::string prefix, layer_buffer* buf, unsigned rows, unsigned cols) = 0;

		virtual void start();

		virtual void next_mini_batch();

		virtual void adjust(const mat_arr& gradient_noarr,
		                    mat_arr* target_noarr,
		                    std::string prefix,
		                    layer_buffer* buf) = 0;
	};

	class abstract_gradient_based_optimizer : public gradient_based_optimizer
	{
	public:
		const unsigned buffer_count;

		void add_to_buffer(std::string prefix, layer_buffer* buf, unsigned rows, unsigned cols) override;

		void adjust(const mat_arr& gradient_noarr,
		            mat_arr* target_noarr,
		            std::string prefix,
		            layer_buffer* buf) override;
	protected:
		explicit abstract_gradient_based_optimizer(unsigned buffer_count);

		virtual void adjust(const mat_arr& gradient_noarr,
		                    mat_arr* buffer,
		                    mat_arr* target_noarr) = 0;
	};

	class ordinary_sgd : public abstract_gradient_based_optimizer
	{
	public:
		explicit ordinary_sgd(float learning_rate);

		float learning_rate;

		void adjust(const mat_arr& gradient_noarr,
		            mat_arr* buffer,
		            mat_arr* target_noarr) override;
	};

	class momentum_sgd : public abstract_gradient_based_optimizer
	{
	public:
		explicit momentum_sgd(float learning_rate, float alpha);

		float learning_rate;
		float alpha;

		void adjust(const mat_arr& gradient_noarr,
		            mat_arr* buffer,
		            mat_arr* target_noarr) override;
	};

	class adam : public abstract_gradient_based_optimizer
	{
	public:
		adam();

		explicit adam(float alpha, float beta1, float beta2);

		float alpha;
		float beta1;
		float beta2;

		float beta1_pow_t;
		float beta2_pow_t;
		float alpha_t;

		void start() override;

		void next_mini_batch() override;

		void adjust(const mat_arr& gradient_noarr,
		            mat_arr* buffer,
		            mat_arr* target_noarr) override;
	};
}

#endif
