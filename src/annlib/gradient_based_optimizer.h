#ifndef GRADIENT_BASED_OPTIMIZER_H
#define GRADIENT_BASED_OPTIMIZER_H

#include <vector>
#include "mat_arr.h"
#include "training_buffer.h"

using namespace linalg;

namespace annlib
{
	class gradient_based_optimizer
	{
	public:
		const unsigned buffer_count;

		virtual ~gradient_based_optimizer() = default;

		virtual void start();

		virtual void next_mini_batch();

		virtual void adjust(opt_target target);

	protected:
		explicit gradient_based_optimizer(unsigned buffer_count);

		virtual void adjust(const mat_arr& gradient, std::vector<mat_arr> buffer, mat_arr* target) = 0;
	};

	class ordinary_sgd : public gradient_based_optimizer
	{
	public:
		explicit ordinary_sgd(fpt learning_rate);

		fpt learning_rate;

		void adjust(const mat_arr& gradient, std::vector<mat_arr> buffer, mat_arr* target) override;
	};

	class momentum_sgd : public gradient_based_optimizer
	{
	public:
		explicit momentum_sgd(fpt learning_rate, fpt alpha);

		fpt learning_rate;
		fpt alpha;

		void adjust(const mat_arr& gradient, std::vector<mat_arr> buffer, mat_arr* target) override;
	};

	class adam : public gradient_based_optimizer
	{
	public:
		adam();

		explicit adam(fpt alpha, fpt beta1, fpt beta2);

		fpt alpha;
		fpt beta1;
		fpt beta2;

		fpt beta1_pow_t;
		fpt beta2_pow_t;
		fpt alpha_t;

		void start() override;

		void next_mini_batch() override;

		void adjust(const mat_arr& gradient, std::vector<mat_arr> buffer, mat_arr* target) override;
	};
}

#endif
