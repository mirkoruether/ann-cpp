#ifndef GRADIENT_BASED_OPTIMIZER_H
#define GRADIENT_BASED_OPTIMIZER_H

#include "mat_arr.h"
using namespace linalg;

namespace annlib
{
	enum adjust_target
	{
		weights,
		biases
	};

	class gradient_based_optimizer
	{
	public:
		virtual ~gradient_based_optimizer() = default;

		virtual void init(const std::vector<unsigned>& sizes);

		virtual void next_mini_batch();

		virtual void adjust(const mat_arr& gradient_noarr,
		                    mat_arr* target_noarr,
		                    const adjust_target& at,
		                    unsigned layer_no) = 0;
	};

	class abstract_gradient_based_optimizer : public gradient_based_optimizer
	{
	public:
		const unsigned buffer_count;
		std::vector<mat_arr> weights_buffers;
		std::vector<mat_arr> biases_buffers;

		void init(const std::vector<unsigned>& sizes) override;

		void adjust(const mat_arr& gradient_noarr,
		            mat_arr* target_noarr,
		            const adjust_target& at,
		            unsigned layer_no) override;
	protected:
		explicit abstract_gradient_based_optimizer(unsigned buffer_count);

		virtual void adjust(const mat_arr& gradient_noarr,
		                    mat_arr* buffer,
		                    mat_arr* target_noarr) = 0;
	};

	class ordinary_sgd : public abstract_gradient_based_optimizer
	{
	public:
		explicit ordinary_sgd(double learning_rate);

		double learning_rate;

		void adjust(const mat_arr& gradient_noarr,
		            mat_arr* buffer,
		            mat_arr* target_noarr) override;
	};

	class momentum_sgd : public abstract_gradient_based_optimizer
	{
	public:
		explicit momentum_sgd(double learning_rate, double alpha);

		double learning_rate;
		double alpha;

		void adjust(const mat_arr& gradient_noarr,
		            mat_arr* buffer,
		            mat_arr* target_noarr) override;
	};

	class adam : public abstract_gradient_based_optimizer
	{
	public:
		explicit adam(double alpha, double beta1, double beta2);

		double alpha;
		double beta1;
		double beta2;

		double beta1_pow_t;
		double beta2_pow_t;
		double alpha_t;

		void init(const std::vector<unsigned>& sizes) override;

		void next_mini_batch() override;

		void adjust(const mat_arr& gradient_noarr,
		            mat_arr* buffer,
		            mat_arr* target_noarr) override;
	};
}

#endif
