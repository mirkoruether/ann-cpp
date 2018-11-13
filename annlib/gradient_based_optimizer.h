#ifndef GRADIENT_BASED_OPTIMIZER_H
#define GRADIENT_BASED_OPTIMIZER_H

#include "mat_arr.h"
using namespace std;
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

		virtual void init(const vector<unsigned>& sizes);

		virtual void next_mini_batch();

		virtual void adjust(const mat_arr& gradient_noarr,
		                    mat_arr* target_noarr,
		                    const adjust_target& at,
		                    unsigned layer_no) = 0;
	};

	template <size_t buffers>
	class abstract_gradient_based_optimizer : public gradient_based_optimizer
	{
	public:
		vector<array<mat_arr, buffers>> weights_buffers;
		vector<array<mat_arr, buffers>> biases_buffers;

		void init(const vector<unsigned>& sizes) override;

		void adjust(const mat_arr& gradient_noarr,
		            mat_arr* target_noarr,
		            const adjust_target& at,
		            unsigned layer_no) override;

		virtual void adjust(const mat_arr& gradient_noarr,
		                    array<mat_arr, buffers>* buffers_noarr,
		                    mat_arr* target_noarr) = 0;
	};

	class ordinary_sgd : public abstract_gradient_based_optimizer<0>
	{
	public:
		explicit ordinary_sgd(double learning_rate);

		double learning_rate;

		void adjust(const mat_arr& gradient_noarr,
		            array<mat_arr, 0>* buffers_noarr,
		            mat_arr* target_noarr) override;
	};

	class momentum_sgd : public abstract_gradient_based_optimizer<1>
	{
	public:
		explicit momentum_sgd(double learning_rate, double alpha);

		double learning_rate;
		double alpha;

		void adjust(const mat_arr& gradient_noarr,
		            array<mat_arr, 1>* buffers_noarr,
		            mat_arr* target_noarr) override;
	};

	class adam : public abstract_gradient_based_optimizer<2>
	{
	public:
		explicit adam(double alpha, double beta1, double beta2);

		double alpha;
		double beta1;
		double beta2;

		double beta1_pow_t;
		double beta2_pow_t;
		double alpha_t;

		void init(const vector<unsigned>& sizes) override;

		void next_mini_batch() override;

		void adjust(const mat_arr& gradient_noarr,
		            array<mat_arr, 2>* buffers_noarr,
		            mat_arr* target_noarr) override;
	};
}

#endif
