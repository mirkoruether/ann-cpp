#ifndef GRADIENT_BASED_OPTIMIZER_H
#define GRADIENT_BASED_OPTIMIZER_H

#include "mat_arr.h"
using namespace std;
using namespace linalg;

namespace annlib
{
	class gradient_based_optimizer
	{
	public:
		virtual ~gradient_based_optimizer() = default;

		virtual void init(const vector<unsigned>& sizes)
		{
		}

		virtual void adjust_weights(const vector<mat_arr>& gradient_weights_noarr,
		                            vector<mat_arr>* weights_noarr) = 0;

		virtual void adjust_biases(const vector<mat_arr>& gradient_biases_rv_noarr,
		                           vector<mat_arr>* biases_rv_noarr) = 0;
	};

	template <size_t buffers>
	class abstract_gradient_based_optimizer : public gradient_based_optimizer
	{
		vector<array<mat_arr, buffers>> weight_buffers;
		vector<array<mat_arr, buffers>> biases_buffers;

		void init(const vector<unsigned>& sizes) override;

		void adjust_weights(const vector<mat_arr>& gradient_weights_noarr, vector<mat_arr>* weights_noarr) override;

		void adjust_biases(const vector<mat_arr>& gradient_biases_rv_noarr, vector<mat_arr>* biases_rv_noarr) override;

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

		void adjust(const mat_arr& gradient_noarr,
		            array<mat_arr, 2>* buffers_noarr,
		            mat_arr* target_noarr) override;
	};
}

#endif
