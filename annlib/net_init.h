#ifndef NET_INIT_H
#define NET_INIT_H

#include "mat_arr.h"
#include <random>
using namespace linalg;

namespace annlib
{
	class net_init
	{
	public:
		virtual ~net_init() = default;
		virtual void init_biases(mat_arr* biases_noarr_rv) = 0;
		virtual void init_weights(mat_arr* weights_noarr) = 0;
	};

	class gaussian_net_init : public net_init
	{
	private:
		normal_distribution<double> distribution;
		default_random_engine rng;
	public:
		gaussian_net_init();
		void fill_with_gaussian(mat_arr* mat);
		void init_biases(mat_arr* biases_noarr_rv) override;
		void init_weights(mat_arr* weights_noarr) override;
	};

	class normalized_gaussian_net_init : public gaussian_net_init
	{
	public:
		void init_weights(mat_arr* weights_noarr) override;
	};
}

#endif
