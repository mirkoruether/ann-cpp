#include "net_init.h"
#include "mat_arr_math.h"
#include <chrono>

annlib::gaussian_net_init::gaussian_net_init()
	: distribution(0.0f, 1.0f)
{
	rng.seed(std::chrono::system_clock::now().time_since_epoch().count());
}

void annlib::gaussian_net_init::fill_with_gaussian(mat_arr* mat)
{
	mat_element_wise_operation(*mat, mat,
	                           [&](float val)
	                           {
		                           return distribution(rng);
	                           });
}


void annlib::gaussian_net_init::init_biases(mat_arr* biases_noarr_rv)
{
	fill_with_gaussian(biases_noarr_rv);
}

void annlib::gaussian_net_init::init_weights(mat_arr* weights_noarr)
{
	fill_with_gaussian(weights_noarr);
}

void annlib::normalized_gaussian_net_init::init_weights(mat_arr* weights_noarr)
{
	gaussian_net_init::init_weights(weights_noarr);
	const auto factor = static_cast<float>(2.0 / std::sqrt(1.0 * weights_noarr->rows));
	mat_element_wise_mul(*weights_noarr, factor, weights_noarr);
}
