#include "weight_norm_penalty.h"
#include "mat_arr_math.h"
#include <functional>

using namespace annlib;

abstract_weight_norm_penalty::abstract_weight_norm_penalty(double regularization_parameter)
	: regularization_parameter(regularization_parameter)
{
}

void abstract_weight_norm_penalty::add_penalty_to_gradient(const mat_arr& weights_noarr,
                                                           mat_arr* gradient_noarr) const
{
	add_penalty_to_gradient(weights_noarr, regularization_parameter, gradient_noarr);
}

L1_regularization::L1_regularization(double regularization_parameter)
	: abstract_weight_norm_penalty(regularization_parameter)
{
}

L1_regularization::L1_regularization(double normalized_regularization_parameter,
                                     unsigned training_set_size)
	: abstract_weight_norm_penalty(normalized_regularization_parameter / training_set_size)
{
}

void L1_regularization::add_penalty_to_gradient(const mat_arr& weights_noarr,
                                                double regularization_parameter,
                                                mat_arr* gradient_noarr) const
{
	const std::function<double(double)> sgn = [](double x) { return x > 0 ? 1.0 : -1.0; };

	mat_element_by_element_operation(*gradient_noarr, weights_noarr, gradient_noarr,
	                                 [&](double d, double w)
	                                 {
		                                 return d + regularization_parameter * sgn(w);
	                                 });
}

L2_regularization::L2_regularization(double regularization_parameter)
	: abstract_weight_norm_penalty(regularization_parameter)
{
}

L2_regularization::L2_regularization(double normalized_regularization_parameter,
                                     unsigned training_set_size)
	: abstract_weight_norm_penalty(normalized_regularization_parameter / training_set_size)
{
}

void L2_regularization::add_penalty_to_gradient(const mat_arr& weights_noarr,
                                                double regularization_parameter,
                                                mat_arr* gradient_noarr) const
{
	mat_element_by_element_operation(*gradient_noarr, weights_noarr, gradient_noarr,
	                                 [&](double d, double w)
	                                 {
		                                 return d + regularization_parameter * w;
	                                 });
}
