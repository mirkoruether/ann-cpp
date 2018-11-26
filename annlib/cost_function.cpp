#include "cost_function.h"
#include "mat_arr_math.h"
#include <cmath>

using namespace linalg;
using namespace annlib;

void cost_function::calculate_output_layer_error(const mat_arr& net_output_rv,
                                                 const mat_arr& solution_rv,
                                                 const mat_arr& output_layer_weighted_input_rv,
                                                 const function<double(double)>& derivative_activation_function,
                                                 mat_arr* output_layer_error_rv) const
{
	calculate_gradient(net_output_rv, solution_rv, output_layer_error_rv);

	mat_element_by_element_operation(*output_layer_error_rv, output_layer_weighted_input_rv, output_layer_error_rv,
	                                 [derivative_activation_function](double grad, double wi)
	                                 {
		                                 return grad * derivative_activation_function(wi);
	                                 });
}

double quadratic_costs::calculate_costs(const mat_arr& net_output_rv, const mat_arr& solution_rv) const
{
	const size_t size = net_output_rv.size();
	if (solution_rv.size() != size)
	{
		throw runtime_error("sizes differ");
	}

	const double* no_element = net_output_rv.start();
	const double* s_element = solution_rv.start();

	double result = 0.0;
	for (unsigned i = 0; i < size; i++)
	{
		result += pow(*no_element - *s_element, 2);
		no_element++;
		s_element++;
	}
	return result;
}

void quadratic_costs::calculate_gradient(const mat_arr& net_output_rv,
                                         const mat_arr& solution_rv,
                                         mat_arr* gradient_rv) const
{
	mat_element_wise_sub(net_output_rv, solution_rv, gradient_rv);
}

double cross_entropy_costs::calculate_costs(const mat_arr& net_output_rv,
                                            const mat_arr& solution_rv) const
{
	const size_t size = net_output_rv.size();
	if (solution_rv.size() != size)
	{
		throw runtime_error("sizes differ");
	}

	const double* no_element = net_output_rv.start();
	const double* s_element = solution_rv.start();

	double result = 0;
	for (unsigned i = 0; i < size; i++)
	{
		const double a = *no_element;
		const double y = *s_element;
		result += y * log(a) + (1 - y) * log(1 - a);

		no_element++;
		s_element++;
	}
	return -result;
}

void cross_entropy_costs::calculate_gradient(const mat_arr& net_output_rv,
                                             const mat_arr& solution_rv,
                                             mat_arr* gradient_rv) const
{
	mat_element_by_element_operation(net_output_rv, solution_rv, gradient_rv,
	                                 [](double a, double y)
	                                 {
		                                 return y / a - (1 - y) / (1 - a);
	                                 });
}

void cross_entropy_costs::calculate_output_layer_error(const mat_arr& net_output_rv,
                                                       const mat_arr& solution_rv,
                                                       const mat_arr& output_layer_weighted_input_rv,
                                                       const function<double(double)>& derivative_activation_function,
                                                       mat_arr* output_layer_error_rv) const
{
	//TODO Check for logistic activation function
	mat_element_wise_sub(net_output_rv, solution_rv, output_layer_error_rv);
}
