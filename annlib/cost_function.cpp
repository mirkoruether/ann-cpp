#include "cost_function.h"
#include "mat_arr_math.h"
#include "mat_arr_math_t.h"
#include <cmath>
#include "activation_function.h"

#include "_calc_macros.h"

#ifdef ANNLIB_USE_CUDA
#include "cost_function_cudaops.cuh"
#include "cuda/linalg_cudaops.cuh"
#endif

using namespace linalg;
using namespace annlib;

float quadratic_costs::calculate_costs(const mat_arr& net_output_rv, const mat_arr& solution_rv) const
{
	const size_t size = net_output_rv.size();
	if (solution_rv.size() != size)
	{
		throw std::runtime_error("sizes differ");
	}

	const float* no_element = net_output_rv.start();
	const float* s_element = solution_rv.start();

	float result = 0.0;
	for (unsigned i = 0; i < size; i++)
	{
		result += (*no_element - *s_element) * (*no_element - *s_element);
		no_element++;
		s_element++;
	}
	return result;
}

void quadratic_costs::calculate_gradient(const mat_arr& net_output_rv,
                                         const mat_arr& solution_rv,
                                         mat_arr* gradient_rv) const
{
	M_SUB(net_output_rv, solution_rv, gradient_rv);
}

float cross_entropy_costs::calculate_costs(const mat_arr& net_output_rv,
                                           const mat_arr& solution_rv) const
{
	const size_t size = net_output_rv.size();
	if (solution_rv.size() != size)
	{
		throw std::runtime_error("sizes differ");
	}

	const float* no_element = net_output_rv.start();
	const float* s_element = solution_rv.start();

	float result = 0;
	for (unsigned i = 0; i < size; i++)
	{
		const float a = *no_element;
		const float y = *s_element;
		result += y * std::log(a) + (1 - y) * std::log(1 - a);

		no_element++;
		s_element++;
	}
	return -result;
}

struct cross_entropy_gradient
{
	float operator()(float a, float y) const
	{
		return (1.0f - y) / (1.0f - a) - y / a;
	}
};

void cross_entropy_costs::calculate_gradient(const mat_arr& net_output_rv,
                                             const mat_arr& solution_rv,
                                             mat_arr* gradient_rv) const
{
#ifdef ANNLIB_USE_CUDA
	cuda::cuda_cross_entropy_cost_gradient(net_output_rv, solution_rv, gradient_rv);
#else
	mat_element_by_element_operation(net_output_rv, solution_rv, gradient_rv,
	                                 cross_entropy_gradient());
#endif
}
