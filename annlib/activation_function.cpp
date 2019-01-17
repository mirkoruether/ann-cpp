#include "activation_function.h"

#include "mat_arr_math.h"
#include "mat_arr_math_t.h"
#include <cmath>

#ifdef ANNLIB_USE_CUDA
#include "activation_function_cudaops.cuh"
#endif

using namespace annlib;

namespace annlib
{
	mat_arr activation_function::apply(const mat_arr& in, mat_arr* target) const
	{
		return mat_element_wise_operation(in, target,
		                                  [&](float f)
		                                  {
			                                  return apply(f);
		                                  });
	}

	mat_arr activation_function::apply_derivative(const mat_arr& in, mat_arr* target) const
	{
		return mat_element_wise_operation(in, target,
		                                  [&](float f)
		                                  {
			                                  return apply_derivative(f);
		                                  });
	}

	abstract_activation_function::abstract_activation_function(std::function<float(float)> f,
	                                                           std::function<float(float)> df)
		: f(std::move(f)), df(std::move(df))
	{
	}

	float abstract_activation_function::apply(float d) const
	{
		return f(d);
	}

	float abstract_activation_function::apply_derivative(float d) const
	{
		return df(d);
	}

	float logistic_activation_function::apply(float d) const
	{
		return 1.0f / (1.0f + std::exp(-d));
	}

	mat_arr logistic_activation_function::apply(const mat_arr& in, mat_arr* target) const
	{
#ifdef ANNLIB_USE_CUDA
		cuda::cuda_sigmoid_apply(in, target);
		return *target;
#else
		return activation_function::apply(in, target);
#endif
	}

	float logistic_activation_function::apply_derivative(float d) const
	{
		const float e_abs = std::abs(d);
		if (e_abs > 5.0f)
			return 1.0f / std::exp(e_abs);
		const float v = std::exp(d) + 1.0f;
		return std::exp(d) / (v * v);
	}

	mat_arr logistic_activation_function::apply_derivative(const mat_arr& in, mat_arr* target) const
	{
#ifdef ANNLIB_USE_CUDA
		cuda::cuda_sigmoid_apply_derivative(in, target);
		return *target;
#else
		return activation_function::apply_derivative(in, target);
#endif
	}
}
