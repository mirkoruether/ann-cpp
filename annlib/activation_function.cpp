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
	void abstract_activation_function::apply(const mat_arr& in, mat_arr* target, mat_arr* derivative_target) const
	{
		mat_element_wise_operation(in, target,
		                           [&](fpt f)
		                           {
			                           return apply(f);
		                           });

		if (derivative_target != nullptr)
		{
			mat_element_wise_operation(in, derivative_target,
			                           [&](fpt f)
			                           {
				                           return apply_derivative(f);
			                           });
		}
	}

	fpt logistic_activation_function::apply(fpt d) const
	{
		return 1.0f / (1.0f + std::exp(-d));
	}

	fpt logistic_activation_function::apply_derivative(fpt d) const
	{
		const fpt e_abs = std::abs(d);
		if (e_abs > 5.0f)
			return 1.0f / std::exp(e_abs);
		const fpt v = std::exp(d) + 1.0f;
		return std::exp(d) / (v * v);
	}

	fpt relu_activation_function::apply(fpt d) const
	{
		return std::max(static_cast<fpt>(0.0), d);
	}

	fpt relu_activation_function::apply_derivative(fpt d) const
	{
		return d > 0 ? 1.0f : 0.0f;
	}
}
