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
		                           [&](float f)
		                           {
			                           return apply(f);
		                           });

		if (derivative_target != nullptr)
		{
			mat_element_wise_operation(in, derivative_target,
			                           [&](float f)
			                           {
				                           return apply_derivative(f);
			                           });
		}
	}

	float logistic_activation_function::apply(float d) const
	{
		return 1.0f / (1.0f + std::exp(-d));
	}

	float logistic_activation_function::apply_derivative(float d) const
	{
		const float e_abs = std::abs(d);
		if (e_abs > 5.0f)
			return 1.0f / std::exp(e_abs);
		const float v = std::exp(d) + 1.0f;
		return std::exp(d) / (v * v);
	}

	float relu_activation_function::apply(float d) const
	{
		return std::max(0.0f, d);
	}

	float relu_activation_function::apply_derivative(float d) const
	{
		return d > 0 ? 1.0f : 0.0f;
	}

	void softmax_jacobian_times_input(const mat_arr& in_noarr, const mat_arr& softmax_noarr,
	                                  mat_arr* derivative_target_noarr)
	{
		const unsigned size = in_noarr.size();
		const float* a = in_noarr.start();
		const float* y = softmax_noarr.start();
		float* d = derivative_target_noarr->start();

		for (unsigned i = 0; i < size; i++)
		{
			float temp = 0.0f;
			for (unsigned j = 0; j < size; j++)
			{
				if (i == j)
				{
					temp += a[j] * (y[i] * (1 - y[j]));
				}
				else
				{
					temp += a[j] * (-1.0f * y[i] * y[j]);
				}
			}
			d[i] = temp;
		}
	}

	void softmax_activation_function::apply(const mat_arr& in, mat_arr* target, mat_arr* derivative_target) const
	{
		const unsigned count = in.count;
		for (unsigned i = 0; i < count; i++)
		{
			const mat_arr in_mat = in.get_mat(i);
			mat_arr target_mat = target->get_mat(i);

			const float max = mat_max(in_mat);
			mat_element_wise_operation(in_mat, &target_mat, [=](float z)
			{
				return std::exp(z - max);
			});

			const float den = mat_sum(target_mat);
			mat_element_wise_div(target_mat, den, &target_mat);

			if (derivative_target != nullptr)
			{
				mat_arr derivative_target_mat = derivative_target->get_mat(i);
				softmax_jacobian_times_input(in_mat, target_mat, &derivative_target_mat);
			}
		}
	}
}
