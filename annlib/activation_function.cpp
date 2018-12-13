#include "activation_function.h"
#include <cmath>
#include <stdexcept>

using namespace annlib;

namespace annlib
{
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

	float logistic_activation_function::apply_derivative(float d) const
	{
		const float e_abs = std::abs(d);
		if (e_abs > 5.0f)
			return 1.0f / std::exp(e_abs);
		const float v = std::exp(d) + 1.0f;
		return std::exp(d) / (v * v);
	}
}
