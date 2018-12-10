#include "activation_function.h"
#include <cmath>
#include <stdexcept>

using namespace annlib;

namespace annlib
{
	activation_function::activation_function(std::function<float(float)> f,
	                                         std::function<float(float)> df)
		: f(std::move(f)), df(std::move(df))
	{
	}

	float activation_function::apply(float d) const
	{
		return f(d);
	}

	float activation_function::apply_derivative(float d) const
	{
		return df(d);
	}

	logistic_activation_function::logistic_activation_function(float T)
		: activation_function([=](float d)
		                      {
			                      const float e = d / T;
			                      return 1.0f / (1.0f + std::exp(-e));
		                      },
		                      [=](float d)
		                      {
			                      const float e = d / T;
								  const float e_abs = std::abs(e);
								  if (e_abs > 5.0f)
									  return 1.0f / std::exp(e_abs);
			                      const float v = std::exp(e) + 1.0f;
			                      return std::exp(e) / (T * v * v);
		                      })
	{
	}
}
