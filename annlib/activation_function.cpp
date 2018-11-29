#include "activation_function.h"
#include <cmath>

using namespace annlib;

namespace annlib
{
	activation_function::activation_function(std::function<double(double)> f,
	                                         std::function<double(double)> df)
		: f(std::move(f)), df(std::move(df))
	{
	}

	double activation_function::apply(double d) const
	{
		return f(d);
	}

	double activation_function::apply_derivative(double d) const
	{
		return df(d);
	}

	logistic_activation_function::logistic_activation_function(double T)
		: activation_function([=](double d) { return 1 / (1 + exp(-d / T)); },
		                      [=](double d)
		                      {
			                      const double v = std::exp(d / T) + 1.0;
			                      return std::exp(d / T) / (T * v * v);
		                      })
	{
	}
}
