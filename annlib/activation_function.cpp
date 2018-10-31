#include "activation_function.h"

using namespace annlib;

namespace annlib
{
	logistic_activation_function::logistic_activation_function(double T)
		: activation_function([T](double d) { return 1 / (1 + exp(-d / T)); },
		                      [T](double d) { return exp(d / T) / (T * pow(exp(d / T) + 1.0, 2.0)); })
	{
	}
}
