#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <functional>

namespace annlib
{
	class activation_function
	{
	public:
		const std::function<float(float)> f;
		const std::function<float(float)> df;

		activation_function(std::function<float(float)> f, std::function<float(float)> df);

		float apply(float d) const;

		float apply_derivative(float d) const;
	};

	class logistic_activation_function : public activation_function
	{
	public:
		explicit logistic_activation_function(float T);
	};
}

#endif
