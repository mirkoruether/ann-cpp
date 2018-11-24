#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <functional>

using namespace std;

namespace annlib
{
	class activation_function
	{
	public:
		const function<double(double)> f;
		const function<double(double)> df;

		activation_function(function<double(double)> f, function<double(double)> df);

		double apply(double d) const;

		double apply_derivative(double d) const;
	};

	class logistic_activation_function : public activation_function
	{
	public:
		explicit logistic_activation_function(double T);
	};
}

#endif
