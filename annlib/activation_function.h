#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <functional>
#include "mat_arr.h"

using namespace linalg;

namespace annlib
{
	class activation_function
	{
	public:
		virtual ~activation_function() = default;

		virtual void apply(const mat_arr& in, mat_arr* target, mat_arr* derivative_target = nullptr) const = 0;
	};

	class abstract_activation_function : public activation_function
	{
	public:
		virtual ~abstract_activation_function() = default;

		void apply(const mat_arr& in, mat_arr* target, mat_arr* derivative_target = nullptr) const override;

		virtual fpt apply(fpt d) const = 0;

		virtual fpt apply_derivative(fpt d) const = 0;
	};

	class logistic_activation_function : public abstract_activation_function
	{
	public:
		logistic_activation_function() = default;

		fpt apply(fpt d) const override;

		fpt apply_derivative(fpt d) const override;
	};

	class relu_activation_function : public abstract_activation_function
	{
	public:
		relu_activation_function() = default;

		fpt apply(fpt d) const override;

		fpt apply_derivative(fpt d) const override;
	};
}

#endif
