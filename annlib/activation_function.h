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

		virtual float apply(float d) const = 0;

		virtual void apply(const mat_arr& in, mat_arr* target) const;

		virtual float apply_derivative(float d) const = 0;

		virtual void apply_derivative(const mat_arr& in, mat_arr* target) const;
	};

	class abstract_activation_function : public activation_function
	{
	public:
		const std::function<float(float)> f;
		const std::function<float(float)> df;

		abstract_activation_function(std::function<float(float)> f, std::function<float(float)> df);

		float apply(float d) const override;

		float apply_derivative(float d) const override;
	};

	class logistic_activation_function : public activation_function
	{
	public:
		logistic_activation_function() = default;

		float apply(float d) const override;

		void apply(const mat_arr& in, mat_arr* target) const override;

		float apply_derivative(float d) const override;

		void apply_derivative(const mat_arr& in, mat_arr* target) const override;
	};

	class relu_activation_function : public activation_function
	{
	public:
		relu_activation_function() = default;

		float apply(float d) const override;

		void apply(const mat_arr& in, mat_arr* target) const override;

		float apply_derivative(float d) const override;

		void apply_derivative(const mat_arr& in, mat_arr* target) const override;
	};
}

#endif
