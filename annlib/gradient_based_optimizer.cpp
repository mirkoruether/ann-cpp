#include "gradient_based_optimizer.h"
#include "mat_arr_math.h"
#include <cmath>

using namespace annlib;

template <size_t buffers>
void abstract_gradient_based_optimizer<buffers>::init(const vector<unsigned>& sizes)
{
	size_t size = sizes.size() - 1;

	weight_buffers = vector<array<mat_arr, buffers>>(size);
	biases_buffers = vector<array<mat_arr, buffers>>(size);
	for (unsigned i = 0; i < size - 1; i++)
	{
		for (unsigned b = 0; b < buffers; b++)
		{
			(weight_buffers[i])[b] = mat_arr(1, sizes[i], sizes[i + 1]);
			(biases_buffers[i])[b] = mat_arr(1, 1, sizes[i + 1]);
		}
	}
}

template <size_t buffers>
void abstract_gradient_based_optimizer<buffers>::adjust_weights(const vector<mat_arr>& gradient_weights_noarr,
                                                                vector<mat_arr>* weights_noarr)
{
	const size_t size = gradient_weights_noarr.size();

	for (unsigned i = 0; i < size; i++)
	{
		adjust(gradient_weights_noarr[i],
		       &weight_buffers[i],
		       &weights_noarr->operator[](i));
	}
}

template <size_t buffers>
void abstract_gradient_based_optimizer<buffers>::adjust_biases(const vector<mat_arr>& gradient_biases_rv_noarr,
                                                               vector<mat_arr>* biases_rv_noarr)
{
	const size_t size = gradient_biases_rv_noarr.size();

	for (unsigned i = 0; i < size; i++)
	{
		adjust(gradient_biases_rv_noarr[i],
		       &biases_buffers->operator[](i),
		       biases_rv_noarr[i]);
	}
}

ordinary_sgd::ordinary_sgd(double learning_rate)
	: learning_rate(learning_rate)
{
}

void ordinary_sgd::adjust(const mat_arr& gradient_noarr,
                          array<mat_arr, 0>* buffers_noarr,
                          mat_arr* target_noarr)
{
	mat_element_by_element_operation(*target_noarr, gradient_noarr, target_noarr,
	                                 [&](double target, double grad)
	                                 {
		                                 return target - learning_rate * grad;
	                                 });
}

momentum_sgd::momentum_sgd(double learning_rate, double alpha)
	: learning_rate(learning_rate), alpha(alpha)
{
}

void momentum_sgd::adjust(const mat_arr& gradient_noarr,
                          array<mat_arr, 1>* buffers_noarr,
                          mat_arr* target_noarr)
{
	mat_arr* velocities = &buffers_noarr->operator[](0);
	mat_element_by_element_operation(*velocities, gradient_noarr, velocities,
	                                 [&](double v, double grad)
	                                 {
		                                 return alpha * v - learning_rate * grad;
	                                 });

	mat_element_wise_add(*target_noarr, *velocities, target_noarr);
}

adam::adam(double alpha, double beta1, double beta2)
	: alpha(alpha), beta1(beta1), beta2(beta2)
{
}

void adam::adjust(const mat_arr& gradient_noarr,
                  array<mat_arr, 2>* buffers_noarr,
                  mat_arr* target_noarr)
{
	mat_arr* m_buf = &buffers_noarr->operator[](0);
	mat_arr* v_buf = &buffers_noarr->operator[](1);

	mat_element_by_element_operation(*m_buf, gradient_noarr, m_buf,
	                                 [&](double m, double grad)
	                                 {
		                                 return beta1 * m + (1 - beta1) * grad;
	                                 });

	mat_element_by_element_operation(*v_buf, gradient_noarr, v_buf,
	                                 [&](double v, double grad)
	                                 {
		                                 return beta2 * v + (1 - beta2) * pow(grad, 2);
	                                 });

	//TODO Count iterations (t variable in adam algorithm)
	//TODO Add mat_element_by_element_operation with 3 inputs to implement adam update rule
	throw runtime_error("Not implemented yet");
}
