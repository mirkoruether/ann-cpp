#include "gradient_based_optimizer.h"
#include "mat_arr_math.h"
#include <cmath>

using namespace annlib;

void gradient_based_optimizer::init(const vector<unsigned>& sizes)
{
}

void gradient_based_optimizer::next_mini_batch()
{
}

template <size_t buffers>
void abstract_gradient_based_optimizer<buffers>::init(const vector<unsigned>& sizes)
{
	size_t size = sizes.size() - 1;

	weights_buffers = vector<array<mat_arr, buffers>>(size);
	biases_buffers = vector<array<mat_arr, buffers>>(size);
	for (unsigned i = 0; i < size - 1; i++)
	{
		for (unsigned b = 0; b < buffers; b++)
		{
			(weights_buffers[i])[b] = mat_arr(1, sizes[i], sizes[i + 1]);
			(biases_buffers[i])[b] = mat_arr(1, 1, sizes[i + 1]);
		}
	}
}

template <size_t buffers>
void abstract_gradient_based_optimizer<buffers>::adjust(const mat_arr& gradient_noarr,
                                                        mat_arr* target_noarr,
                                                        const adjust_target& at,
                                                        unsigned layer_no)
{
	adjust(gradient_noarr,
	       at == weights ? weights_buffers[layer_no] : biases_buffers[layer_no],
	       target_noarr);
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
	: alpha(alpha), beta1(beta1), beta2(beta2),
	  beta1_pow_t(1.0), beta2_pow_t(1.0), alpha_t(0.0)
{
}

void adam::init(const vector<unsigned>& sizes)
{
	beta1_pow_t = 1.0;
	beta2_pow_t = 1.0;
	alpha_t = 0.0;

	abstract_gradient_based_optimizer<2>::init(sizes);
}

void adam::next_mini_batch()
{
	beta1_pow_t *= beta1;
	beta2_pow_t *= beta2;
	alpha_t = alpha * sqrt(1 - beta2_pow_t) / (1 - beta1_pow_t);
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

	const array<mat_arr*, 3> in{{target_noarr, m_buf, v_buf}};
	mat_multiple_e_by_e_operation<3>(in, target_noarr,
	                                 [&](const array<double, 3>& arr)
	                                 {
		                                 return arr[0]
			                                 - alpha_t * arr[1]
			                                 / (sqrt(arr[2]) + 1e-8);
	                                 });
}
