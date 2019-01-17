#include "gradient_based_optimizer.h"
#include "mat_arr_math.h"
#include "mat_arr_math_t.h"
#include <cmath>

using namespace annlib;

void gradient_based_optimizer::init(const std::vector<unsigned>& sizes)
{
}

void gradient_based_optimizer::next_mini_batch()
{
}

void abstract_gradient_based_optimizer::init(const std::vector<unsigned>& sizes)
{
	const auto size = static_cast<unsigned>(sizes.size() - 1);

	for (unsigned i = 0; i < size; i++)
	{
		weights_buffers.emplace_back(buffer_count, sizes[i], sizes[i + 1]);
		biases_buffers.emplace_back(buffer_count, 1, sizes[i + 1]);
	}
}

void abstract_gradient_based_optimizer::adjust(const mat_arr& gradient_noarr,
                                               mat_arr* target_noarr,
                                               const adjust_target& at,
                                               unsigned layer_no)
{
	adjust(gradient_noarr,
	       at == weights ? &weights_buffers[layer_no] : &biases_buffers[layer_no],
	       target_noarr);
}

abstract_gradient_based_optimizer::abstract_gradient_based_optimizer(unsigned buffer_count)
	: buffer_count(buffer_count)
{
}

ordinary_sgd::ordinary_sgd(float learning_rate)
	: abstract_gradient_based_optimizer(0),
	  learning_rate(learning_rate)
{
}

void ordinary_sgd::adjust(const mat_arr& gradient_noarr,
                          mat_arr* buffer,
                          mat_arr* target_noarr)
{
	mat_element_by_element_operation(*target_noarr, gradient_noarr, target_noarr,
	                                 [&](float target, float grad)
	                                 {
		                                 return target - learning_rate * grad;
	                                 });
}

momentum_sgd::momentum_sgd(float learning_rate, float alpha)
	: abstract_gradient_based_optimizer(1),
	  learning_rate(learning_rate), alpha(alpha)
{
}

void momentum_sgd::adjust(const mat_arr& gradient_noarr,
                          mat_arr* buffer,
                          mat_arr* target_noarr)
{
	mat_element_by_element_operation(*buffer, gradient_noarr, buffer,
	                                 [&](float v, float grad)
	                                 {
		                                 return alpha * v - learning_rate * grad;
	                                 });

	mat_element_wise_add(*target_noarr, *buffer, target_noarr);
}

adam::adam()
	:adam(0.001f, 0.9f, 0.99f) 
{
}

adam::adam(float alpha, float beta1, float beta2)
	: abstract_gradient_based_optimizer(3),
	  alpha(alpha), beta1(beta1), beta2(beta2),
	  beta1_pow_t(1.0f), beta2_pow_t(1.0f), alpha_t(0.0f)
{
}

void adam::init(const std::vector<unsigned>& sizes)
{
	beta1_pow_t = 1.0f;
	beta2_pow_t = 1.0f;
	alpha_t = 0.0f;

	abstract_gradient_based_optimizer::init(sizes);
}

void adam::next_mini_batch()
{
	beta1_pow_t *= beta1;
	beta2_pow_t *= beta2;
	alpha_t = alpha * std::sqrt(1 - beta2_pow_t) / (1 - beta1_pow_t);
}

void adam::adjust(const mat_arr& gradient_noarr,
                  mat_arr* buffer,
                  mat_arr* target_noarr)
{
	mat_arr m_buf = buffer->get_mat(0);
	mat_arr v_buf = buffer->get_mat(1);

	mat_arr delta = buffer->get_mat(2);

	mat_element_by_element_operation(m_buf, gradient_noarr, &m_buf,
	                                 [&](float m, float grad)
	                                 {
		                                 return beta1 * m + (1.0f - beta1) * grad;
	                                 });

	mat_element_by_element_operation(v_buf, gradient_noarr, &v_buf,
	                                 [&](float v, float grad)
	                                 {
		                                 return beta2 * v + (1.0f - beta2) * grad * grad;
	                                 });

	mat_element_by_element_operation(m_buf, v_buf, &delta, 
									 [&](float m, float v) 
									 {
										 return alpha_t * m / (std::sqrt(v) + 1e-8f);
									 });

	mat_element_wise_sub(*target_noarr, delta, target_noarr);
}
