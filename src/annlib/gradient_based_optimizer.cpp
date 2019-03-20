#include "gradient_based_optimizer.h"
#include "mat_arr_math.h"
#include "mat_arr_math_t.h"
#include "_calc_macros.h"

#ifdef ANNLIB_USE_CUDA
#include "gradient_based_optimizer_cudaops.cuh"
#endif

using namespace annlib;

void gradient_based_optimizer::start()
{
}

void gradient_based_optimizer::next_mini_batch()
{
}

void abstract_gradient_based_optimizer::add_to_buffer(std::string prefix, layer_buffer* buf,
                                                      unsigned count, unsigned rows, unsigned cols)
{
	buf->add_custom_count(prefix, buffer_count * count, rows, cols);
}

void abstract_gradient_based_optimizer::adjust(const mat_arr& gradient, mat_arr* target,
                                               std::string prefix, layer_buffer* buf)
{
	std::vector<mat_arr> buffer;
	mat_arr complete_buffer = buf->get_val(prefix);
	unsigned count_per_buffer = gradient.count;
	for (unsigned i = 0; i < buffer_count; i++)
	{
		buffer.emplace_back(complete_buffer.get_mats(i * count_per_buffer, count_per_buffer));
	}
	adjust(gradient, std::move(buffer), target);
}

abstract_gradient_based_optimizer::abstract_gradient_based_optimizer(unsigned buffer_count)
	: buffer_count(buffer_count)
{
}

ordinary_sgd::ordinary_sgd(fpt learning_rate)
	: abstract_gradient_based_optimizer(0),
	  learning_rate(learning_rate)
{
}

void ordinary_sgd::adjust(const mat_arr& gradient, std::vector<mat_arr> buffer, mat_arr* target)
{
#ifdef ANNLIB_USE_CUDA
	cuda::cuda_ordinary_sgd_update(learning_rate, gradient_noarr, target_noarr);
#else
	mat_element_by_element_operation(*target, gradient, target,
	                                 [&](fpt target, fpt grad)
	                                 {
		                                 return target - learning_rate * grad;
	                                 });
#endif
}

momentum_sgd::momentum_sgd(fpt learning_rate, fpt alpha)
	: abstract_gradient_based_optimizer(1),
	  learning_rate(learning_rate), alpha(alpha)
{
}

void momentum_sgd::adjust(const mat_arr& gradient, std::vector<mat_arr> buffer, mat_arr* target)
{
#ifdef ANNLIB_USE_CUDA
	cuda::cuda_momentum_sgd_update_velocities(alpha, learning_rate, gradient_noarr, buffer);
#else
	mat_element_by_element_operation(buffer[0], gradient, &buffer[0],
	                                 [&](fpt v, fpt grad)
	                                 {
		                                 return alpha * v - learning_rate * grad;
	                                 });
#endif

	M_ADD(*target, buffer[0], target);
}

adam::adam()
	: adam(0.001f, 0.9f, 0.99f)
{
}

adam::adam(fpt alpha, fpt beta1, fpt beta2)
	: abstract_gradient_based_optimizer(3),
	  alpha(alpha), beta1(beta1), beta2(beta2),
	  beta1_pow_t(1.0f), beta2_pow_t(1.0f), alpha_t(0.0f)
{
}

void adam::start()
{
	beta1_pow_t = 1.0f;
	beta2_pow_t = 1.0f;
	alpha_t = 0.0f;
}

void adam::next_mini_batch()
{
	beta1_pow_t *= beta1;
	beta2_pow_t *= beta2;
	alpha_t = alpha * std::sqrt(1 - beta2_pow_t) / (1 - beta1_pow_t);
}

void adam::adjust(const mat_arr& gradient, std::vector<mat_arr> buffer, mat_arr* target)
{
	mat_arr m_buf = buffer[0];
	mat_arr v_buf = buffer[1];

	mat_arr delta = buffer[2];

	mat_element_by_element_operation(m_buf, gradient, &m_buf,
	                                 [&](fpt m, fpt grad)
	                                 {
		                                 return beta1 * m + (1.0f - beta1) * grad;
	                                 });

	mat_element_by_element_operation(v_buf, gradient, &v_buf,
	                                 [&](fpt v, fpt grad)
	                                 {
		                                 return beta2 * v + (1.0f - beta2) * grad * grad;
	                                 });

	mat_element_by_element_operation(m_buf, v_buf, &delta,
	                                 [&](fpt m, fpt v)
	                                 {
		                                 return alpha_t * m / (std::sqrt(v) + 1e-8f);
	                                 });

	mat_element_wise_sub(*target, delta, target);
}
