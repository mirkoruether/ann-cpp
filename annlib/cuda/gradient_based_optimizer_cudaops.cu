#include "gradient_based_optimizer_cudaops.cuh"

#include "cuda/linalg_cudaops_t.cuh"

struct momentum_update
{
	float alpha, learning_rate;

	momentum_update(float alpha, float learning_rate)
		: alpha(alpha),
		  learning_rate(learning_rate)
	{
	}

	__device__ float operator()(float v, float grad) const
	{
		return alpha * v - learning_rate * grad;
	}
};

void annlib::cuda::cuda_momentum_sgd_update_velocities(float alpha, float learning_rate,
                                                       const mat_arr& gradient_noarr,
                                                       mat_arr* velocities_noarr)
{
	linalg::cuda::cuda_element_by_element_operation(*velocities_noarr, gradient_noarr, velocities_noarr,
	                                                momentum_update(alpha, learning_rate), transpose_no);
}

struct ordinary_sgd_update
{
	float learning_rate;

	ordinary_sgd_update(float learning_rate)
		: learning_rate(learning_rate)
	{
	}

	__device__ float operator()(float target, float grad) const
	{
		return target - learning_rate * grad;
	}
};

void annlib::cuda::cuda_ordinary_sgd_update(float learning_rate,
                                            const mat_arr& gradient_noarr,
                                            mat_arr* target_noarr)
{
	linalg::cuda::cuda_element_by_element_operation(*target_noarr, gradient_noarr, target_noarr,
	                                                ordinary_sgd_update(learning_rate), transpose_no);
}

__global__ void adam_update_kernel(float alpha_t, float beta1, float beta2,
                                   const float* grad, float* m_buf, float* v_buf,
                                   float* target, unsigned size)
{
	const unsigned pos = cuda::current_pos_linear();
	if (pos < size)
	{
		const float gr = grad[pos];
		const float m_new = beta1 * m_buf[pos] + (1.0f - beta1) * gr;
		const float v_new = beta2 * v_buf[pos] + (1.0f - beta2) * gr * gr;
		m_buf[pos] = m_new;
		v_buf[pos] = v_new;
		target[pos] -= alpha_t * m_new / (std::sqrt(v_new) + 1e-8f);
	}
}

void annlib::cuda::cuda_adam_update(float alpha_t, float beta1, float beta2,
                                    const mat_arr& gradient_noarr,
                                    mat_arr* buffer,
                                    mat_arr* target_noarr)
{
	float* m_buf_dev = buffer->dev_start();
	float* v_buf_dev = m_buf_dev + buffer->rows * buffer->cols;

	linalg::cuda::prepare_launch_linear(*target_noarr, [&](unsigned size, unsigned threads, unsigned blocks)
	{
		adam_update_kernel << <blocks, threads >> >(alpha_t, beta1, beta2,
		                                            gradient_noarr.dev_start(), m_buf_dev, v_buf_dev,
		                                            target_noarr->dev_start(), size);
	});
}
