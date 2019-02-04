#include "sgd_trainer.h"
#include "mat_arr.h"
#include "mat_arr_math.h"
#include <iostream>
#include <future>
#include <random>

#include "_calc_macros.h"
#include "output_layer.h"

#ifdef ANNLIB_USE_CUDA
#include "sgd_trainer_cudaops.cuh"
#include "cuda/linalg_cudaops.cuh"
#endif

using namespace linalg;
using namespace annlib;

unsigned sgd_trainer::get_layer_count() const
{
	return static_cast<unsigned>(layers.size());
}

std::vector<unsigned> sgd_trainer::get_sizes() const
{
	std::vector<unsigned> result;
	result.emplace_back(get_input_size());
	for (const auto& layer : layers)
	{
		result.emplace_back(layer->output_size);
	}
	return result;
}

unsigned sgd_trainer::get_input_size() const
{
	if (layers.empty())
	{
		throw std::runtime_error("No layers");
	}
	return layers.front()->input_size;
}

unsigned sgd_trainer::get_output_size() const
{
	if (layers.empty())
	{
		throw std::runtime_error("No layers");
	}
	return layers.back()->output_size;
}

void sgd_trainer::add_layer(std::shared_ptr<network_layer> layer)
{
	if (layer == nullptr)
	{
		throw std::runtime_error("null");
	}

	if (!layers.empty() && layer->input_size != get_output_size())
	{
		throw std::runtime_error("sizes do not fit");
	}

	layers.emplace_back(layer);
}

void sgd_trainer::init()
{
	std::mt19937 rng;
	rng.seed(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));
	for (const auto& layer_ptr : layers)
	{
		layer_ptr->init(&rng);
	}
}

void sgd_trainer::train_epochs(const training_data& training_data, gradient_based_optimizer* opt,
                               const unsigned mini_batch_size, const double epoch_count, bool print)
{
	const auto batch_count = static_cast<unsigned>((epoch_count / mini_batch_size) * training_data.input.count);

	training_buffer buf(mini_batch_size, get_sizes());
	for (unsigned i = 0; i < layers.size(); i++)
	{
		layers[i]->prepare_buffer(buf.lbuf(i), opt);
	}

#ifdef _OPENMP
	std::vector<training_buffer> partial_buffers = buf.do_split(std::thread::hardware_concurrency());
#endif

	mini_batch_builder mb_builder(training_data);

	for (unsigned batch_no = 0; batch_no < batch_count; batch_no++)
	{
		if (print && batch_no % 100 == 0)
		{
			std::cout << "\r" << batch_no << "/" << batch_count
				<< " [" << unsigned(100.0 * (batch_no + 1) / batch_count) << "%]";
		}

		mb_builder.build_mini_batch(buf.in(0), buf.sol());


#ifdef _OPENMP
#pragma omp parallel for
		for (int i = 0; i < static_cast<int>(partial_buffers.size()); i++)
		{
			do_feed_forward_and_backprop(&partial_buffers[i]);
		}
#else
		do_feed_forward_and_backprop(&buf);
#endif

		do_adjustments(opt, &buf);
	}

	if (print)
	{
		std::cout << "\r" << batch_count << "/" << batch_count << " [100%]" << std::endl;
	}
}

mat_arr sgd_trainer::feed_forward(const mat_arr& in) const
{
	std::vector<mat_arr> buffers;
	for (unsigned i = 0; i < get_layer_count(); i++)
	{
		buffers.emplace_back(in.count, 1, layers[i]->output_size);
	}

	const mat_arr* la_in = &in;
	for (unsigned i = 0; i < get_layer_count(); i++)
	{
		mat_arr* la_out = &buffers[i];
		layers[i]->feed_forward(*la_in, la_out);
		la_in = la_out;
	}
	return *la_in;
}

float sgd_trainer::calculate_costs(const mat_arr& net_output, const mat_arr& solution) const
{
	return dynamic_cast<output_layer*>(layers.back().get())->calculate_costs(net_output, solution);
}

void sgd_trainer::do_feed_forward_and_backprop(training_buffer* buffer) const
{
	const auto layer_count = get_layer_count();
	for (unsigned i = 0; i < layer_count; i++)
	{
		layers[i]->feed_forward_detailed(*buffer->in(i), buffer->out(i), buffer->lbuf(i));
	}

	const auto last_layer = layer_count - 1;

	for (unsigned i = last_layer; i >= 1; i--)
	{
		layers[i]->backprop(*buffer->error(i), buffer->error(i - 1), buffer->lbuf(i));
	}
}

void sgd_trainer::do_adjustments(gradient_based_optimizer* opt, training_buffer* buffer)
{
	opt->next_mini_batch();

#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(get_layer_count()); i++)
	{
		layers[i]->optimize(*buffer->error(i), opt, buffer->lbuf(i));
	}
}

mini_batch_builder::mini_batch_builder(training_data data)
	: data(std::move(data)),
	  distribution(0, data.input.count - 1)
{
	rng.seed(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));
}

void mini_batch_builder::build_mini_batch(mat_arr* input_rv, mat_arr* solution_rv)
{
	const unsigned mini_batch_size = input_rv->count;
	std::vector<unsigned> batch_indices(mini_batch_size);
	for (unsigned i = 0; i < mini_batch_size; i++)
	{
		batch_indices[i] = distribution(rng);
	}

	M_SELECT(data.input, batch_indices, input_rv);
	M_SELECT(data.solution, batch_indices, solution_rv);
}
