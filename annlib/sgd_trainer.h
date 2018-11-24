#ifndef SGD_TRAINER_H
#define SGD_TRAINER_H

#include <vector>
#include "mat_arr.h"
#include "cost_function.h"
#include "activation_function.h"
#include "weight_norm_penalty.h"
#include "training_data.h"
#include "gradient_based_optimizer.h"
#include <random>
#include "net_init.h"
#include "neural_network.h"

using namespace std;
using namespace linalg;

namespace annlib
{
	class training_buffer;

	class sgd_trainer
	{
	public:
		unsigned mini_batch_size;
		shared_ptr<activation_function> activation_f;
		shared_ptr<cost_function> cost_f;
		shared_ptr<weight_norm_penalty> weight_norm_penalty;
		shared_ptr<gradient_based_optimizer> optimizer;
		shared_ptr<net_init> net_init;

		vector<unsigned> sizes() const;

		unsigned get_layer_count() const;

		void init(vector<unsigned>& sizes);

		void train_epochs(const training_data& training_data, double epoch_count);

		neural_network to_neural_network(bool copy_parameters = false);

	private:
		vector<mat_arr> weights_noarr;
		vector<mat_arr> biases_noarr_rv;

		void feed_forward_detailed(const mat_arr& input,
		                           vector<mat_arr>* weighted_inputs_rv,
		                           vector<mat_arr>* activations_rv) const;

		void calculate_error(const mat_arr& net_output_rv,
		                     const mat_arr& solution_rv,
		                     const vector<mat_arr>& weighted_inputs_rv,
		                     vector<mat_arr>* errors_rv) const;

		void calculate_gradient_weight(const mat_arr& previous_activation_rv,
		                               const mat_arr& error_rv,
		                               mat_arr* gradient_weight_noarr) const;

		void calculate_gradient_bias(const mat_arr& error_rv,
		                             mat_arr* gradient_bias_noarr_rv) const;

		void adjust_weights(unsigned layer_no, training_buffer* buffer);
		void adjust_biases(unsigned layer_no, training_buffer* buffer);
	};

	class training_buffer
	{
	public:
		training_buffer(vector<unsigned> sizes, unsigned mini_batch_size);

		vector<mat_arr> weighted_inputs_rv;
		vector<mat_arr> activations_rv;
		vector<mat_arr> errors_rv;
		vector<mat_arr> gradient_biases_rv_noarr;
		vector<mat_arr> gradient_weights_noarr;
		mat_arr input_rv;
		mat_arr solution_rv;

		vector<mat_arr*> all();
		void clear();
	};

	class mini_batch_builder
	{
	public:
		const training_data& data;
		explicit mini_batch_builder(training_data data);

		void build_mini_batch(mat_arr* input_rv, mat_arr* solution_rv);
	private:
		const uniform_int_distribution<unsigned> distribution;
		default_random_engine rng;
	};
}
#endif
