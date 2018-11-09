#ifndef SGD_TRAINER_H
#define SGD_TRAINER_H

#include <vector>
#include "mat_arr.h"
#include "cost_function.h"
#include "activation_function.h"
#include "weight_norm_penalty.h"
#include "training_data.h"
#include "gradient_based_optimizer.h"

using namespace std;
using namespace linalg;

namespace annlib
{
	class training_buffer;

	class sgd_trainer
	{
	private:
		vector<mat_arr> weights_noarr;
		vector<mat_arr> biases_noarr_rv;

		void build_batch(const training_data& training_data,
		                 mat_arr* input_rv,
		                 mat_arr* solution_rv,
		                 const function<double()>& random) const;

		vector<unsigned> sizes() const;

		void train_epochs(const training_data& training_data, unsigned epoch_count);

		void feed_forward_detailed(const mat_arr& input,
		                           vector<mat_arr>* weighted_inputs_rv,
		                           vector<mat_arr>* activations_rv) const;

		void calculate_error(const mat_arr& net_output_rv,
		                     const mat_arr& solution_rv,
		                     const vector<mat_arr>& weighted_inputs_rv,
		                     vector<mat_arr>* errors_rv) const;

		void calculate_gradient_weights(const mat_arr& input_rv,
		                                const vector<mat_arr>& activations_rv,
		                                const vector<mat_arr>& errors_rv,
		                                vector<mat_arr>* gradient_weights_noarr) const;

		void calculate_gradient_biases(const vector<mat_arr>& errors_rv,
		                               vector<mat_arr>* gradient_biases_noarr_rv) const;
	public:
		unsigned mini_batch_size;
		shared_ptr<activation_function> activation_f;
		shared_ptr<cost_function> cost_f;
		shared_ptr<weight_norm_penalty> weight_norm_penalty;
		shared_ptr<gradient_based_optimizer> optimizer;
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
}
#endif
