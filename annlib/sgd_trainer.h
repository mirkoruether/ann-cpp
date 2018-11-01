#ifndef SGD_TRAINER_H
#define SGD_TRAINER_H

#include <vector>
#include "mat_arr.h"
#include "cost_function.h"
#include "activation_function.h"

using namespace std;
using namespace linalg;

namespace annlib
{
	class sgd_trainer
	{
	private:
		vector<mat_arr> weights_noarr;
		vector<mat_arr> biases_noarr_rv;

		void feed_forward_detailed(const mat_arr& input,
		                           vector<mat_arr>* weighted_inputs_rv,
		                           vector<mat_arr>* activations_rv) const;

		void calculate_error(const mat_arr& net_output_rv,
		                     const mat_arr& solution_rv,
		                     const vector<mat_arr>& weighted_inputs_rv,
		                     const vector<mat_arr>& activations_rv,
		                     vector<mat_arr>* errors_rv) const;

		void calculate_weight_decay(const mat_arr& input_rv,
		                          const vector<mat_arr>& activations_rv,
		                          const vector<mat_arr>& errors_rv,
		                          vector<mat_arr>* weight_decays_noarr) const;
	public:
		double learning_rate;
		unsigned batch_size;
		shared_ptr<activation_function> activation_f;
		shared_ptr<cost_function> cost_f;
	};
}
#endif
