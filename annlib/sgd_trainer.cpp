#include "sgd_trainer.h"
#include "mat_arr.h"
#include "mat_arr_math.h"

using namespace std;
using namespace linalg;
using namespace annlib;

void sgd_trainer::feed_forward_detailed(const mat_arr& input,
                                        vector<mat_arr>* weighted_inputs_rv,
                                        vector<mat_arr>* activations_rv) const
{
	const size_t layer_count = weights_noarr.size();

	const mat_arr* layerInput = &input;
	for (unsigned layerNo = 0; layerNo < layer_count; layerNo++)
	{
		mat_matrix_mul(*layerInput,
		               weights_noarr[layerNo],
		               &weighted_inputs_rv->operator[](layerNo));

		mat_element_wise_add(weighted_inputs_rv->operator[](layerNo),
		                     biases_noarr_rv[layerNo],
		                     &weighted_inputs_rv->operator[](layerNo));

		mat_element_wise_operation(weighted_inputs_rv->operator[](layerNo),
		                           &activations_rv->operator[](layerNo),
		                           activation_f->f);

		layerInput = &activations_rv->operator[](layerNo);
	}
}

void sgd_trainer::calculate_error(const mat_arr& net_output_rv, const mat_arr& solution_rv,
                                  const vector<mat_arr>& weighted_inputs_rv,
                                  const vector<mat_arr>& activations_rv,
                                  vector<mat_arr>* errors_rv) const
{
	const size_t layer_count = weights_noarr.size();

	cost_f->calculate_output_layer_error(net_output_rv,
	                                     solution_rv,
	                                     weighted_inputs_rv[layer_count - 1],
	                                     activation_f->df,
	                                     &errors_rv->operator[](layer_count - 1));


}
