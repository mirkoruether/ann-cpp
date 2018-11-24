#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "mat_arr.h"
#include <functional>

using namespace linalg;

class neural_network
{
private:
	const vector<mat_arr>& weights_noarr;
	const vector<mat_arr>& biases_noarr_rv;
	const function<double(double)>& activation_function;

	void feed_forward_internal(const mat_arr& input, mat_arr* result, vector<mat_arr>* buffer) const;
	void feed_forward_result_nonnull(const mat_arr& input, mat_arr* result, vector<mat_arr>* buffer) const;

public:
	neural_network(const vector<mat_arr>& weights_noarr,
	               const vector<mat_arr>& biases_noarr_rv,
	               const function<double(double)>& activation_function);

	vector<mat_arr> build_buffer(unsigned count) const;

	mat_arr feed_forward(const mat_arr& input, mat_arr* result = nullptr, vector<mat_arr>* buffer = nullptr) const;
};

#endif
