#include "training_data.h"
#include "mat_arr_math.h"

using namespace linalg;
using namespace annlib;

training_data::training_data(mat_arr input, mat_arr solution)
	: input(std::move(input)),
	  solution(std::move(solution))
{
	if (input.count != solution.count)
	{
		throw std::runtime_error("Dimensions differ");
	}
}

unsigned training_data::entry_count() const
{
	return input.count;
}

unsigned training_data::input_size() const
{
	return input.cols;
}

unsigned training_data::output_size() const
{
	return solution.cols;
}
