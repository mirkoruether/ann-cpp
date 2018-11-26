#include "training_data.h"

annlib::training_data::training_data(mat_arr input, mat_arr solution)
	: input(std::move(input)),
	  solution(std::move(solution))
{
	if (input.count != solution.count)
	{
		throw runtime_error("Dimensions differ");
	}
}

unsigned annlib::training_data::entry_count() const
{
	return input.count;
}

unsigned annlib::training_data::input_size() const
{
	return input.cols;
}

unsigned annlib::training_data::output_size() const
{
	return solution.cols;
}
