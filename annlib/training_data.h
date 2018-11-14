#ifndef TRAINING_DATA_H
#define TRAINING_DATA_H

#include <utility>
#include "mat_arr.h"

using namespace linalg;

namespace annlib
{
	class training_data
	{
	public:
		mat_arr input;
		mat_arr solution;

		training_data(mat_arr input, mat_arr solution)
			: input(std::move(input)), solution(std::move(solution))
		{
			if(input.count != solution.count)
			{
				throw runtime_error("Different dimensions");
			}
		}
	};
}


#endif
