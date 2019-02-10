#ifndef TRAINING_DATA_H
#define TRAINING_DATA_H

#include "mat_arr.h"

using namespace linalg;

namespace annlib
{
	class training_data
	{
	public:
		const mat_arr input;
		const mat_arr solution;

		training_data(mat_arr input, mat_arr solution);

		unsigned entry_count() const;

		unsigned input_size() const;

		unsigned output_size() const;
	};
}

#endif
