//
// Created by Mirko on 23.07.2018.
//

#ifndef ANN_CPP_TRAININGDATA_H
#define ANN_CPP_TRAININGDATA_H

#include <utility>
#include "dmatrix.h"

using namespace linalg;

namespace annlib
{
	class TrainingData
	{
	public:
		DRowVector input;
		DRowVector solution;

		TrainingData() = default;

		TrainingData(DRowVector input, DRowVector solution)
			: input(std::move(input)), solution(std::move(solution))
		{
		}
	};
}


#endif //ANN_CPP_TRAININGDATA_H
