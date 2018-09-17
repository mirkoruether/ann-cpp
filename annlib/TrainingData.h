//
// Created by Mirko on 23.07.2018.
//

#ifndef ANN_CPP_TRAININGDATA_H
#define ANN_CPP_TRAININGDATA_H

#include "DMatrix.h"

using namespace linalg;

namespace annlib {
    class TrainingData {
    public:
        const DRowVector &input;
        const DRowVector &solution;

        TrainingData(const DRowVector &input, const DRowVector &solution)
                : input(input), solution(solution) {}
    };
}


#endif //ANN_CPP_TRAININGDATA_H
