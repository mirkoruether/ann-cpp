//
// Created by Mirko on 05.06.2018.
//

#ifndef ANN_CPP_TRAININGDATA_H
#define ANN_CPP_TRAININGDATA_H

#include "DRowVector.h"

class TrainingData {
public:
    const DRowVector &input;
    const DRowVector &solution;

    TrainingData(const DRowVector &input, const DRowVector &solution)
            : input(input), solution(solution) {}
};


#endif //ANN_CPP_TRAININGDATA_H
