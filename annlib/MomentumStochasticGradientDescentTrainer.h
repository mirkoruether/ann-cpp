//
// Created by Mirko on 16.05.2018.
//

#ifndef ANN_CPP_MOMENTUMSTOCHASTICGRADIENTDESCENTTRAINER_H
#define ANN_CPP_MOMENTUMSTOCHASTICGRADIENTDESCENTTRAINER_H

#include "DMatrix.h";
#include <vector>

using namespace std;
using namespace linalg;

namespace annlib {
    class MomentumStochasticGradientDescentTrainer {
    private:
        vector<DMatrix> velocities;
        double learningRate;
        double momentumCoEfficient;

        function<DMatrix(DMatrix, double, int)> costFunctionRegularization;

    };
}

#endif //ANN_CPP_MOMENTUMSTOCHASTICGRADIENTDESCENTTRAINER_H
