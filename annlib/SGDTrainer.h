//
// Created by Mirko on 16.05.2018.
//

#ifndef ANN_CPP_MOMENTUMSTOCHASTICGRADIENTDESCENTTRAINER_H
#define ANN_CPP_MOMENTUMSTOCHASTICGRADIENTDESCENTTRAINER_H

#include <vector>
#include "DMatrix.h"
#include "ActivationFunction.h"
#include "CostFunction.h"
#include "TrainingData.h"
#include "CostFunctionRegularization.h"

using namespace std;
using namespace linalg;

namespace annlib {
    class SGDTrainer {
    private:
        vector<DMatrix> weights;
        vector<DRowVector> biases;

        vector<DMatrix> velocities;

    public:

        //------------------
        // Hyper Parameters
        //------------------
        double learningRate;
        double momentumCoEfficient;
        unsigned miniBatchSize;
        shared_ptr<ActivationFunction> activationFunction;
        shared_ptr<CostFunction> costFunction;
        shared_ptr<CostFunctionRegularization> costFunctionRegularization;

        SGDTrainer();

        void train(const vector<TrainingData> &data, int epochs);

        void trainEpoch(const vector<TrainingData> &data);

        void trainMiniBatch(const vector<TrainingData> &batch);
    };
}

#endif //ANN_CPP_MOMENTUMSTOCHASTICGRADIENTDESCENTTRAINER_H
