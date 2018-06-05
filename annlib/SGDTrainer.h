//
// Created by Mirko on 16.05.2018.
//

#ifndef ANN_CPP_MOMENTUMSTOCHASTICGRADIENTDESCENTTRAINER_H
#define ANN_CPP_MOMENTUMSTOCHASTICGRADIENTDESCENTTRAINER_H

#include <vector>
#include "DMatrix.h";
#include "ActivationFunction.h"
#include "CostFunction.h"
#include "TrainingData.h"
#include <vector>

using namespace std;
using namespace linalg;

namespace annlib {
    class SGDTrainer {
        using REG = function<DMatrix(DMatrix, double, unsigned)>;
    private:
        vector<DMatrix> weights;
        vector<DRowVector> biases;

        vector<DMatrix> velocities;

        //------------------
        // Hyper Parameters
        //------------------
        double learningRate;
        double momentumCoEfficient;
        unsigned miniBatchSize;
        ActivationFunction activationFunction;
        CostFunction costFunction;
        REG costFunctionRegularization;

    public:
        SGDTrainer()
                : learningRate(0.1),
                  momentumCoEfficient(0),
                  miniBatchSize(10),
                  activationFunction(LogisticActivationFunction(1)),
                  costFunction(costFunction), //TODO default init
                  costFunctionRegularization(costFunctionRegularization) //TODO default init
        {}

        void train(const vector<TrainingData> &data, int epochs);

        void trainEpoch(const vector<TrainingData> &data);

        void trainMiniBatch(const vector<TrainingData> &batch);

        SGDTrainer &withLearningRate(double learningRate) {
            this->learningRate = learningRate;
            return *this;
        }

        SGDTrainer &withMomentumCoEfficient(double momentumCoEfficient) {
            this->momentumCoEfficient = momentumCoEfficient;
            return *this;
        }

        SGDTrainer &withMiniBatchSize(unsigned miniBatchSize) {
            this->miniBatchSize = miniBatchSize;
            return *this;
        }

        SGDTrainer &withActivationFunction(const ActivationFunction &activationFunction) {
            this->activationFunction = activationFunction;
            return *this;
        }

        SGDTrainer &withCostFunction(const CostFunction &costFunction) {
            this->costFunction = costFunction;
            return *this;
        }

        SGDTrainer &withCostFunctionRegularization(const REG &costFunctionRegularization) {
            this->costFunctionRegularization = costFunctionRegularization;
            return *this;
        }
    };
}

#endif //ANN_CPP_MOMENTUMSTOCHASTICGRADIENTDESCENTTRAINER_H
