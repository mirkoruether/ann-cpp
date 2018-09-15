//
// Created by Mirko on 16.05.2018.
//

#include "SGDTrainer.h"
#include <random>

using namespace annlib;
using namespace std;

void SGDTrainer::train(const vector<TrainingData> &data, int epochs) {
    for (int i = 0; i < epochs; ++i) {
        trainEpoch(data);
    }
}

void SGDTrainer::trainEpoch(const vector<TrainingData> &data) {
    vector<TrainingData> miniBatch = vector<TrainingData>();

    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<unsigned> randomNumber(0, (unsigned) data.size());

    for (int i = 0; i < miniBatchSize; ++i) {
        miniBatch.push_back(data[randomNumber(rng)]);
    }

    trainMiniBatch(miniBatch);
}

void SGDTrainer::trainMiniBatch(const vector<TrainingData> &batch) {
    //TODO implement
}

SGDTrainer::SGDTrainer()
        : learningRate(0.1),
          momentumCoEfficient(0),
          miniBatchSize(10),
          activationFunction(make_shared<ActivationFunction>(LogisticActivationFunction(1))),
          costFunction(make_shared<QuadraticCosts>(QuadraticCosts())),
          costFunctionRegularization(shared_ptr<REG>()) //TODO default init
{}

