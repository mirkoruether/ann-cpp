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
    vector<TrainingData> miniBatch = vector(miniBatchSize);

    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<unsigned> randomNumber(0, (unsigned) data.size());

    for (int i = 0; i < miniBatchSize; ++i) {
        miniBatch[i] = data[randomNumber(rng)];
    }

    trainMiniBatch(miniBatch);
}

void SGDTrainer::trainMiniBatch(const vector<TrainingData> &batch) {
    //TODO implement
}

