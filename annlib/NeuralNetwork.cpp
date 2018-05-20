//
// Created by Mirko on 04.05.2018.
//

#include "NeuralNetwork.h"


unsigned NeuralNetwork::getInputSize() const {
    return layers[0].getInputSize();
}

unsigned NeuralNetwork::getOutputSize() const {
    return layers[layers.size() - 1].getOutputSize();
}

const vector<NetworkLayer> &NeuralNetwork::getLayers() const {
    return layers;
}

vector<NetworkLayer> &NeuralNetwork::getLayers() {
    return layers;
}

DRowVector NeuralNetwork::feedForward(const DRowVector &in) const {
    if (!in.isRowVector() || in.getLength() != getInputSize()) {
        throw runtime_error("Wrong input size!");
    }

    const DRowVector *result = &in;
    for (const NetworkLayer &layer : layers) {
        const DRowVector layerResult = layer.feedForward(*result);
        result = &layerResult;
    }
    return *result;
}