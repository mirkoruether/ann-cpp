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

DMatrix NeuralNetwork::feedForward(const DMatrix &in) const {
    if (!in.isRowVector() || in.getLength() != getInputSize()) {
        throw runtime_error("Wrong input size!");
    }

    DMatrix result = in;
    for (const NetworkLayer &layer : layers) {
        result = layer.feedForward(result);
    }
    return result;
}