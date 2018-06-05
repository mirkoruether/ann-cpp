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

    DRowVector result = in;
    for (const NetworkLayer &layer : layers) {
        const DRowVector layerResult = layer.feedForward(result);
        result = layerResult;
    }
    return result;
}

NeuralNetwork::NeuralNetwork(const vector<NetworkLayer> &layers)
        : layers(layers) {}

NeuralNetwork::NeuralNetwork(const vector<DMatrix> &weightsList, const vector<DRowVector> &biasesList,
                             const function<double(double)> &activationFunction) {
    if (weightsList.size() != biasesList.size())
        throw runtime_error("weightsList and biasesList differ in length");

    layers = vector<NetworkLayer>(biasesList.size());
    for (int i = 0; i < layers.size(); ++i) {
        layers[i] = NetworkLayer(weightsList[i], biasesList[i], activationFunction);
    }
}
