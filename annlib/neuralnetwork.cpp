#include "neuralnetwork.h"
#include <utility>


unsigned NeuralNetwork::getInputSize() const {
    return layers[0].getInputSize();
}

unsigned NeuralNetwork::getOutputSize() const {
    return layers[layers.size() - 1].getOutputSize();
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

NeuralNetwork::NeuralNetwork(vector<NetworkLayer> layers)
        : layers(move(layers)) {}

NeuralNetwork::NeuralNetwork(const vector<DMatrix> &weightsList, const vector<DRowVector> &biasesList,
                             const function<double(double)> &activationFunction) {
    if (weightsList.size() != biasesList.size())
        throw runtime_error("weightsList and biasesList differ in length");

    layers = vector<NetworkLayer>();
    for (int i = 0; i < biasesList.size(); ++i) {
        layers.emplace_back(NetworkLayer(weightsList[i], biasesList[i], activationFunction));
    }
}
