//
// Created by Mirko on 04.05.2018.
//

#include "NetworkLayer.h"
#include "MatrixMulUtil.h"

using namespace linalg;
using namespace std;

namespace annlib {
    NetworkLayer::NetworkLayer(const DRowVector &biases, const DMatrix &weights,
                               const ActivationFunction &activationFunction)
            : biases(biases), weights(weights), activationFunction(activationFunction) {
        if (!biases.isRowVector() || biases.getLength() != weights.getColumnCount()) {
            throw runtime_error("column count of weights and length of biases differ");
        }
    }

    unsigned NetworkLayer::getInputSize() const {
        return weights.getRowCount();
    }

    unsigned NetworkLayer::getOutputSize() const {
        return biases.getLength();
    }

    const DRowVector &NetworkLayer::getBiases() const {
        return biases;
    }

    DRowVector &NetworkLayer::getBiases() {
        return biases;
    }

    const DMatrix &NetworkLayer::getWeights() const {
        return weights;
    }

    DMatrix &NetworkLayer::getWeights() {
        return weights;
    }

    const ActivationFunction &NetworkLayer::getActivationFunction() const {
        return activationFunction;
    }

    DRowVector NetworkLayer::calculateWeightedInput(const DRowVector &in) const {
        if (!in.isRowVector() || in.getLength() != getInputSize()) {
            throw runtime_error("wrong input dimensions");
        }
        return (DRowVector) matrix_Mul(false, false, in, weights).addInPlace(biases);
    }

    DRowVector NetworkLayer::feedForward(const DRowVector &in) const {
        return (DRowVector) calculateWeightedInput(in).applyFunctionToElementsInPlace(activationFunction.f);
    }

    pair<DMatrix, DMatrix> NetworkLayer::feedForwardDetailed(const DRowVector &in) {
        DMatrix weightedInput = calculateWeightedInput(in);
        DMatrix activation = weightedInput.applyFunctionToElements(activationFunction.f);
        return pair<DMatrix, DMatrix>(weightedInput, activation);
    }
}