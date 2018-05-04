//
// Created by Mirko on 04.05.2018.
//

#include "NetworkLayer.h"
#include "MatrixMulUtil.h"

using namespace linalg;
using namespace std;

namespace annlib {
    NetworkLayer::NetworkLayer(const DMatrix &biases, const DMatrix &weights,
                               const function<double(double)> &activationFunction)
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

    const DMatrix &NetworkLayer::getBiases() const {
        return biases;
    }

    const DMatrix &NetworkLayer::getWeights() const {
        return weights;
    }

    const function<double(double)> &NetworkLayer::getActivationFunction() const {
        return activationFunction;
    }

    DMatrix NetworkLayer::calculateWeightedInput(const DMatrix &in) const {
        if (!in.isRowVector() || in.getLength() != getInputSize()) {
            throw runtime_error("wrong input dimensions");
        }
        return matrix_Mul(false, false, in, weights).addInPlace(biases);
    }

    DMatrix NetworkLayer::feedForward(const DMatrix &in) const {
        return calculateWeightedInput(in).applyFunctionToElementsInPlace(activationFunction);
    }
}