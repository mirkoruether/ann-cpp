//
// Created by Mirko on 04.05.2018.
//

#ifndef ANN_CPP_NETWORKLAYER_H
#define ANN_CPP_NETWORKLAYER_H

#include "DMatrix.h"
#include "ActivationFunction.h"
#include "DRowVector.h"

using namespace linalg;
using namespace std;

namespace annlib {
    class NetworkLayer {
    private:
        DRowVector biases;
        DMatrix weights;
        ActivationFunction activationFunction;
    public:
        NetworkLayer(const DRowVector &biases, const DMatrix &weights, const ActivationFunction &activationFunction);

        unsigned getInputSize() const;

        unsigned getOutputSize() const;

        const DRowVector &getBiases() const;

        DRowVector &getBiases();

        const DMatrix &getWeights() const;

        DMatrix &getWeights();

        const ActivationFunction &getActivationFunction() const;

        DRowVector calculateWeightedInput(const DRowVector &in) const;

        DRowVector feedForward(const DRowVector &in) const;

        pair<DMatrix, DMatrix> feedForwardDetailed(const DRowVector &in);
    };
}

#endif //ANN_CPP_NETWORKLAYER_H
