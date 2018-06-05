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
        function<double(double)> activationFunction;
    public:
        NetworkLayer(const DMatrix &weights, const DRowVector &biases,
                     const function<double(double)> &activationFunction);

        unsigned getInputSize() const;

        unsigned getOutputSize() const;

        const DRowVector &getBiases() const;

        DRowVector &getBiases();

        const DMatrix &getWeights() const;

        DMatrix &getWeights();

        const function<double(double)> &getActivationFunction() const;

        DRowVector calculateWeightedInput(const DRowVector &in) const;

        DRowVector feedForward(const DRowVector &in) const;
    };
}

#endif //ANN_CPP_NETWORKLAYER_H
