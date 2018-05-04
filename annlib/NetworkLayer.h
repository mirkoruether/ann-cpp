//
// Created by Mirko on 04.05.2018.
//

#ifndef ANN_CPP_NETWORKLAYER_H
#define ANN_CPP_NETWORKLAYER_H

#include "DMatrix.h"
#include <functional>

using namespace linalg;
using namespace std;

namespace annlib {
    class NetworkLayer {
    private:
        DMatrix biases;
        DMatrix weights;
        function<double(double)> activationFunction;
    public:
        NetworkLayer(const DMatrix &biases, const DMatrix &weights, const function<double(double)> &activationFunction);

        unsigned getInputSize() const;

        unsigned getOutputSize() const;

        const DMatrix &getBiases() const;

        const DMatrix &getWeights() const;

        const function<double(double)> &getActivationFunction() const;

        DMatrix calculateWeightedInput(const DMatrix &in) const;

        DMatrix feedForward(const DMatrix &in) const;
    };
}

#endif //ANN_CPP_NETWORKLAYER_H
