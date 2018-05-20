//
// Created by Mirko on 04.05.2018.
//

#ifndef ANN_CPP_NEURALNETWORK_H
#define ANN_CPP_NEURALNETWORK_H

#include <vector>
#include "NetworkLayer.h"
#include "DRowVector.h"

using namespace linalg;
using namespace annlib;
using namespace std;

namespace annlib {
    class NeuralNetwork {
    private:
        vector<NetworkLayer> layers;

    public:
        unsigned getInputSize() const;

        unsigned getOutputSize() const;

        DRowVector feedForward(const DRowVector &in) const;

        const vector<NetworkLayer> &getLayers() const;

        vector<NetworkLayer> &getLayers();
    };
}


#endif //ANN_CPP_NEURALNETWORK_H
