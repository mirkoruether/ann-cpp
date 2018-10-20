#ifndef ANN_CPP_NEURALNETWORK_H
#define ANN_CPP_NEURALNETWORK_H

#include <vector>
#include "networklayer.h"

using namespace linalg;
using namespace annlib;
using namespace std;

namespace annlib {
    class NeuralNetwork {
    private:
        vector<NetworkLayer> layers;

    public:
        NeuralNetwork(const vector<DMatrix> &weightsList, const vector<DRowVector> &biasesList,
                      const function<double(double)> &activationFunction);

        explicit NeuralNetwork(vector<NetworkLayer> layers);

        unsigned getInputSize() const;

        unsigned getOutputSize() const;

        DRowVector feedForward(const DRowVector &in) const;
    };
}


#endif //ANN_CPP_NEURALNETWORK_H
