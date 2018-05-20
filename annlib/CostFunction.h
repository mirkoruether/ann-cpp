//
// Created by Mirko on 18.05.2018.
//

#ifndef ANN_CPP_COSTFUNCTION_H
#define ANN_CPP_COSTFUNCTION_H

#include "DMatrix.h"
#include <functional>

using namespace std;
using namespace linalg;

namespace annlib {
    class CostFunction {
    public:
        virtual double calculateCosts(const DMatrix &netOutput, const DMatrix &solution) const = 0;

        virtual DMatrix calculateGradient(const DMatrix &netOutput, const DMatrix &solution) const = 0;

        virtual DMatrix calculateErrorOfLastLayer(const DMatrix &netOutput, const DMatrix &solution,
                                                  const DMatrix &lastLayerDerivativeActivation);
    };
}


#endif //ANN_CPP_COSTFUNCTION_H
