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
        virtual double calculateCosts(const DRowVector &netOutput, const DRowVector &solution) const = 0;

        virtual DRowVector calculateGradient(const DRowVector &netOutput, const DRowVector &solution) const = 0;

        virtual DRowVector calculateErrorOfLastLayer(const DRowVector &netOutput, const DRowVector &solution,
                                                     const DRowVector &lastLayerDerivativeActivation);
    };

    class QuadraticCosts : public CostFunction {
    public:
        double calculateCosts(const DRowVector &netOutput, const DRowVector &solution) const override;

        DRowVector calculateGradient(const DRowVector &netOutput, const DRowVector &solution) const override;
    };
}


#endif //ANN_CPP_COSTFUNCTION_H
