//
// Created by Mirko on 18.05.2018.
//

#include "CostFunction.h"
#include <cmath>


DMatrix annlib::CostFunction::calculateErrorOfLastLayer(const DMatrix &netOutput, const DMatrix &solution,
                                                        const DMatrix &lastLayerDerivativeActivation) {
    return calculateGradient(netOutput, solution).elementWiseMulInPlace(lastLayerDerivativeActivation);
}

double annlib::QuadraticCosts::calculateCosts(const DMatrix &netOutput, const DMatrix &solution) const {
    netOutput.assertSameSize(solution);

    double result = 0.0;
    for (int i = 0; i < netOutput.getLength(); ++i) {
        result += pow(netOutput[i] - solution[i], 2);
    }
    return 0.5 * result;
}

DMatrix annlib::QuadraticCosts::calculateGradient(const DMatrix &netOutput, const DMatrix &solution) const {
    return netOutput - solution;
}
