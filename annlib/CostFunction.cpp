#include "CostFunction.h"
#include <cmath>


DRowVector annlib::CostFunction::calculateErrorOfLastLayer(const DRowVector &netOutput, const DRowVector &solution,
                                                           const DRowVector &lastLayerDerivativeActivation) {
    return static_cast<DRowVector>(calculateGradient(netOutput, solution)
	    .elementWiseMulInPlace(lastLayerDerivativeActivation));
}

double annlib::QuadraticCosts::calculateCosts(const DRowVector &netOutput, const DRowVector &solution) const {
    netOutput.assertSameSize(solution);

    double result = 0.0;
    for (unsigned i = 0; i < netOutput.getLength(); ++i) {
        result += pow(netOutput[i] - solution[i], 2);
    }
    return 0.5 * result;
}

DRowVector annlib::QuadraticCosts::calculateGradient(const DRowVector &netOutput, const DRowVector &solution) const {
    return static_cast<DRowVector>(netOutput - solution);
}
