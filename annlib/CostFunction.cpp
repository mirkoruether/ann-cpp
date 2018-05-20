//
// Created by Mirko on 18.05.2018.
//

#include "CostFunction.h"


DMatrix annlib::CostFunction::calculateErrorOfLastLayer(const DMatrix &netOutput, const DMatrix &solution,
                                                        const DMatrix &lastLayerDerivativeActivation) {
    return calculateGradient(netOutput, solution).elementWiseMulInPlace(lastLayerDerivativeActivation);
}
