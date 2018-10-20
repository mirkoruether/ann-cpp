//
// Created by Mirko on 16.07.2018.
//

#include "costfunctionreg.h"


double annlib::AbstractCostFunctionReg::getRegularizationParameter() const {
    return regularizationParameter;
}

void annlib::AbstractCostFunctionReg::setRegularizationParameter(double regularizationParameter) {
    this->regularizationParameter = regularizationParameter;
}

DMatrix annlib::AbstractCostFunctionReg::calculateWeightDecay(const DMatrix &weights, double learningRate,
                                                                         int trainingSetSize) const {
    return calculateWeightDecay(weights, learningRate, trainingSetSize, regularizationParameter);
}

annlib::AbstractCostFunctionReg::AbstractCostFunctionReg(double regularizationParameter)
        : regularizationParameter(regularizationParameter) {}

annlib::L1Regularization::L1Regularization(double regularizationParameter)
        : AbstractCostFunctionReg(regularizationParameter) {}

DMatrix annlib::L1Regularization::calculateWeightDecay(const DMatrix &weights, double learningRate, int trainingSetSize,
                                                       double regularizationParameter) const {
    function<double(double)> sgn = [](double x) { return x > 0 ? 1.0 : -1.0; };
    return weights.applyFunctionToElements(sgn)
            .scalarMulInPlace(learningRate * regularizationParameter / trainingSetSize);
}

annlib::L2Regularization::L2Regularization(double regularizationParameter)
        : AbstractCostFunctionReg(regularizationParameter) {}

DMatrix annlib::L2Regularization::calculateWeightDecay(const DMatrix &weights, double learningRate, int trainingSetSize,
                                                       double regularizationParameter) const {
    return weights * (learningRate * regularizationParameter / trainingSetSize);
}
