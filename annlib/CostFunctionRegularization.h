//
// Created by Mirko on 16.07.2018.
//

#ifndef ANN_CPP_COSTFUNCTIONREGULARIZATION_H
#define ANN_CPP_COSTFUNCTIONREGULARIZATION_H

#include "DMatrix.h"

using namespace linalg;

namespace annlib {
    class CostFunctionRegularization {
    public:
        virtual DMatrix
        calculateWeightDecay(const DMatrix &weights, double learningRate, int trainingSetSize) const = 0;
    };

    class AbstractCostFunctionRegularization : CostFunctionRegularization {
    private:
        double regularizationParameter;
    protected:
        explicit AbstractCostFunctionRegularization(double regularizationParameter);

    public:
        DMatrix calculateWeightDecay(const DMatrix &weights, double learningRate, int trainingSetSize) const override;

        virtual DMatrix calculateWeightDecay(const DMatrix &weights, double learningRate, int trainingSetSize,
                                             double regularizationParameter) const = 0;

        double getRegularizationParameter() const;

        void setRegularizationParameter(double regularizationParameter);
    };

    class L1Regularization : AbstractCostFunctionRegularization {
    public:
        explicit L1Regularization(double regularizationParameter);

        DMatrix calculateWeightDecay(const DMatrix &weights, double learningRate, int trainingSetSize,
                                     double regularizationParameter) const override;
    };

    class L2Regularization : AbstractCostFunctionRegularization {
    public:
        explicit L2Regularization(double regularizationParameter);

        DMatrix calculateWeightDecay(const DMatrix &weights, double learningRate, int trainingSetSize,
                                     double regularizationParameter) const override;
    };
}


#endif //ANN_CPP_COSTFUNCTIONREGULARIZATION_H
