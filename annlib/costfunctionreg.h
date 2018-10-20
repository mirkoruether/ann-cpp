#ifndef ANN_CPP_COSTFUNCTIONREG_H
#define ANN_CPP_COSTFUNCTIONREG_H

#include "dmatrix.h"

using namespace linalg;

namespace annlib
{
	class CostFunctionReg
	{
	public:
		virtual ~CostFunctionReg() = default;
		virtual DMatrix
		calculateWeightDecay(const DMatrix& weights, double learningRate, int trainingSetSize) const = 0;
	};

	class AbstractCostFunctionReg : public CostFunctionReg
	{
	private:
		double regularizationParameter;
	protected:
		explicit AbstractCostFunctionReg(double regularizationParameter);

	public:
		DMatrix calculateWeightDecay(const DMatrix& weights, double learningRate, int trainingSetSize) const override;

		virtual DMatrix calculateWeightDecay(const DMatrix& weights, double learningRate, int trainingSetSize,
		                                     double regularizationParameter) const = 0;

		double getRegularizationParameter() const;

		void setRegularizationParameter(double regularizationParameter);
	};

	class L1Regularization : public AbstractCostFunctionReg
	{
	public:
		explicit L1Regularization(double regularizationParameter);

		DMatrix calculateWeightDecay(const DMatrix& weights, double learningRate, int trainingSetSize,
		                             double regularizationParameter) const override;
	};

	class L2Regularization : public AbstractCostFunctionReg
	{
	public:
		explicit L2Regularization(double regularizationParameter);

		DMatrix calculateWeightDecay(const DMatrix& weights, double learningRate, int trainingSetSize,
		                             double regularizationParameter) const override;
	};
}


#endif //ANN_CPP_COSTFUNCTIONREG_H
