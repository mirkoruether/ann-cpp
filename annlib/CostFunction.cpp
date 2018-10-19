#include "CostFunction.h"
#include <cmath>


DRowVector annlib::CostFunction::calculateErrorOfLastLayer(const DRowVector& netOutput, const DRowVector& solution,
                                                           const DRowVector& lastLayerDerivativeActivation) const
{
	return static_cast<DRowVector>(calculateGradient(netOutput, solution)
		.elementWiseMulInPlace(lastLayerDerivativeActivation));
}

double annlib::QuadraticCosts::calculateCosts(const DRowVector& netOutput, const DRowVector& solution) const
{
	netOutput.assertSameSize(solution);

	double result = 0.0;
	for (unsigned i = 0; i < netOutput.getLength(); ++i)
	{
		result += pow(netOutput[i] - solution[i], 2);
	}
	return 0.5 * result;
}

DRowVector annlib::QuadraticCosts::calculateGradient(const DRowVector& netOutput, const DRowVector& solution) const
{
	return static_cast<DRowVector>(netOutput - solution);
}

double annlib::CrossEntropyCosts::calculateCosts(const DRowVector& netOutput, const DRowVector& solution) const
{
	double result = 0;
	for (unsigned i = 0; i < netOutput.getLength(); i++)
	{
		double a = netOutput[i];
		double y = solution[i];
		result += y * log(a) + (1 - y) * log(1 - a);
	}
	return -result;
}

DRowVector annlib::CrossEntropyCosts::calculateGradient(const DRowVector& netOutput, const DRowVector& solution) const
{
	const unsigned le = netOutput.getLength();
	const DRowVector& a = netOutput;
	const DRowVector& y = solution;
	// y/a - (1-y)/(1-a) element wise
	DRowVector grad = y.elementWiseDiv(a)
	                   .subInPlace(DMatrix::ones(1, le).subInPlace(y)
	                                                   .elementWiseDiv(DMatrix::ones(1, le).subInPlace(a)))
	                   .asRowVector();
	return grad;
}

DRowVector annlib::CrossEntropyCosts::calculateErrorOfLastLayer(const DRowVector& netOutput, const DRowVector& solution,
                                                                const DRowVector& lastLayerDerivativeActivation) const
{
	return (netOutput - solution).asRowVector();
}
