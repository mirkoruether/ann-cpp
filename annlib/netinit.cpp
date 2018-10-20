#include "netinit.h"
#include <random>
#include <cmath>

using namespace annlib;
using namespace std;
using namespace linalg;

DRowVector GaussianInit::initBiases(unsigned size) const
{
	default_random_engine generator;
	normal_distribution<double> gaussian(0.0, 1.0);
	DRowVector result(size);

	for (unsigned i = 0; i < size; i++)
	{
		result[i] = gaussian(generator);
	}

	return result;
}

DMatrix GaussianInit::initWeights(unsigned inputSize, unsigned outputSize) const
{
	default_random_engine generator;
	normal_distribution<double> gaussian(0.0, 1.0);
	DMatrix result(inputSize, outputSize);

	for (unsigned i = 0; i < inputSize * outputSize; i++)
	{
		result[i] = gaussian(generator);
	}

	return result;
}

DMatrix NormalizedGaussianInit::initWeights(unsigned inputSize, unsigned outputSize) const
{
	return GaussianInit::initWeights(inputSize, outputSize)
		.scalarMulInPlace(1.0 / sqrt(inputSize));
}
