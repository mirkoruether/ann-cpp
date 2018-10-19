#include "NetworkLayer.h"
#include <utility>
#include "MatrixMulUtil.h"

using namespace linalg;
using namespace std;

namespace annlib
{
	NetworkLayer::NetworkLayer(const DMatrix& weights, const DRowVector& biases,
	                           function<double(double)> activationFunction)
		: biases(biases), weights(weights), activationFunction(move(activationFunction))
	{
		if (!biases.isRowVector() || biases.getLength() != weights.getColumnCount())
		{
			throw runtime_error("column count of weights and length of biases differ");
		}
	}

	unsigned NetworkLayer::getInputSize() const
	{
		return weights.getRowCount();
	}

	unsigned NetworkLayer::getOutputSize() const
	{
		return biases.getLength();
	}

	const DRowVector NetworkLayer::getBiases() const
	{
		return biases;
	}

	DRowVector NetworkLayer::getBiases()
	{
		return biases;
	}

	const DMatrix NetworkLayer::getWeights() const
	{
		return weights;
	}

	DMatrix NetworkLayer::getWeights()
	{
		return weights;
	}

	function<double(double)> NetworkLayer::getActivationFunction() const
	{
		return activationFunction;
	}

	DRowVector NetworkLayer::feedForward(const DRowVector& in) const
	{
		if (!in.isRowVector() || in.getLength() != getInputSize())
		{
			throw runtime_error("wrong input dimensions");
		}

		return matrix_Mul(false, false, in, weights)
		       .addInPlace(biases)
		       .applyFunctionToElementsInPlace(activationFunction)
		       .asRowVector();
	}
}
