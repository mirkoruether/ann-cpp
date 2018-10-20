#ifndef ANN_CPP_NETWORKLAYER_H
#define ANN_CPP_NETWORKLAYER_H

#include "dmatrix.h"
#include "activationfunction.h"

using namespace linalg;
using namespace std;

namespace annlib
{
	class NetworkLayer
	{
	private:
		DRowVector biases;
		DMatrix weights;
		function<double(double)> activationFunction;
	public:
		NetworkLayer(const DMatrix& weights, const DRowVector& biases,
		             function<double(double)> activationFunction);

		unsigned getInputSize() const;

		unsigned getOutputSize() const;

		const DRowVector getBiases() const;

		DRowVector getBiases();

		const DMatrix getWeights() const;

		DMatrix getWeights();

		function<double(double)> getActivationFunction() const;

		DRowVector feedForward(const DRowVector& in) const;
	};
}

#endif //ANN_CPP_NETWORKLAYER_H
