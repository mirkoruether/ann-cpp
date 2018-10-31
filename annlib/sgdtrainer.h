#ifndef ANN_CPP_MOMENTUMSTOCHASTICGRADIENTDESCENTTRAINER_H
#define ANN_CPP_MOMENTUMSTOCHASTICGRADIENTDESCENTTRAINER_H

#include <vector>
#include "dmatrix.h"
#include "activation_function.h"
#include "costfunction.h"
#include "trainingdata.h"
#include "costfunctionreg.h"
#include "netinit.h"
#include "neuralnetwork.h"

using namespace std;
using namespace linalg;

using backpropResult = tuple<vector<DRowVector>, vector<DRowVector>>;

namespace annlib
{
	class SGDTrainer
	{
	private:
		vector<DMatrix> weights;
		vector<DRowVector> biases;

		backpropResult feedForwardAndBackpropError(TrainingData data);

		void updateWeights(vector<backpropResult> results, unsigned trainingSetSize);
		void updateBiases(vector<backpropResult> results);

	public:

		//------------------
		// Hyper Parameters
		//------------------
		double learningRate;
		double momentumCoEfficient;
		unsigned miniBatchSize;
		shared_ptr<activation_function> activationFunction;
		shared_ptr<CostFunction> costFunction;
		shared_ptr<CostFunctionReg> costFunctionRegularization;
		shared_ptr<NetInit> netInit;

		SGDTrainer();

		void initNet(const vector<unsigned>& sizes);

		void train(const vector<TrainingData>& data, int epochs);

		void trainEpoch(const vector<TrainingData>& data);

		const NeuralNetwork toNeuralNet() const;
	};
}

#endif //ANN_CPP_MOMENTUMSTOCHASTICGRADIENTDESCENTTRAINER_H
