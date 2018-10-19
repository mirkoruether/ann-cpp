#ifndef ANN_CPP_MOMENTUMSTOCHASTICGRADIENTDESCENTTRAINER_H
#define ANN_CPP_MOMENTUMSTOCHASTICGRADIENTDESCENTTRAINER_H

#include <vector>
#include "DMatrix.h"
#include "ActivationFunction.h"
#include "CostFunction.h"
#include "TrainingData.h"
#include "CostFunctionRegularization.h"
#include "NeuralNetwork.h"

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
		shared_ptr<ActivationFunction> activationFunction;
		shared_ptr<CostFunction> costFunction;
		shared_ptr<CostFunctionRegularization> costFunctionRegularization;

		SGDTrainer(const vector<unsigned>& sizes);

		void train(const vector<TrainingData>& data, int epochs);

		void trainEpoch(const vector<TrainingData>& data);

		void trainMiniBatch(const vector<TrainingData>& batch, unsigned trainingSetSize);

		const NeuralNetwork toNeuralNet() const;
	};
}

#endif //ANN_CPP_MOMENTUMSTOCHASTICGRADIENTDESCENTTRAINER_H
