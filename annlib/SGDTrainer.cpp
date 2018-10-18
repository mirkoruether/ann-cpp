#include "SGDTrainer.h"
#include <random>
#include "MatrixMulUtil.h"

using namespace annlib;
using namespace std;

using backpropResult = tuple<vector<DRowVector>, vector<DRowVector>>;

void SGDTrainer::train(const vector<TrainingData>& data, int epochs)
{
	for (int i = 0; i < epochs; ++i)
	{
		trainEpoch(data);
	}
}

void SGDTrainer::trainEpoch(const vector<TrainingData>& data)
{
	for (unsigned batchNo = 0; batchNo < data.size() / miniBatchSize; batchNo++)
	{
		vector<TrainingData> miniBatch = vector<TrainingData>();

		random_device rd;
		mt19937 rng(rd());
		const uniform_int_distribution<unsigned> randomNumber(0, static_cast<unsigned>(data.size()));

		for (unsigned i = 0; i < miniBatchSize; ++i)
		{
			miniBatch.push_back(data[randomNumber(rng)]);
		}

		trainMiniBatch(miniBatch, data.size());
	}
}

void SGDTrainer::trainMiniBatch(const vector<TrainingData>& batch, unsigned trainingSetSize)
{
	vector<backpropResult> backpropResults(batch.size());

	for (unsigned i = 0; i < backpropResults.size(); i++)
	{
		//TODO Parallel
		backpropResults[i] = feedForwardAndBackpropError(batch[i]);
	}

	updateWeights(backpropResults, trainingSetSize);
	updateBiases(backpropResults);
}


backpropResult SGDTrainer::feedForwardAndBackpropError(TrainingData data)
{
	vector<DRowVector> activationsInclInput(biases.size() + 1);
	vector<DRowVector> weightedInput(biases.size());

	activationsInclInput[0] = data.input;
	for (unsigned i = 0; i < biases.size(); i++)
	{
		weightedInput[i] = matrix_Mul(false, false, activationsInclInput[i], weights[i])
		                   .addInPlace(biases[i])
		                   .asRowVector();

		activationsInclInput[i + 1] = weightedInput[i]
		                              .applyFunctionToElements(activationFunction->f)
		                              .asRowVector();
	}

	vector<DRowVector> error(biases.size());

	const unsigned lastLayer = static_cast<unsigned>(error.size()) - 1;

	const DRowVector lastLayerDerivativeActivation = weightedInput[lastLayer]
	                                                 .applyFunctionToElements(activationFunction->df)
	                                                 .asRowVector();

	error[lastLayer] = costFunction->calculateErrorOfLastLayer(activationsInclInput[lastLayer + 1],
	                                                           data.solution,
	                                                           lastLayerDerivativeActivation);

	for (unsigned i = lastLayer - 1; i >= 0; i--)
	{
		error[i] = matrix_Mul(false, true, error[i + 1], weights[i + 1])
		           .elementWiseMulInPlace(weightedInput[i].applyFunctionToElements(activationFunction->df))
		           .asRowVector();
	}

	return backpropResult(activationsInclInput, error);
}

void SGDTrainer::updateWeights(vector<backpropResult> results, unsigned trainingSetSize)
{
	for (unsigned layer = 0; layer < biases.size(); layer++)
	{
		DMatrix decay(weights[layer].getRowCount(), weights[layer].getColumnCount());
		for (auto& result : results)
		{
			decay += matrix_Mul(true, false, get<0>(result)[layer], get<1>(result)[layer]);
		}

		decay *= learningRate / results.size();

		if (costFunctionRegularization != nullptr)
		{
			decay += costFunctionRegularization->calculateWeightDecay(weights[layer], learningRate, trainingSetSize);
		}

		weights[layer] -= decay;
	}
}

void SGDTrainer::updateBiases(vector<backpropResult> results)
{
	for (unsigned layer = 0; layer < biases.size(); layer++)
	{
		DRowVector decay(biases[layer].getColumnCount());
		for (auto& result : results)
		{
			decay += get<1>(result)[layer];
		}

		decay *= learningRate / results.size();

		biases[layer] -= decay;
	}
}

SGDTrainer::SGDTrainer()
	: learningRate(0.1),
	  momentumCoEfficient(0),
	  miniBatchSize(10),
	  activationFunction(make_shared<ActivationFunction>(LogisticActivationFunction(1))),
	  costFunction(make_shared<QuadraticCosts>(QuadraticCosts())),
	  costFunctionRegularization(nullptr) //TODO default init
{
}
