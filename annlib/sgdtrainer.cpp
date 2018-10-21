#include "sgdtrainer.h"
#include <random>
#include <ppl.h>
#include "matrixmulutil.h"
#include "costfunctionreg.h"
#include <iostream>

#define minibatch_parallel

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
	random_device rd;
	mt19937 rng(rd());
	const uniform_int_distribution<unsigned> randomNumber(0, static_cast<unsigned>(data.size() - 1));

	const unsigned currentAllocs = DMatrix::ALLOCS;
	for (unsigned batchNo = 0; batchNo < data.size() / miniBatchSize; batchNo++)
	{
		vector<backpropResult> backpropResults(miniBatchSize);

#ifdef minibatch_parallel
		Concurrency::parallel_for(size_t(0), backpropResults.size(), [&](size_t i)
		{
			backpropResults[i] = feedForwardAndBackpropError(data[randomNumber(rng)]);
		});
#endif

#ifndef minibatch_parallel
		for(unsigned i = 0; i < backpropResults.size(); i++)
		{
			backpropResults[i] = feedForwardAndBackpropError(data[randomNumber(rng)]);
		}
#endif

		updateWeights(backpropResults, data.size());
		updateBiases(backpropResults);
	}
	cout << "Number of memory allocations by DMatrix constructor" << endl
		<< "    Average per batch training cycle: "
		<< (1.0 * DMatrix::ALLOCS - currentAllocs) / (1.0 * data.size() / miniBatchSize) << endl
		<< "    Total:                            " << DMatrix::ALLOCS << endl;
}

const NeuralNetwork SGDTrainer::toNeuralNet() const
{
	return NeuralNetwork(weights, biases, activationFunction->f);
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

	for (int i = lastLayer - 1; i >= 0; i--)
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

		decay *= learningRate / static_cast<double>(results.size());

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
	: learningRate(5),
	  momentumCoEfficient(0),
	  miniBatchSize(8),
	  activationFunction(make_shared<LogisticActivationFunction>(1.0)),
	  costFunction(make_shared<CrossEntropyCosts>()),
	  costFunctionRegularization(make_shared<L2Regularization>(3.0)),
	  netInit(make_shared<NormalizedGaussianInit>())
{
}

void SGDTrainer::initNet(const vector<unsigned>& sizes)
{
	weights = vector<DMatrix>();
	biases = vector<DRowVector>();
	for (unsigned i = 0; i < sizes.size() - 1; i++)
	{
		weights.emplace_back(netInit->initWeights(sizes[i], sizes[i + 1]));
		biases.emplace_back(netInit->initBiases(sizes[i + 1]));
	}
}
