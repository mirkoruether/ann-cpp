#include <iostream>
#include <chrono>
#include <dmatrix.h>
#include <neuralnetwork.h>
#include "sgdtrainer.h"
#include "mnist.h"

typedef std::chrono::high_resolution_clock Clock;

using namespace std;
using namespace chrono;
using namespace linalg;

unsigned getMaxIndex(const DRowVector& vec)
{
	double max = -1000.0;
	unsigned maxIndex = 0;
	for (unsigned i = 0; i < vec.getLength(); i++)
	{
		if (vec[i] > max)
		{
			max = vec[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}

int main()
{
	//TODO add paths
	const string folder = "C:\\";
	const string trainingImages = folder + "train-images.idx3-ubyte";
	const string trainingLabels = folder + "train-labels.idx1-ubyte";
	const string testImages = folder + "t10k-images.idx3-ubyte";
	const string testLabels = folder + "t10k-labels.idx1-ubyte";

	vector<unsigned> sizes(3);
	sizes[0] = 784;
	sizes[1] = 30;
	sizes[2] = 10;

	cout << "Init trainer & net" << endl;
	SGDTrainer trainer;
	trainer.initNet(sizes);
	cout << "Finished" << endl;

	cout << "Load training data" << endl;
	const vector<TrainingData> trainingData = MNISTLoadCombined(trainingImages, trainingLabels);
	cout << "Finished" << endl;

	cout << "Load test data" << endl;
	const vector<TrainingData> testData = MNISTLoadCombined(testImages, testLabels);
	cout << "Finished" << endl;

	cout << "Train 5 epochs" << endl;
	const auto calcStart = Clock::now();

	for (int i = 0; i < 5; i++)
	{
		trainer.trainEpoch(trainingData);
	}

	const double gpu_calc_millis = duration_cast<nanoseconds>(Clock::now() - calcStart).count() / double(1e6);
	cout << "Finished, time elapsed: " << gpu_calc_millis << " ms" << endl;


	cout << "Test" << endl;
	NeuralNetwork net = trainer.toNeuralNet();

	unsigned correct = 0;
	for (const auto& testEntry : testData)
	{
		const DRowVector netOutput = net.feedForward(testEntry.input);
		if (getMaxIndex(netOutput) == getMaxIndex(testEntry.solution))
		{
			correct++;
		}
	}
	cout << "Finished" << endl;
	cout << "Test accuracy: " << (100.0 * correct) / testData.size() << "%" << endl;

	cin.get();
}
