#include <iostream>
#include <DMatrix.h>
#include <NeuralNetwork.h>
#include "SGDTrainer.h"
#include "MNIST.h"

using namespace std;
using namespace linalg;

using MNIST_Data = vector<tuple<DRowVector, DRowVector>>;

int main()
{
	//TODO add paths
	const string trainingImages = "";
	const string trainingLabels = "";
	const string testImages = "";
	const string testLabels = "";

	vector<unsigned> sizes(3);
	sizes[0] = 784;
	sizes[1] = 100;
	sizes[2] = 10;

	SGDTrainer trainer(sizes);
	trainer.costFunction = make_shared<CrossEntropyCosts>(CrossEntropyCosts());

	MNIST_Data training = MNISTLoadCombined(trainingImages, trainingLabels);
}
