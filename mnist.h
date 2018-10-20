#ifndef MNIST_H
#define MNIST_H

#include <string>
#include "dmatrix.h"
#include "trainingdata.h"

using namespace std;
using namespace linalg;
using namespace annlib;

vector<TrainingData> MNISTLoadCombined(const string& imageFile, const string& labelFile);

vector<DRowVector> MNISTLoadImages(const string& file);

vector<int> MNISTLoadLabels(const string& file);

#endif
