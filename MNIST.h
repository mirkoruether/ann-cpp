#ifndef MNIST_H
#define MNIST_H

#include <string>
#include "DMatrix.h"

using namespace std;
using namespace linalg;

vector<tuple<DRowVector, DRowVector>> MNISTLoadCombined(const string& imageFile, const string& labelFile);

vector<DRowVector> MNISTLoadImages(const string& file);

vector<int> MNISTLoadLabels(const string& file);

#endif
