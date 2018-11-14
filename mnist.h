#ifndef MNIST_H
#define MNIST_H

#include <string>
#include <vector>
#include "training_data.h"

using namespace std;
using namespace linalg;
using namespace annlib;

training_data mnist_load_combined(const string& image_file, const string& label_file);

mat_arr mnist_load_images(const string& file);

vector<int> mnist_load_labels(const string& file);

#endif
