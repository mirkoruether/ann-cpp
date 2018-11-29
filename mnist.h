#ifndef MNIST_H
#define MNIST_H

#include <string>
#include <vector>
#include "training_data.h"
#include "mat_arr.h"

using namespace linalg;
using namespace annlib;

training_data mnist_load_combined(const std::string& image_file, const std::string& label_file);

mat_arr mnist_load_images(const std::string& file);

std::vector<int> mnist_load_labels(const std::string& file);

#endif
