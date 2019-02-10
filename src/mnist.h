#ifndef MNIST_H
#define MNIST_H

#include <string>
#include <vector>
#include "unambiguous_classification.h"
#include "mat_arr.h"

using namespace linalg;
using namespace annlib;
using namespace annlib::tasks;

classification_data mnist_load_combined(const std::string& image_file, const std::string& label_file);

mat_arr mnist_load_images(const std::string& file);

std::vector<unsigned> mnist_load_labels(const std::string& file);

#endif
