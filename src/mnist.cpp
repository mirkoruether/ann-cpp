#include "mnist.h"
#include <fstream>

using namespace linalg;
using namespace annlib;

unsigned read_next_int(std::ifstream* ifs)
{
	char buffer[4];
	ifs->read(buffer, 4);

	return static_cast<unsigned>(
		(buffer[0] & 0xff) << 24
			| (buffer[1] & 0xff) << 16
			| (buffer[2] & 0xff) << 8
			| (buffer[3] & 0xff) << 0);
}

unsigned read_next_byte(std::ifstream* ifs)
{
	char buffer[1];
	ifs->read(buffer, 1);

	return static_cast<unsigned >(buffer[0] & 0xff);
}

classification_data mnist_load_combined(const std::string& image_file, const std::string& label_file)
{
	mat_arr images = mnist_load_images(image_file);
	std::vector<unsigned> labels = mnist_load_labels(label_file);

	if (labels.size() != images.count)
	{
		throw std::runtime_error("Image and label size do not match");
	}

	return classification_data(std::move(images), std::move(labels));
}

mat_arr mnist_load_images(const std::string& file)
{
	std::ifstream ifs = std::ifstream(file.c_str(), std::ios::in | std::ios::binary);

	const unsigned magicNum = read_next_int(&ifs);

	if (magicNum != 2051)
	{
		ifs.close();
		throw std::runtime_error("This file is no image file");
	}

	const unsigned count = read_next_int(&ifs);
	const unsigned rows = read_next_int(&ifs);
	const unsigned cols = read_next_int(&ifs);
	const unsigned size = count * rows * cols;

	mat_arr images(count, 1, rows * cols);
	fpt* im = images.start();

	for (unsigned i = 0; i < size; i++)
	{
		im[i] = read_next_byte(&ifs) / 255.0f;
	}

	ifs.close();
	return images;
}

std::vector<unsigned> mnist_load_labels(const std::string& file)
{
	std::ifstream ifs = std::ifstream(file.c_str(), std::ios::in | std::ios::binary);

	const unsigned magicNum = read_next_int(&ifs);

	if (magicNum != 2049)
	{
		ifs.close();
		throw std::runtime_error("This file is no label file");
	}

	const unsigned count = read_next_int(&ifs);

	std::vector<unsigned> labels(count);
	for (unsigned i = 0; i < count; i++)
	{
		labels[i] = read_next_byte(&ifs);
	}
	ifs.close();
	return labels;
}
