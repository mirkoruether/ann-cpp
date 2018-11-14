#include "mnist.h"
#include <fstream>

using namespace std;
using namespace linalg;

int read_next_int(ifstream* ifs)
{
	char buffer[4];
	ifs->read(buffer, 4);

	return (buffer[0] & 0xff) << 24
		| (buffer[1] & 0xff) << 16
		| (buffer[2] & 0xff) << 8
		| (buffer[3] & 0xff) << 0;
}

int read_next_byte(ifstream* ifs)
{
	char buffer[1];
	ifs->read(buffer, 1);

	return buffer[0] & 0xff;
}

training_data mnist_load_combined(const string& image_file, const string& label_file)
{
	const mat_arr images = mnist_load_images(image_file);
	vector<int> labels = mnist_load_labels(label_file);

	if (images.count != labels.size())
	{
		throw runtime_error("Image and label size do not match");
	}

	mat_arr solution(labels.size(), 1, 10);
	double* sol = solution.start();
	for (unsigned i = 0; i < solution.count; i++)
	{
		*(sol + labels[i]) = 1.0;
		sol += 10;
	}
	return training_data(images, solution);
}

mat_arr mnist_load_images(const string& file)
{
	ifstream ifs = ifstream(file.c_str(), ios::in | ios::binary);

	const int magicNum = read_next_int(&ifs);

	if (magicNum != 2051)
	{
		ifs.close();
		throw runtime_error("This file is no image file");
	}

	const int count = read_next_int(&ifs);
	const int rows = read_next_int(&ifs);
	const int cols = read_next_int(&ifs);
	const int size = count * rows * cols;

	mat_arr images(count, 1, rows * cols);
	double* im = images.start();

	for (int i = 0; i < size; i++)
	{
		*(im + i) = read_next_byte(&ifs) / 255.0;
	}

	ifs.close();
	return images;
}

vector<int> mnist_load_labels(const string& file)
{
	ifstream ifs = ifstream(file.c_str(), ios::in | ios::binary);

	const int magicNum = read_next_int(&ifs);

	if (magicNum != 2049)
	{
		ifs.close();
		throw runtime_error("This file is no label file");
	}

	const int count = read_next_int(&ifs);

	vector<int> labels(count);
	for (int i = 0; i < count; i++)
	{
		labels[i] = read_next_byte(&ifs);
	}

	ifs.close();
	return labels;
}
