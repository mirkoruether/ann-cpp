#include "MNIST.h"
#include "DMatrix.h"
#include <fstream>

using namespace std;
using namespace linalg;


int readNextInt(ifstream* ifs)
{
	char buffer[4];
	ifs->read(buffer, 4);

	return (buffer[0] & 0xff) << 24
		| (buffer[1] & 0xff) << 16
		| (buffer[2] & 0xff) << 8
		| (buffer[3] & 0xff) << 0;
}

vector<tuple<DRowVector, DRowVector>> MNISTLoadCombined(const string& imageFile, const string& labelFile)
{
	vector<DRowVector> images = MNISTLoadImages(imageFile);
	vector<int> labels = MNISTLoadLabels(labelFile);

	if (images.size() != labels.size())
	{
		throw runtime_error("Image and label size do not match");
	}

	vector<tuple<DRowVector, DRowVector>> result(images.size());
	for (unsigned i = 0; i < result.size(); i++)
	{
		DRowVector label(10);
		label[labels[i]] = 1.0;
		result[i] = tuple<DRowVector, DRowVector>(images[i], label);
	}
	return result;
}

vector<DRowVector> MNISTLoadImages(const string& file)
{
	ifstream ifs = ifstream(file.c_str(), ios::in | ios::binary);

	const int magicNum = readNextInt(&ifs);

	if (magicNum != 2051)
	{
		ifs.close();
		throw runtime_error("This file is no image file");
	}

	const int count = readNextInt(&ifs);
	const int rows = readNextInt(&ifs);
	const int cols = readNextInt(&ifs);

	vector<DRowVector> images(count);

	for (int i = 0; i < count; i++)
	{
		DRowVector image(rows * cols);
		for (int j = 0; j < rows * cols; j++)
		{
			image[j] = readNextInt(&ifs) / 255.0;
		}
		images.push_back(image);
	}

	ifs.close();
	return images;
}

vector<int> MNISTLoadLabels(const string& file)
{
	ifstream ifs = ifstream(file.c_str(), ios::in | ios::binary);

	const int magicNum = readNextInt(&ifs);

	if (magicNum != 2049)
	{
		ifs.close();
		throw runtime_error("This file is no label file");
	}

	const int count = readNextInt(&ifs);

	vector<int> labels(count);

	for (int i = 0; i < count; i++)
	{
		labels.push_back(readNextInt(&ifs));
	}

	ifs.close();
	return labels;
}
