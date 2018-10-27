#include <iostream>
#include <chrono>
#include <dmatrix.h>
#include <neuralnetwork.h>
#include "sgdtrainer.h"
#include "mnist.h"
#include "mat_arr.h"
#include "mat_arr_math.h"
#include <cassert>

typedef std::chrono::high_resolution_clock Clock;

using namespace std;
using namespace chrono;
using namespace linalg;

void randomMatrixArr(mat_arr* m)
{
	const unsigned size = m->size();
	double* mat = m->start();
	for (unsigned i = 0; i < size; i++)
	{
		*(mat + i) = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
	}
}

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

void timeExecution(const string& name, const function<void()>& func)
{
	cout << name << " started" << endl;
	const auto calcStart = Clock::now();
	func();
	const double gpu_calc_millis = duration_cast<nanoseconds>(Clock::now() - calcStart).count() / double(1e6);
	cout << name << " finished, time elapsed: " << gpu_calc_millis << " ms" << endl;
}

void trainAndTest()
{
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
}

void speedAddTranspose()
{
	mat_arr a(500, 1000, 100);
	mat_arr b(500, 1000, 100);
	mat_arr a_t(500, 100, 1000);
	mat_arr b_t(500, 100, 1000);
	mat_arr c(500, 1000, 100);

	mat_arr* c_addr = &c;
	mat_arr* a_t_addr = &a_t;

	randomMatrixArr(&a);
	randomMatrixArr(&b);
	randomMatrixArr(&b_t);
	randomMatrixArr(&c);

	timeExecution("transpose", [a, a_t_addr]() { mat_transpose(a, a_t_addr); });
	timeExecution("noTranspose add", [a, b, c_addr]() { mat_element_wise_add(a, b, c_addr, transpose_no); });
	timeExecution("transposeA add", [a_t, b, c_addr]() { mat_element_wise_add(a_t, b, c_addr, transpose_A); });
	timeExecution("transposeB add", [a, b_t, c_addr]() { mat_element_wise_add(a, b_t, c_addr, transpose_B); });
	timeExecution("transposeBoth add", [a_t, b_t, c_addr]()
	{
		mat_element_wise_add(a_t, b_t, c_addr, transpose_both);
	});
}

void testTranspose()
{
	mat_arr A(1, 2, 3);
	double* a = A.start();
	*(a + 0) = 0;
	*(a + 1) = 1;
	*(a + 2) = 2;
	*(a + 3) = 3;
	*(a + 4) = 4;
	*(a + 5) = 5;

	mat_arr C(1, 3, 2);
	mat_transpose(A, &C);
	double* c = C.start();

	assert(*(c+0) == 0);
	assert(*(c+1) == 3);
	assert(*(c+2) == 1);
	assert(*(c+3) == 4);
	assert(*(c+4) == 2);
	assert(*(c+5) == 5);
}

void __testAddTranspose_check(double* c)
{
	assert(*(c + 0) == 12);
	assert(*(c + 1) == 15);
	assert(*(c + 2) == 18);
	assert(*(c + 3) == 21);
	assert(*(c + 4) == 24);
	assert(*(c + 5) == 27);
}

void testAddTranspose()
{
	mat_arr A(1, 2, 3);
	double* a = A.start();
	*(a + 0) = 0;
	*(a + 1) = 1;
	*(a + 2) = 2;
	*(a + 3) = 3;
	*(a + 4) = 4;
	*(a + 5) = 5;

	mat_arr B(1, 2, 3);
	double* b = B.start();
	*(b + 0) = 12;
	*(b + 1) = 14;
	*(b + 2) = 16;
	*(b + 3) = 18;
	*(b + 4) = 20;
	*(b + 5) = 22;

	mat_arr A_t(1, 3, 2);
	mat_transpose(A, &A_t);

	mat_arr B_t(1, 3, 2);
	mat_transpose(B, &B_t);

	mat_arr C(1, 2, 3);
	mat_element_wise_add(A, B, &C);
	__testAddTranspose_check(C.start());

	mat_element_wise_add(A_t, B, &C, transpose_A);
	__testAddTranspose_check(C.start());

	mat_element_wise_add(A, B_t, &C, transpose_B);
	__testAddTranspose_check(C.start());

	mat_element_wise_add(A_t, B_t, &C, transpose_both);
	__testAddTranspose_check(C.start());
}

void testMatMul()
{
	mat_arr A(1, 2, 3);
	double* a = A.start();
	*(a + 0) = 0;
	*(a + 1) = 1;
	*(a + 2) = 2;
	*(a + 3) = 3;
	*(a + 4) = 4;
	*(a + 5) = 5;

	mat_arr B(1, 3, 2);
	double* b = B.start();
	*(b + 0) = 12;
	*(b + 1) = 14;
	*(b + 2) = 16;
	*(b + 3) = 18;
	*(b + 4) = 20;
	*(b + 5) = 22;

	mat_arr C(1, 2, 2);
	mat_matrix_mul(A, B, &C);
	double* c = C.start();

	assert(*(c + 0) == 56);
	assert(*(c + 1) == 62);
	assert(*(c + 2) == 200);
	assert(*(c + 3) == 224);
}

int main()
{
	timeExecution("testTranspose ", []() { testTranspose(); });
	timeExecution("testAddTranspose ", []() { testAddTranspose(); });
	timeExecution("testMatrixMul ", []() { testMatMul(); });
	//timeExecution("speedAddTranspose ", []() { speedAddTranspose(); });
	cin.get();
}
