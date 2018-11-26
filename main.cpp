#include <iostream>
#include <chrono>
#include "mnist.h"
#include "mat_arr.h"
#include "mat_arr_math.h"
#include "sgd_trainer.h"

typedef std::chrono::high_resolution_clock Clock;

using namespace std;
using namespace chrono;
using namespace linalg;

void random_matrix_arr(mat_arr* m)
{
	const unsigned size = m->size();
	double* mat = m->start();
	for (unsigned i = 0; i < size; i++)
	{
		*(mat + i) = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
	}
}

unsigned get_max_index(const mat_arr& vec)
{
	double max = -1000.0;
	unsigned maxIndex = 0;
	for (unsigned i = 0; i < vec.size(); i++)
	{
		if (vec[i] > max)
		{
			max = vec[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}

template <typename T>
T time_execution_func(const string& name, const function<T()>& func)
{
	cout << name << " started" << endl;
	const auto calcStart = Clock::now();
	T result = func();
	const double gpu_calc_millis = duration_cast<nanoseconds>(Clock::now() - calcStart).count() / double(1e6);
	cout << name << " finished, time elapsed: " << gpu_calc_millis << " ms" << endl;
	return result;
}

void time_execution(const string& name, const function<void()>& func)
{
	time_execution_func<bool>(name, [&]()
	{
		func();
		return true;
	});
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

	random_matrix_arr(&a);
	random_matrix_arr(&b);
	random_matrix_arr(&b_t);
	random_matrix_arr(&c);

	time_execution("transpose", [a, a_t_addr]() { mat_transpose(a, a_t_addr); });
	time_execution("noTranspose add", [a, b, c_addr]() { mat_element_wise_add(a, b, c_addr, transpose_no); });
	time_execution("transposeA add", [a_t, b, c_addr]() { mat_element_wise_add(a_t, b, c_addr, transpose_A); });
	time_execution("transposeB add", [a, b_t, c_addr]() { mat_element_wise_add(a, b_t, c_addr, transpose_B); });
	time_execution("transposeBoth add", [a_t, b_t, c_addr]()
	{
		mat_element_wise_add(a_t, b_t, c_addr, transpose_both);
	});
}

double test_network_accuracy(neural_network net, training_data test_data)
{
	const mat_arr net_output = net.feed_forward(test_data.input);
	unsigned correct = 0;
	for (unsigned i = 0; i < net_output.count; i++)
	{
		const mat_arr& output = net_output.get_mat(i);
		const mat_arr& solution = test_data.solution.get_mat(i);
		if (get_max_index(output) == get_max_index(solution))
		{
			correct++;
		}
	}
	return (100.0 * correct) / net_output.count;
}

int main()
{
	const string folder = "C:\\";
	const string training_images = folder + "train-images.idx3-ubyte";
	const string training_labels = folder + "train-labels.idx1-ubyte";
	const string test_images = folder + "t10k-images.idx3-ubyte";
	const string test_labels = folder + "t10k-labels.idx1-ubyte";

	const training_data mnist_training
		= time_execution_func<training_data>("Load MNIST training data", [&]()
		{
			return mnist_load_combined(training_images, training_labels);
		});

	const training_data mnist_test
		= time_execution_func<training_data>("Load MNIST test data", [&]()
		{
			return mnist_load_combined(test_images, test_labels);
		});

	sgd_trainer trainer;
	trainer.mini_batch_size = 8;
	trainer.activation_f = make_shared<logistic_activation_function>(1.0);
	trainer.cost_f = make_shared<cross_entropy_costs>();
	trainer.weight_norm_penalty = make_shared<L2_regularization>(3.0 / mnist_training.entry_count());
	trainer.optimizer = make_shared<momentum_sgd>(5.0, 0.0);
	trainer.net_init = make_shared<normalized_gaussian_net_init>();

	vector<unsigned> sizes{{784, 30, 10}};
	trainer.init(sizes);

	time_execution("Train five epochs", [&]()
	{
		trainer.train_epochs(mnist_training, 5);
	});

	neural_network net = trainer.to_neural_network(true);

	const auto accuracy = time_execution_func<double>("Test", [&]()
	{
		return test_network_accuracy(net, mnist_test);
	});

	cout << "Test accuracy: " << accuracy << "%" << endl;
	cin.get();
}
