#include "mat_arr.h"
#include "mnist.h"
#include "sgd_trainer.h"
#include <chrono>
#include <iostream>
#include <thread>
#include <string>
#include "output_layer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

typedef std::chrono::high_resolution_clock Clock;

using namespace annlib;
using namespace std::chrono;
using namespace linalg;

void random_matrix_arr(mat_arr* m)
{
	const unsigned size = m->size();
	fpt* mat = m->start();
	for (unsigned i = 0; i < size; i++)
	{
		*(mat + i) = static_cast<fpt>(rand()) / static_cast<fpt>(RAND_MAX);
	}
}

unsigned get_max_index(const mat_arr& vec)
{
	fpt max = -1.0f * std::numeric_limits<fpt>::infinity();
	unsigned maxIndex = std::numeric_limits<unsigned>::max();
	for (unsigned i = 0; i < vec.size(); i++)
	{
		if (std::isfinite(vec[i]) && vec[i] > max)
		{
			max = vec[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}

template <typename T>
T time_execution_func(const std::string& name, const std::function<T()>& func)
{
	std::cout << name.c_str() << " started" << std::endl;
	const auto calcStart = Clock::now();
	T result = func();
	const double gpu_calc_millis = duration_cast<nanoseconds>(Clock::now() - calcStart).count() / double(1e6);
	std::cout << name.c_str() << " finished, time elapsed: " << gpu_calc_millis << " ms" << std::endl;
	return result;
}

void time_execution(const std::string& name, const std::function<void()>& func)
{
	time_execution_func<bool>(name, [&]()
	{
		func();
		return true;
	});
}

void print_settings()
{
	std::cout << "ann-cpp demo app" << std::endl << std::endl;

	const unsigned n_threads = std::thread::hardware_concurrency();
	std::cout << n_threads << " concurrent threads are supported.\n"
		<< std::endl;

#ifdef _OPENMP
	std::cout << "OpenMP enabled, launching four test threads" << std::endl;
#pragma omp parallel num_threads(4)
	{
#pragma omp critical
		std::cout << "Hello, I am thread number " << omp_get_thread_num() << std::endl;
	}
#else
	std::cout << "OpenMP disabled" << std::endl;
#endif

	std::cout << std::endl;

#ifdef LINALG_CUDA_SUPPORT
	std::cout << "CUDA supported" << std::endl;
#else
	std::cout << "CUDA not supported" << std::endl;
#endif

	std::cout << std::endl;

#ifdef ANNLIB_USE_CUDA
	std::cout << "CUDA enabled" << std::endl;
#else
	std::cout << "CUDA disabled" << std::endl;
#endif

	std::cout << std::endl;

#ifdef __linux__
	std::cout << "LINUX" << std::endl;
#else
	std::cout << "WINDOWS" << std::endl;
#endif

	std::cout << std::endl;
}

struct test_result
{
	double costs;
	double accuracy;
};

test_result test_network(const sgd_trainer& trainer, const training_data& data)
{
	const mat_arr net_output = trainer.feed_forward(data.input);
	double costs = 0.0;
	unsigned correct = 0;
	for (unsigned i = 0; i < net_output.count; i++)
	{
		const mat_arr& output = net_output.get_mat(i);
		const mat_arr& solution = data.solution.get_mat(i);

		costs += trainer.calculate_costs(output, solution);

		if (get_max_index(output) == get_max_index(solution))
		{
			correct++;
		}
	}

	return test_result{
		costs / net_output.count,
		100.0 * correct / net_output.count
	};
}

struct cycle_result
{
	double training_costs;
	double training_accuracy;
	double test_costs;
	double test_accuracy;

	bool nan_detected;
};

cycle_result train_and_test(unsigned mini_batch_size, double epochs_per_cycle,
                            sgd_trainer* trainer,
                            gradient_based_optimizer* opt,
                            const training_data& train_data,
                            const training_data& test_data)
{
	trainer->train_epochs(train_data, opt, mini_batch_size, epochs_per_cycle);

	const test_result train_result = test_network(*trainer, train_data);
	const test_result test_result = test_network(*trainer, test_data);

	return cycle_result{
		train_result.costs,
		train_result.accuracy,
		test_result.costs,
		test_result.accuracy,
		false
	};
}

int main(int argc, char** argv)
{
	double epoch_count = argc <= 1 ? 1 : std::stod(std::string(argv[1]));

#ifdef __linux__
	const std::string folder = "/mnt/c/";
#else
	const std::string folder = "C:\\\\";
#endif

	const std::string training_images = folder + "train-images.idx3-ubyte";
	const std::string training_labels = folder + "train-labels.idx1-ubyte";
	const std::string test_images = folder + "t10k-images.idx3-ubyte";
	const std::string test_labels = folder + "t10k-labels.idx1-ubyte";

	const training_data mnist_training = time_execution_func<training_data>("Load MNIST training data", [&]()
	{
		return mnist_load_combined(training_images, training_labels);
	});

	const training_data mnist_test = time_execution_func<training_data>("Load MNIST test data", [&]()
	{
		return mnist_load_combined(test_images, test_labels);
	});

	sgd_trainer trainer;
	auto wnp = std::make_shared<L2_regularization>(
		static_cast<fpt>(3.0 / mnist_training.entry_count()));

	//trainer.add_new_layer<fully_connected_layer>(784, 500);
	//trainer.add_new_layer<relu_activation_layer>(500);

	//trainer.add_new_layer<fully_connected_layer>(500, 300);
	//trainer.add_new_layer<relu_activation_layer>(300);

	//trainer.add_new_layer<fully_connected_layer>(300, 200);
	//trainer.add_new_layer<relu_activation_layer>(200);

	//trainer.add_new_layer<fully_connected_layer>(200, 50);
	//trainer.add_new_layer<relu_activation_layer>(50);

	trainer.add_new_layer<fully_connected_layer>(784, 30, wnp);
	trainer.add_new_layer<logistic_activation_layer>(30);

	trainer.add_new_layer<fully_connected_layer>(30, 10, wnp);
	trainer.add_new_layer<softmax_act_cross_entropy_costs>(10);

	trainer.init();

	//auto opt = momentum_sgd(.5f, .9f);
	auto opt = adam();

	const std::function<void(training_status)> logf([&](training_status stat)
	{
		const unsigned batch_no = stat.batch_no;
		const unsigned batch_count = stat.batch_count;

		if (batch_no < batch_count)
		{
			std::cout << "\r" << batch_no << "/" << batch_count
				<< " [" << unsigned(100.0 * (batch_no + 1) / batch_count) << "%]";
		}
		else
		{
			std::cout << "\r" << batch_count << "/" << batch_count << " [100%]" << std::endl;
		}
	});

	time_execution("Train " + std::to_string(epoch_count) + " epochs", [&]()
	{
		trainer.train_epochs(mnist_training, &opt, 8, epoch_count, &logf, 100);
	});

	const auto accuracy = time_execution_func<double>("Test", [&]()
	{
		return test_network(trainer, mnist_test).accuracy;
	});

	std::cout << "Test accuracy: " << accuracy << "%" << std::endl;

	std::cout << "Press any button to continue...";
	std::cin.get();
}
