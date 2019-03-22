#include "mnist.h"
#include "output_layer.h"
#include "convolution_layer.h"
#include <chrono>
#include <fstream>
#include <thread>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef std::chrono::high_resolution_clock Clock;

using namespace annlib;
using namespace annlib::tasks;
using namespace std::chrono;
using namespace linalg;

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
	std::cout << n_threads << " concurrent threads are supported.\n" << std::endl;

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

typedef annlib::tasks::classification_test_result test_result;

struct cycle_result
{
	annlib::tasks::classification_test_result training_result;
	annlib::tasks::classification_test_result test_result;
};

cycle_result train_and_test(unambiguous_classification* cl,
                            double epoch_count,
                            const classification_data& train_data,
                            const classification_data& test_data)
{
	cl->train(epoch_count);

	const test_result train_result = cl->test(train_data);
	const test_result test_result = cl->test(test_data);

	return cycle_result{
		train_result,
		test_result
	};
}

void log_cycle_result(unsigned i, cycle_result res, std::ofstream* fs)
{
	std::cout << std::endl;
	std::cout << "Cycle Number:           " << i << std::endl;
	std::cout << "Training costs:         " << res.training_result.average_costs << std::endl;
	std::cout << "Training accuracy:      " << res.training_result.accuracy << std::endl;
	std::cout << "Test costs:             " << res.test_result.average_costs << std::endl;
	std::cout << "Test accuracy:          " << res.test_result.accuracy << std::endl;
	std::cout << std::fixed << std::setprecision(0);
	std::cout << "Test Confusion Matrix:  " << std::endl << res.test_result.total_confusion_matrix;
	std::cout << std::defaultfloat << std::setprecision(4);
	std::cout << std::endl;

	*fs << i << ','
	    << res.training_result.average_costs << ','
	    << res.training_result.accuracy << ','
	    << res.test_result.average_costs << ','
	    << res.test_result.accuracy << ','
	    << std::endl;
}

void train_and_test(double epochs_per_cycle, unsigned cycles,
                    unambiguous_classification* cl,
                    const classification_data& train_data,
                    const classification_data& test_data)
{
	const std::string path = "log.csv";
	std::ofstream fs(path);

	fs << "cycle_number" << ','
	   << "training_costs" << ','
	   << "training_accuracy" << ','
	   << "test_costs" << ','
	   << "test_accuracy" << ','
	   << std::endl;

	for (unsigned c = 0; c < cycles; c++)
	{
		time_execution("train cycle " + std::to_string(c), [&]()
		{
			const cycle_result cres = train_and_test(cl, epochs_per_cycle, train_data, test_data);
			log_cycle_result(c, cres, &fs);
		});
	}
}

int main(int argc, char** argv)
{
	try
	{
		const unsigned cycle_count = 100;
		const double epochs_per_cycle = (argc <= 1 ? 100.0 : std::stod(std::string(argv[1]))) / cycle_count;
		std::cout << "Cycle count: " << cycle_count << std::endl;
		std::cout << "Epochs per cycle: " << epochs_per_cycle << std::endl;

#ifdef __linux__
		const std::string folder = "/mnt/c/";
#else
		const std::string folder = "C:\\\\";
#endif

		const std::string training_images = folder + "train-images.idx3-ubyte";
		const std::string training_labels = folder + "train-labels.idx1-ubyte";
		const std::string test_images = folder + "t10k-images.idx3-ubyte";
		const std::string test_labels = folder + "t10k-labels.idx1-ubyte";

		const auto mnist_training = time_execution_func<classification_data>("Load MNIST training data", [&]()
		{
			return mnist_load_combined(training_images, training_labels);
		});

		const auto mnist_test = time_execution_func<classification_data>("Load MNIST test data", [&]()
		{
			return mnist_load_combined(test_images, test_labels);
		});

		sgd_trainer trainer;
		auto wnp = std::make_shared<L2_regularization>(static_cast<fpt>(3.0 / mnist_training.input.count));

		conv_layer_hyperparameters conv_p
			{
				20,     // map count
				28, 28, // image dimension
				5, 5,   // mask dimension
				1, 1    // stride
			};

		pooling_layer_hyperparameters pool_p
			{
				20,     // map count
				24, 24, // map dimension
				3, 3    // mask dimension
			};

		trainer.add_new_layer<convolution_layer>(conv_p);
		trainer.add_new_layer<max_pooling_layer>(pool_p);

		const unsigned size_after_pooling = pool_p.out_size_total();
		std::cout << "Size after pooling: " << size_after_pooling << std::endl;

		trainer.add_new_layer<relu_activation_layer>(size_after_pooling);

		trainer.add_new_layer<fully_connected_layer>(size_after_pooling, 600, wnp);
		trainer.add_new_layer<relu_activation_layer>(600);

		trainer.add_new_layer<fully_connected_layer>(600, 200, wnp);
		trainer.add_new_layer<relu_activation_layer>(200);

		trainer.add_new_layer<fully_connected_layer>(200, 10, wnp);
		trainer.add_new_layer<softmax_act_cross_entropy_costs>(10);

		trainer.init();

		unambiguous_classification cl;
		cl.set_optimizer<adam>(1e-4f, 0.9f, 0.99f);
		cl.set_trainer<sgd_trainer>(trainer);
		cl.set_data(mnist_training, 8);

		train_and_test(epochs_per_cycle, cycle_count, &cl, mnist_training, mnist_test);

		std::cout << "Press any button to continue...";
		std::cin.get();

		return 0;
	}
	catch (std::exception& ex)
	{
		std::cout << "error:" << ex.what() << std::endl;
		std::cin.get();
		return 1;
	}
}
