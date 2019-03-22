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
		const cycle_result cres = train_and_test(cl, epochs_per_cycle, train_data, test_data);
		log_cycle_result(c, cres, &fs);
	}
}

void random_matrix_arr(mat_arr* m)
{
	const unsigned size = m->size();

	fpt* mat = m->start();
	for (unsigned i = 0; i < size; i++)
	{
		*(mat + i) = static_cast<fpt>(rand()) / static_cast<fpt>(RAND_MAX);
	}
}

void test_convolve()
{
	annlib::convolve(1, 1, 1, 4, 4, 2, 2, 2, 2, std::function<void(unsigned, unsigned, unsigned)>
		([&](unsigned i_in, unsigned i_out, unsigned i_mask)
		 {
			 std::cout << i_in << " | " << i_mask << " | " << i_out << std::endl;
		 }));
}

void test_max_pool_single()
{
	mat_arr in(3, 1, 8);
	mat_arr out(3, 1, 8);

	pooling_layer_hyperparameters pool_test_p
		{
			2,    // map count
			2, 2, // map dimension
			1, 1  // mask dimension
		};

	layer_buffer lbuf(3, in, out, mat_arr(3, 1, 8));

	max_pooling_layer la(pool_test_p);
	la.prepare_buffer(&lbuf);

	std::cout << std::endl;
	std::cout << "feed forward" << std::endl;
	std::cout << std::endl;
	for (unsigned i = 0; i < 4; i++)
	{
		random_matrix_arr(&in);
		la.feed_forward_detailed(in, &out, &lbuf);
		std::cout << in << out << std::endl;
	}

	std::cout << "backprop" << std::endl;
	std::cout << std::endl;

	for (unsigned i = 0; i < 4; i++)
	{
		random_matrix_arr(&in);
		la.backprop(in, &out, &lbuf);
		std::cout << in << out << std::endl;
	}
}

void test_max_pool()
{
	pooling_layer_hyperparameters pool_test_p
		{
			2,    // map count
			4, 4, // map dimension
			2, 2  // mask dimension
		};

	mat_arr in(3, 1, pool_test_p.in_size_total());
	mat_arr out(3, 1, pool_test_p.out_size_total());

	mat_arr err_in(3, 1, pool_test_p.out_size_total());
	mat_arr err_out(3, 1, pool_test_p.in_size_total());

	layer_buffer lbuf(3, in, out, mat_arr(3, 1, pool_test_p.out_size_total()));

	max_pooling_layer la(pool_test_p);
	la.prepare_buffer(&lbuf);

	for (unsigned i = 0; i < 4; i++)
	{
		random_matrix_arr(&in);
		la.feed_forward_detailed(in, &out, &lbuf);
		std::cout << in << out << lbuf.get_val("df") << std::endl;

		random_matrix_arr(&err_in);
		la.backprop(err_in, &err_out, &lbuf);
		std::cout << err_in << err_out << lbuf.get_val("df") << std::endl << std::endl;
	}
}

void test_conv_layer1()
{
	conv_layer_hyperparameters p
		{
			2,     // map count
			2, 2, // image dimension
			1, 1,   // mask dimension
			1, 1    // stride
		};

	mat_arr in(1, 1, p.input_size());
	mat_arr out(1, 1, p.output_size());

	fpt* in_ptr = in.start();
	in_ptr[0] = 1;
	in_ptr[1] = 2;
	in_ptr[2] = 3;
	in_ptr[3] = 4;

	convolution_layer la(p);
	fpt* mb_ptr = la.mask_biases.start();
	mb_ptr[0] = 1;
	mb_ptr[1] = 2;

	fpt* mw_ptr = la.mask_weights.start();
	mw_ptr[0] = 2;
	mw_ptr[1] = 10;

	for (unsigned i = 0; i < 4; i++)
	{
		la.feed_forward(in, &out);
		std::cout << out << std::endl;
	}
}

int main(int argc, char** argv)
{
	try
	{
		//test_convolve();
		//test_max_pool_single();
		//test_max_pool();
		test_conv_layer1();
		const unsigned cycle_count = 100;
		const double epochs_per_cycle = (argc <= 1 ? 1.0 : std::stod(std::string(argv[1]))) / cycle_count;

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

		//trainer.add_new_layer<max_pooling_layer>(pool_test_p);
		//trainer.add_new_layer<max_pooling_layer>(pool_test_p);

		trainer.add_new_layer<convolution_layer>(conv_p);
		trainer.add_new_layer<max_pooling_layer>(pool_p);

		//trainer.add_new_layer<fully_connected_layer>(784, 500);
		//trainer.add_new_layer<relu_activation_layer>(500);

		//trainer.add_new_layer<fully_connected_layer>(500, 300);
		//trainer.add_new_layer<relu_activation_layer>(300);

		//trainer.add_new_layer<fully_connected_layer>(300, 200);
		//trainer.add_new_layer<relu_activation_layer>(200);

		//trainer.add_new_layer<fully_connected_layer>(200, 50);
		//trainer.add_new_layer<relu_activation_layer>(50);

		const unsigned size_after_pooling = pool_p.out_size_total();
		std::cout << "Size after pooling: " << size_after_pooling << std::endl;

		trainer.add_new_layer<logistic_activation_layer>(size_after_pooling);

		trainer.add_new_layer<fully_connected_layer>(size_after_pooling, 100, wnp);
		trainer.add_new_layer<logistic_activation_layer>(100);

		trainer.add_new_layer<fully_connected_layer>(100, 10, wnp);
		trainer.add_new_layer<softmax_act_cross_entropy_costs>(10);

		trainer.init();

		unambiguous_classification cl;
		cl.set_optimizer<adam>();
		cl.set_trainer<sgd_trainer>(trainer);
		cl.set_data(mnist_training, 8);

		for (unsigned i = 0; i < cycle_count; i++)
		{
			time_execution("train cycle " + std::to_string(i), [&]()
			{
				train_and_test(epochs_per_cycle, 1, &cl, mnist_training, mnist_test);
				std::cout << static_cast<convolution_layer*>(trainer.get_layer(0))->mask_weights << std::endl;
				std::cout << static_cast<convolution_layer*>(trainer.get_layer(0))->mask_biases << std::endl;
			});
		}

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
