#include <iostream>
#include <chrono>
#include "mnist.h"
#include "mat_arr.h"
#include "mat_arr_math.h"
#include "sgd_trainer.h"
#include <thread>

typedef std::chrono::high_resolution_clock Clock;

using namespace annlib;
using namespace std::chrono;
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
T time_execution_func(const std::string& name, const std::function<T()>& func)
{
	std::cout << name << " started" << std::endl;
	const auto calcStart = Clock::now();
	T result = func();
	const double gpu_calc_millis = duration_cast<nanoseconds>(Clock::now() - calcStart).count() / double(1e6);
	std::cout << name << " finished, time elapsed: " << gpu_calc_millis << " ms" << std::endl;
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

struct add3
{
	double operator()(double a, double b) const { return a + b; }
};

double test_add(double a, double b)
{
	return a + b;
}

template <typename F>
void test_add_3(const mat_arr& mat_a, const mat_arr& mat_b, mat_arr* mat_c, F f)
{
	const unsigned n3 = mat_a.size();
	for (unsigned iterations = 0; iterations < 10; iterations++)
	{
		const double* a = mat_a.start();
		const double* b = mat_b.start();
		double* c = mat_c->start();

		for (unsigned i = 0; i < n3; i++)
		{
			c[i] = f(a[i], b[i]);
		}
	}
}

void mat_arr_math_add_speed_test2()
{
	const unsigned n = 200;
	mat_arr mat_a(n, n, n);
	mat_arr mat_b(n, n, n);
	mat_arr mat_c(n, n, n);

	mat_set_all(1, &mat_a);
	mat_set_all(2, &mat_b);

	test_add_3(mat_a, mat_b, &mat_c, add3());
}

void mat_arr_math_add_speed_test()
{
	mat_arr_math_add_speed_test2();
	const unsigned n = 200;
	const unsigned n3 = n * n * n;
	std::vector<double> vec_a(n3);
	std::vector<double> vec_b(n3);
	std::vector<double> vec_c(n3);

	mat_arr mat_a(n, n, n);
	mat_arr mat_b(n, n, n);
	mat_arr mat_c(n, n, n);

	time_execution("mat add", [&]
	{
		for (unsigned iterations = 0; iterations < 10; iterations++)
		{
			mat_element_wise_add(mat_a, mat_b, &mat_c);
		}
	});

	time_execution("mat add 2", [&]
	{
		for (unsigned iterations = 0; iterations < 10; iterations++)
		{
			double* a = mat_a.start();
			double* b = mat_b.start();
			double* c = mat_c.start();

			for (unsigned i = 0; i < n3; i++)
			{
				*(c + i) = *(a + i) + *(b + i);
			}
		}
	});

	time_execution("mat add 3", [&]
	{
		const std::function<double(double, double)> add = [](double a, double b) { return a + b; };
		for (unsigned iterations = 0; iterations < 10; iterations++)
		{
			double* a = mat_a.start();
			double* b = mat_b.start();
			double* c = mat_c.start();

			for (unsigned i = 0; i < n3; i++)
			{
				*(c + i) = add(*(a + i), *(b + i));
			}
		}
	});

	time_execution("mat add 4", [&]
	{
		test_add_3(mat_a, mat_b, & mat_c, test_add);
	});

	time_execution("mat add 5", [&]
	{
		test_add_3(mat_a, mat_b, &mat_c, add3());
	});

	time_execution("vector add", [&]
	{
		for (unsigned iterations = 0; iterations < 10; iterations++)
		{
			for (unsigned i = 0; i < n3; i++)
			{
				vec_c[i] = vec_a[i] + vec_b[i];
			}
		}
	});
}

void mat_arr_math_mat_mul_speed_test()
{
	const unsigned n = 100;
	const unsigned n2 = n * n;
	const unsigned n3 = n * n * n;
	std::vector<double> vec_a(n3);
	std::vector<double> vec_b(n3);
	std::vector<double> vec_c(n3);

	mat_arr mat_a(n, n, n);
	mat_arr mat_b(n, n, n);
	mat_arr mat_c(n, n, n);

	time_execution("mat mul", [&]
	{
		for (unsigned iterations = 0; iterations < 10; iterations++)
		{
			mat_matrix_mul(mat_a, mat_b, &mat_c);
		}
	});

	time_execution("mat mul 2", [&]
	{
		for (unsigned iterations = 0; iterations < 10; iterations++)
		{
			for (unsigned matNo = 0; matNo < n; matNo++)
			{
				const double* a = mat_a.start() + (matNo * n2);
				const double* b = mat_b.start() + (matNo * n2);
				double* c = mat_c.start() + (matNo * n2);

				for (unsigned i = 0; i < n; i++)
				{
					for (unsigned j = 0; j < n; j++)
					{
						for (unsigned k = 0; k < n; k++)
						{
							*(c + (i * n + k)) += *(a + (i * n + j)) * *(b + (j * n + k));
						}
					}
				}
			}
		}
	});

	time_execution("vector mul", [&]
	{
		for (unsigned iterations = 0; iterations < 10; iterations++)
		{
			for (unsigned mat_no = 0; mat_no < n; mat_no++)
			{
				for (unsigned i = 0; i < n; i++)
				{
					for (unsigned j = 0; j < n; j++)
					{
						for (unsigned k = 0; k < n; k++)
						{
							vec_c[mat_no * n2 + i * n + k]
								= vec_a[mat_no * n2 + i * n + j]
								* vec_b[mat_no * n2 + j * n + k];
						}
					}
				}
			}
		}
	});
}

int main()
{
	mat_arr_math_add_speed_test();

	const unsigned n_threads = std::thread::hardware_concurrency();
	std::cout << n_threads << " concurrent threads are supported.\n";

#ifdef __linux__
	const std::string folder = "/mnt/c/";
#else
	const std::string folder = "C:\\\\";
#endif

	const std::string training_images = folder + "train-images.idx3-ubyte";
	const std::string training_labels = folder + "train-labels.idx1-ubyte";
	const std::string test_images = folder + "t10k-images.idx3-ubyte";
	const std::string test_labels = folder + "t10k-labels.idx1-ubyte";

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
	trainer.activation_f = std::make_shared<logistic_activation_function>(1.0);
	trainer.cost_f = std::make_shared<cross_entropy_costs>();
	trainer.weight_norm_penalty = std::make_shared<L2_regularization>(3.0 / mnist_training.entry_count());
	trainer.optimizer = std::make_shared<momentum_sgd>(5.0, 0.0);
	trainer.net_init = std::make_shared<normalized_gaussian_net_init>();

	std::vector<unsigned> sizes{{784, 30, 10}};
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

	std::cout << "Test accuracy: " << accuracy << "%" << std::endl;
	std::cin.get();
}
