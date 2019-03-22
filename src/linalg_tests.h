#ifndef LINALG_TESTS_H
#define LINALG_TESTS_H

#include "mat_arr_math.h"
#include "mat_arr_math_t.h"
#include <functional>

#ifdef LINALG_CUDA_SUPPORT
#include "cuda/cuda_util.cuh"
#include "cuda/linalg_cudaops.cuh"
#endif

template <typename T>
T time_execution_func(const std::string& name, const std::function<T()>& func);

void time_execution(const std::string& name, const std::function<void()>& func);

void random_matrix_arr(mat_arr* m)
{
	const unsigned size = m->size();

	fpt* mat = m->start();
	for (unsigned i = 0; i < size; i++)
	{
		*(mat + i) = static_cast<fpt>(rand()) / static_cast<fpt>(RAND_MAX);
	}
}

inline void do_assert(bool b)
{
	if (!b)
	{
		throw std::runtime_error("Assertion failed");
	}
}

inline void testTranspose()
{
	mat_arr A(1, 2, 3);
	fpt* a = A.start();
	*(a + 0) = 0;
	*(a + 1) = 1;
	*(a + 2) = 2;
	*(a + 3) = 3;
	*(a + 4) = 4;
	*(a + 5) = 5;

	mat_arr C(1, 3, 2);
	mat_transpose(A, &C);
	fpt* c = C.start();

	do_assert(*(c + 0) == 0);
	do_assert(*(c + 1) == 3);
	do_assert(*(c + 2) == 1);
	do_assert(*(c + 3) == 4);
	do_assert(*(c + 4) == 2);
	do_assert(*(c + 5) == 5);
}

inline void __testAddTranspose_check(const fpt* c)
{
	do_assert(*(c + 0) == 12);
	do_assert(*(c + 1) == 15);
	do_assert(*(c + 2) == 18);
	do_assert(*(c + 3) == 21);
	do_assert(*(c + 4) == 24);
	do_assert(*(c + 5) == 27);
}

inline void testAddTranspose()
{
	mat_arr A(1, 2, 3);
	fpt* a = A.start();
	*(a + 0) = 0;
	*(a + 1) = 1;
	*(a + 2) = 2;
	*(a + 3) = 3;
	*(a + 4) = 4;
	*(a + 5) = 5;

	mat_arr B(1, 2, 3);
	fpt* b = B.start();
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

inline void testMatMul_CheckC(const fpt* c)
{
	do_assert(*(c + 0) == 56);
	do_assert(*(c + 1) == 62);
	do_assert(*(c + 2) == 200);
	do_assert(*(c + 3) == 224);
}

inline void testMatMul()
{
	mat_arr A(1, 2, 3);
	fpt* a = A.start();
	*(a + 0) = 0;
	*(a + 1) = 1;
	*(a + 2) = 2;
	*(a + 3) = 3;
	*(a + 4) = 4;
	*(a + 5) = 5;

	mat_arr B(1, 3, 2);
	fpt* b = B.start();
	*(b + 0) = 12;
	*(b + 1) = 14;
	*(b + 2) = 16;
	*(b + 3) = 18;
	*(b + 4) = 20;
	*(b + 5) = 22;

	mat_arr C(1, 2, 2);
	mat_matrix_mul(A, B, &C);
	fpt* c = C.start();
	testMatMul_CheckC(c);
}

inline void testMatMulTransposed()
{
	mat_arr A(1, 2, 3);
	fpt* a = A.start();
	*(a + 0) = 0;
	*(a + 1) = 1;
	*(a + 2) = 2;
	*(a + 3) = 3;
	*(a + 4) = 4;
	*(a + 5) = 5;

	mat_arr A_t(1, 3, 2);
	mat_transpose(A, &A_t);

	mat_arr B(1, 3, 2);
	fpt* b = B.start();
	*(b + 0) = 12;
	*(b + 1) = 14;
	*(b + 2) = 16;
	*(b + 3) = 18;
	*(b + 4) = 20;
	*(b + 5) = 22;

	mat_arr B_t(1, 2, 3);
	mat_transpose(B, &B_t);

	mat_arr C(1, 2, 2);
	fpt* c = C.start();


	mat_matrix_mul(A, B, &C, transpose_no);
	testMatMul_CheckC(c);

	mat_matrix_mul(A_t, B, &C, transpose_A);
	testMatMul_CheckC(c);

	mat_matrix_mul(A, B_t, &C, transpose_B);
	testMatMul_CheckC(c);

	mat_matrix_mul(A_t, B_t, &C, transpose_both);
	testMatMul_CheckC(c);
}

inline void speedAddTranspose()
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

struct add3
{
	fpt operator()(fpt a, fpt b) const { return a + b; }
};

inline fpt test_add(fpt a, fpt b)
{
	return a + b;
}

template <typename F>
void test_add_3(const mat_arr& mat_a, const mat_arr& mat_b, mat_arr* mat_c, F f)
{
	const unsigned n3 = mat_a.size();
	for (unsigned iterations = 0; iterations < 10; iterations++)
	{
		const fpt* a = mat_a.start();
		const fpt* b = mat_b.start();
		fpt* c = mat_c->start();

		for (unsigned i = 0; i < n3; i++)
		{
			c[i] = f(a[i], b[i]);
		}
	}
}

inline void mat_arr_math_add_speed_test2()
{
	const unsigned n = 200;
	mat_arr mat_a(n, n, n);
	mat_arr mat_b(n, n, n);
	mat_arr mat_c(n, n, n);

	mat_set_all(1, &mat_a);
	mat_set_all(2, &mat_b);

	test_add_3(mat_a, mat_b, &mat_c, add3());
}

inline void mat_arr_math_add_speed_test()
{
	std::cout << std::endl;
	mat_arr_math_add_speed_test2();
	const unsigned n = 200;
	const unsigned n3 = n * n * n;
	std::vector<fpt> vec_a(n3);
	std::vector<fpt> vec_b(n3);
	std::vector<fpt> vec_c(n3);

	mat_arr mat_a(n, n, n);
	mat_arr mat_b(n, n, n);
	mat_arr mat_c(n, n, n);

	time_execution("mat add                                ", [&]
	{
		for (unsigned iterations = 0; iterations < 10; iterations++)
		{
			mat_element_wise_add(mat_a, mat_b, &mat_c);
		}
	});

	time_execution("mat e by e operation with std::function", [&]
	{
		const std::function<fpt(fpt, fpt)> func = [](fpt a, fpt b) { return a + b; };
		for (unsigned iterations = 0; iterations < 10; iterations++)
		{
			mat_element_by_element_operation(mat_a, mat_b, &mat_c, func);
		}
	});

	time_execution("mat add loop                           ", [&]
	{
		for (unsigned iterations = 0; iterations < 10; iterations++)
		{
			fpt* a = mat_a.start();
			fpt* b = mat_b.start();
			fpt* c = mat_c.start();

			for (unsigned i = 0; i < n3; i++)
			{
				*(c + i) = *(a + i) + *(b + i);
			}
		}
	});

	time_execution("mat add loop with std::function call   ", [&]
	{
		const std::function<fpt(fpt, fpt)> add = [](fpt a, fpt b) { return a + b; };
		for (unsigned iterations = 0; iterations < 10; iterations++)
		{
			fpt* a = mat_a.start();
			fpt* b = mat_b.start();
			fpt* c = mat_c.start();

			for (unsigned i = 0; i < n3; i++)
			{
				*(c + i) = add(*(a + i), *(b + i));
			}
		}
	});

	time_execution("mat add template function with function", [&]
	{
		test_add_3(mat_a, mat_b, &mat_c, test_add);
	});

	time_execution("mat add template function with struct  ", [&]
	{
		test_add_3(mat_a, mat_b, &mat_c, add3());
	});

	time_execution("vector add                             ", [&]
	{
		for (unsigned iterations = 0; iterations < 10; iterations++)
		{
			for (unsigned i = 0; i < n3; i++)
			{
				vec_c[i] = vec_a[i] + vec_b[i];
			}
		}
	});
	std::cout << std::endl;
}

inline void mat_arr_math_mat_mul_speed_test()
{
	std::cout << std::endl;
	const unsigned n = 100;
	const unsigned n2 = n * n;
	const unsigned n3 = n * n * n;
	std::vector<fpt> vec_a(n3);
	std::vector<fpt> vec_b(n3);
	std::vector<fpt> vec_c(n3);

	mat_arr mat_a(n, n, n);
	mat_arr mat_b(n, n, n);
	mat_arr mat_c(n, n, n);

	time_execution("mat mul      ", [&]
	{
		for (unsigned iterations = 0; iterations < 10; iterations++)
		{
			mat_matrix_mul(mat_a, mat_b, &mat_c);
		}
	});

	time_execution("mat mul loops", [&]
	{
		for (unsigned iterations = 0; iterations < 10; iterations++)
		{
			for (unsigned matNo = 0; matNo < n; matNo++)
			{
				const fpt* a = mat_a.start() + (matNo * n2);
				const fpt* b = mat_b.start() + (matNo * n2);
				fpt* c = mat_c.start() + (matNo * n2);

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

	time_execution("vector mul   ", [&]
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
							vec_c[mat_no * n2 + i * n + k] = vec_a[mat_no * n2 + i * n + j] * vec_b[mat_no * n2 + j * n
								+ k];
						}
					}
				}
			}
		}
	});
	std::cout << std::endl;
}

struct mat_arr_math_scalar_mul_test_kernel
{
	fpt operator()(fpt a, fpt b) const
	{
		return a * b;
	}
};

template <typename Fc>
void mat_arr_math_scalar_mul_test_template(const mat_arr& mat_a, fpt b, mat_arr* mat_c, Fc f)
{
	const unsigned n3 = mat_a.size();
	const fpt* a = mat_a.start();
	fpt* c = mat_c->start();
	for (unsigned i = 0; i < n3; i++)
	{
		c[i] = f(a[i], b);
	}
}

inline void mat_arr_math_scalar_mul_speed_test()
{
	std::cout << std::endl;
	const unsigned n_it = 1000;
	const unsigned n = 100;
	const unsigned n3 = n * n * n;
	std::vector<fpt> vec_a(n3);
	std::vector<fpt> vec_c(n3);

	mat_arr mat_a(n, n, n);
	mat_arr mat_c(n, n, n);

	time_execution("scalar mul                          ", [&]
	{
		for (unsigned iterations = 0; iterations < n_it; iterations++)
		{
			mat_element_wise_mul(mat_a, 2.5f, &mat_c);
		}
	});

	time_execution("scalar mul loop                     ", [&]
	{
		for (unsigned iterations = 0; iterations < n_it; iterations++)
		{
			fpt* a = mat_a.start();
			fpt* c = mat_c.start();

			for (unsigned i = 0; i < n3; i++)
			{
				c[i] = a[i] * 2.5f;
			}
		}
	});

	time_execution("scalar mul template func with 2 args", [&]
	{
		for (unsigned iterations = 0; iterations < n_it; iterations++)
		{
			fpt* a = mat_a.start();
			fpt* c = mat_c.start();

			for (unsigned i = 0; i < n3; i++)
			{
				c[i] = a[i] * 2.5f;
			}
		}
	});
	std::cout << std::endl;
}

struct addmul
{
	fpt factor;

	explicit addmul(fpt factor) : factor(factor)
	{
	}

	fpt operator()(fpt a, fpt b) const { return a + factor * b; }
};

struct addmul_fixed4
{
	fpt operator()(fpt a, fpt b) const { return a + 4.0f * b; }
};

inline void mat_arr_math_addmul_speed_test()
{
	std::cout << std::endl;
	mat_arr_math_add_speed_test2();
	const unsigned n = 200;

	mat_arr mat_a(n, n, n);
	mat_arr mat_b(n, n, n);
	mat_arr mat_c(n, n, n);

	random_matrix_arr(&mat_a);
	random_matrix_arr(&mat_b);
	random_matrix_arr(&mat_c);

	fpt factor = 4;

	time_execution("mat add mul with std::function", [&]
	{
		for (unsigned iterations = 0; iterations < 100; iterations++)
		{
			mat_element_by_element_operation(mat_a, mat_b, &mat_c,
			                                 [&](fpt a, fpt b)
			                                 {
				                                 return a + factor * b;
			                                 });
		}
	});

	time_execution("mat add mul with struct       ", [&]
	{
		for (unsigned iterations = 0; iterations < 100; iterations++)
		{
			mat_element_by_element_operation(mat_a, mat_b, &mat_c, addmul(factor));
		}
	});

	time_execution("mat add mul with fixed struct ", [&]
	{
		for (unsigned iterations = 0; iterations < 100; iterations++)
		{
			mat_element_by_element_operation(mat_a, mat_b, &mat_c, addmul_fixed4());
		}
	});
	std::cout << std::endl;
}

inline void mat_arr_math_multipleadd_speed_test()
{
	std::cout << std::endl;
	mat_arr_math_add_speed_test2();
	const unsigned n = 200;

	mat_arr mat_a(n, n, n);
	mat_arr mat_b(n, n, n);
	mat_arr mat_c(n, n, n);

	mat_arr mat_d(n, n, n);
	mat_arr mat_temp(n, n, n);

	random_matrix_arr(&mat_a);
	random_matrix_arr(&mat_b);
	random_matrix_arr(&mat_c);

	time_execution("add twice     ", [&]
	{
		for (unsigned iterations = 0; iterations < 100; iterations++)
		{
			mat_element_wise_add(mat_a, mat_b, &mat_temp);
			mat_element_wise_add(mat_temp, mat_c, &mat_d);
		}
	});

	/*time_execution("add three     ", [&] {
		for (unsigned iterations = 0; iterations < 100; iterations++)
		{
			std::vector<mat_arr*> in = {{&mat_a, &mat_b, &mat_c}};
			mat_multiple_e_by_e_operation(in, &mat_d, [&](const std::vector<fpt>& vec) {
				return vec[0] + vec[1] + vec[2];
			});
		}
	});*/
	std::cout << std::endl;
}

inline void feedfoward_speed_test()
{
	std::cout << std::endl;
	const unsigned n_in = 784;
	const unsigned n_out = 100;
	const unsigned n_count = 64;

	mat_arr in(n_count, 1, n_in);
	mat_arr weights(1, n_in, n_out);
	mat_arr biases(1, 1, n_out);
	mat_arr out(n_count, 1, n_out);

	time_execution("matarr lib", [&]
	{
		for (unsigned iterations = 0; iterations < 1000; iterations++)
		{
			mat_matrix_mul(in, weights, &out);
			mat_element_wise_add(out, biases, &out);
		}
	});
	std::cout << std::endl;
}

void test_e_by_e_with_scalar()
{
	std::cout << std::endl;

	mat_arr A(1, 2, 3);
	fpt* a = A.start();
	*(a + 0) = 0;
	*(a + 1) = 1;
	*(a + 2) = 2;
	*(a + 3) = 3;
	*(a + 4) = 4;
	*(a + 5) = 5;

	mat_arr B(1, 1, 1);
	fpt* b = B.start();
	b[0] = 3;

	mat_arr C(1, 2, 3);

	mat_element_wise_add(A, B, &C);

	std::cout << std::endl;
}

#ifdef  LINALG_CUDA_SUPPORT
void cuda_matrix_mul_test()
{
	std::cout << std::endl;
	std::cout << "CUDA matrix multiplication" << std::endl;
	const unsigned n = 200;
	const unsigned iterations = 1;

	mat_arr mat_a(100, n, n);
	mat_arr mat_b(100, n, n);
	mat_arr mat_c(100, n, n);

	random_matrix_arr(&mat_a);
	random_matrix_arr(&mat_b);

	mat_a.dev_start();
	mat_b.dev_start();
	mat_c.dev_start();

	time_execution("CUDA matrix mul", [&]
	{
		for (unsigned it = 0; it < iterations; it++)
		{
			linalg::cuda::cuda_matrix_mul(mat_a, mat_b, &mat_c, transpose_no);
		}
		cuda::cuda_sync();
	});

	cuda::cuda_sync();

	mat_a.start();
	mat_b.start();
	mat_c.start();

	cuda::cuda_sync();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		throw std::runtime_error("");
	}

	time_execution("CPU  matrix mul", [&]
	{
		for (unsigned it = 0; it < iterations; it++)
		{
			mat_matrix_mul(mat_a, mat_b, &mat_c, transpose_no);
		}
	});

	std::cout << std::endl;
}

void cuda_element_wise_operation_test()
{
	const unsigned n = 20;
	mat_arr mat_a(1024, n, n);
	mat_arr mat_b(1024, n, n);
	mat_arr mat_c(1024, n, n);

	random_matrix_arr(&mat_a);
	random_matrix_arr(&mat_b);

	time_execution("CUDA e by e mul", [&]
	{
		linalg::cuda::cuda_element_wise_mul(mat_a, mat_b, &mat_c);
		cuda::cuda_sync();
	});

	mat_c.start();

	time_execution("CUDA element wise mul", [&]
	{
		linalg::cuda::cuda_element_wise_mul(mat_a, 2.0f, &mat_c);
		cuda::cuda_sync();
	});
}
#endif
#endif // LINALG_TESTS_H
