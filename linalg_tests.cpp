#include "mat_arr_math.h"

void do_assert(bool b)
{
	if (!b) {
		throw runtime_error("Assertion failed");
	}
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

	do_assert(*(c + 0) == 0);
	do_assert(*(c + 1) == 3);
	do_assert(*(c + 2) == 1);
	do_assert(*(c + 3) == 4);
	do_assert(*(c + 4) == 2);
	do_assert(*(c + 5) == 5);
}

void __testAddTranspose_check(double* c)
{
	do_assert(*(c + 0) == 12);
	do_assert(*(c + 1) == 15);
	do_assert(*(c + 2) == 18);
	do_assert(*(c + 3) == 21);
	do_assert(*(c + 4) == 24);
	do_assert(*(c + 5) == 27);
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

void testMatMul_CheckC(double * c)
{
	do_assert(*(c + 0) == 56);
	do_assert(*(c + 1) == 62);
	do_assert(*(c + 2) == 200);
	do_assert(*(c + 3) == 224);
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
	testMatMul_CheckC(c);
}

void testMatMulTransposed()
{
	mat_arr A(1, 2, 3);
	double* a = A.start();
	*(a + 0) = 0;
	*(a + 1) = 1;
	*(a + 2) = 2;
	*(a + 3) = 3;
	*(a + 4) = 4;
	*(a + 5) = 5;

	mat_arr A_t(1, 3, 2);
	mat_transpose(A, &A_t);

	mat_arr B(1, 3, 2);
	double* b = B.start();
	*(b + 0) = 12;
	*(b + 1) = 14;
	*(b + 2) = 16;
	*(b + 3) = 18;
	*(b + 4) = 20;
	*(b + 5) = 22;

	mat_arr B_t(1, 2, 3);
	mat_transpose(B, &B_t);

	mat_arr C(1, 2, 2);
	double* c = C.start();


	mat_matrix_mul(A, B, &C, transpose_no);
	testMatMul_CheckC(c);

	mat_matrix_mul(A_t, B, &C, transpose_A);
	testMatMul_CheckC(c);

	mat_matrix_mul(A, B_t, &C, transpose_B);
	testMatMul_CheckC(c);

	mat_matrix_mul(A_t, B_t, &C, transpose_both);
	testMatMul_CheckC(c);
}