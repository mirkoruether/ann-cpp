#include "mat_arr_math.h"
#include "mat_arr_math_t.h"

using namespace linalg;

namespace linalg
{
	struct _mat_e_by_e_add_kernel
	{
		float operator()(float a, float b) const
		{
			return a + b;
		}
	};

	mat_arr mat_element_wise_add(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		return mat_element_by_element_operation(A, B, C, _mat_e_by_e_add_kernel(), tr);
	}

	struct _mat_e_by_e_sub_kernel
	{
		float operator()(float a, float b) const
		{
			return a - b;
		}
	};

	mat_arr mat_element_wise_sub(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		return mat_element_by_element_operation(A, B, C, _mat_e_by_e_sub_kernel(), tr);
	}

	struct _mat_e_by_e_mul_kernel
	{
		float operator()(float a, float b) const
		{
			return a * b;
		}
	};

	mat_arr mat_element_wise_mul(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		return mat_element_by_element_operation(A, B, C, _mat_e_by_e_mul_kernel(), tr);
	}

	struct _mat_e_by_e_div_kernel
	{
		float operator()(float a, float b) const
		{
			return a / b;
		}
	};

	mat_arr mat_element_wise_div(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		return mat_element_by_element_operation(A, B, C, _mat_e_by_e_div_kernel(), tr);
	}

	struct _mat_e_wise_add_kernel
	{
		float b;

		explicit _mat_e_wise_add_kernel(float b) : b(b)
		{
		}

		float operator()(float a) const
		{
			return a + b;
		}
	};

	mat_arr mat_element_wise_add(const mat_arr& A, float b, mat_arr* C)
	{
		return mat_element_wise_operation(A, C, _mat_e_wise_add_kernel(b));
	}

	mat_arr mat_element_wise_add(float a, const mat_arr& B, mat_arr* C)
	{
		return mat_element_wise_operation(B, C, _mat_e_wise_add_kernel(a));
	}

	struct _mat_e_wise_sub_kernel
	{
		float b;

		explicit _mat_e_wise_sub_kernel(float b) : b(b)
		{
		}

		float operator()(float a) const
		{
			return a - b;
		}
	};

	mat_arr mat_element_wise_sub(const mat_arr& A, float b, mat_arr* C)
	{
		return mat_element_wise_operation(A, C, _mat_e_wise_sub_kernel(b));
	}

	struct _mat_e_wise_sub_kernel2
	{
		float a;

		explicit _mat_e_wise_sub_kernel2(float a) : a(a)
		{
		}

		float operator()(float b) const
		{
			return a - b;
		}
	};

	mat_arr mat_element_wise_sub(float a, const mat_arr& B, mat_arr* C)
	{
		return mat_element_wise_operation(B, C, _mat_e_wise_sub_kernel2(a));
	}

	struct _mat_e_wise_mul_kernel
	{
		float b;

		explicit _mat_e_wise_mul_kernel(float b) : b(b)
		{
		}

		float operator()(float a) const
		{
			return a * b;
		}
	};

	mat_arr mat_element_wise_mul(const mat_arr& A, float b, mat_arr* C)
	{
		return mat_element_wise_operation(A, C, _mat_e_wise_mul_kernel(b));
	}

	mat_arr mat_element_wise_mul(float a, const mat_arr& B, mat_arr* C)
	{
		return mat_element_wise_operation(B, C, _mat_e_wise_mul_kernel(a));
	}

	struct _mat_e_wise_div_kernel
	{
		float b;

		explicit _mat_e_wise_div_kernel(float b) : b(b)
		{
		}

		float operator()(float a) const
		{
			return a / b;
		}
	};

	mat_arr mat_element_wise_div(const mat_arr& A, float b, mat_arr* C)
	{
		return mat_element_wise_operation(A, C, _mat_e_wise_div_kernel(b));
	}

	struct _mat_e_wise_div_kernel2
	{
		float a;

		explicit _mat_e_wise_div_kernel2(float a) : a(a)
		{
		}

		float operator()(float b) const
		{
			return a / b;
		}
	};

	mat_arr mat_element_wise_div(float a, const mat_arr& B, mat_arr* C)
	{
		return mat_element_wise_operation(B, C, _mat_e_wise_div_kernel2(a));
	}

	inline void __matrix_mul_size_check(const unsigned count_a, const unsigned rows_a, const unsigned cols_a,
	                                    const unsigned count_b, const unsigned rows_b, const unsigned cols_b,
	                                    const unsigned count_c, const unsigned rows_c, const unsigned cols_c)
	{
		if (count_a != 1 && count_b != 1 && count_a != count_b)
		{
			throw std::runtime_error("Wrong input array sizes");
		}

		if (count_c != std::max(count_a, count_b))
		{
			throw std::runtime_error("Wrong output array sizes");
		}

		if (cols_a != rows_b)
		{
			throw std::runtime_error("A and B cannot be multiplied");
		}

		if (rows_a != rows_c || cols_b != cols_c)
		{
			throw std::runtime_error("C has wrong size");
		}
	}

	struct _mat_mul_case0_kernel
	{
		// Cache miss analysis: Inner Loop
		// A fixed, B row-wise, C row-wise
		void operator()(unsigned l, unsigned m, unsigned n,
		                const float* a, const float* b, float* c) const
		{
			for (unsigned i = 0; i < l; i++)
			{
				const float* a_row = a + i * m;
				float* c_row = c + i * n;
				for (unsigned j = 0; j < m; j++)
				{
					const float a_val = a_row[j];
					const float* b_row = b + j * n;
					for (unsigned k = 0; k < n; k++)
					{
						c_row[k] += a_val * b_row[k];
					}
				}
			}
		};
	};

	struct _mat_mul_case1_kernel
	{
		// Cache miss analysis: Inner Loop
		// A fixed, B row-wise, C row-wise
		void operator()(unsigned l, unsigned m, unsigned n,
		                const float* a, const float* b, float* c) const
		{
			for (unsigned j = 0; j < m; j++)
			{
				const float* a_row = a + j * l;
				const float* b_row = b + j * n;
				for (unsigned i = 0; i < l; i++)
				{
					const float a_val = a_row[i];
					float* c_row = c + i * n;
					for (unsigned k = 0; k < n; k++)
					{
						c_row[k] += a_val * b_row[k];
					}
				}
			}
		};
	};

	struct _mat_mul_case2_kernel
	{
		// Cache miss analysis: Inner Loop
		// A row-wise, B row-wise, C fixed
		void operator()(unsigned l, unsigned m, unsigned n,
		                const float* a, const float* b, float* c) const
		{
			for (unsigned i = 0; i < l; i++)
			{
				const float* a_row = a + i * m;
				float* c_row = c + i * n;
				for (unsigned k = 0; k < n; k++)
				{
					const float* b_row = b + k * m;
					float c_val = c_row[k];
					for (unsigned j = 0; j < m; j++)
					{
						c_val += a_row[j] * b_row[j];
					}
					c_row[k] = c_val;
				}
			}
		};
	};

	struct _mat_mul_case3_kernel
	{
		// Cache miss analysis: Inner Loop
		// A row-wise, B fixed, C column-wise (many cache misses!)
		void operator()(unsigned l, unsigned m, unsigned n,
		                const float* a, const float* b, float* c) const
		{
			for (unsigned k = 0; k < n; k++)
			{
				const float* b_row = b + k * m;
				for (unsigned j = 0; j < m; j++)
				{
					const float* a_row = a + j * l;
					const float b_val = b_row[j];
					for (unsigned i = 0; i < l; i++)
					{
						c[i * n + k] += a_row[i] * b_val;
					}
				}
			}
		};
	};

	template <typename Case>
	void __mat_matrix_mul_add_launch(const mat_arr& A, const mat_arr& B, mat_arr* C,
	                                 unsigned l, unsigned m, unsigned n, Case cs)
	{
		const unsigned count = C->count;

		const float* a_start = A.start();
		const float* b_start = B.start();
		float* c_start = C->start();

		const bool a_is_array = A.count > 1;
		const bool b_is_array = B.count > 1;

		for (unsigned matNo = 0; matNo < count; matNo++)
		{
			const float* a = a_is_array ? a_start + (matNo * l * m) : a_start;
			const float* b = b_is_array ? b_start + (matNo * m * n) : b_start;
			float* c = c_start + (matNo * l * n);

			cs(l, m, n, a, b, c);
		}

#ifdef MATARRMATH_CHECK_NAN
		if (!C->only_real())
		{
			throw std::runtime_error("nan");
		}
#endif
	}

	void __mat_matrix_mul_add(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		const bool transpose_a = tr == transpose_A || tr == transpose_both;
		const bool transpose_b = tr == transpose_B || tr == transpose_both;
		__matrix_mul_size_check(A.count, transpose_a ? A.cols : A.rows, transpose_a ? A.rows : A.cols,
		                        B.count, transpose_b ? B.cols : B.rows, transpose_b ? B.rows : B.cols,
		                        C->count, C->rows, C->cols);

		if (A.start() == C->start() || B.start() == C->start())
		{
			throw std::runtime_error("Matrix mul in place not possible");
		}

		switch (tr)
		{
		case transpose_no:
			__mat_matrix_mul_add_launch(A, B, C, A.rows, A.cols, B.cols,
			                            _mat_mul_case0_kernel());
			return;
		case transpose_A:
			__mat_matrix_mul_add_launch(A, B, C, A.cols, A.rows, B.cols,
			                            _mat_mul_case1_kernel());
			return;
		case transpose_B:
			__mat_matrix_mul_add_launch(A, B, C, A.rows, A.cols, B.rows,
			                            _mat_mul_case2_kernel());
			return;
		case transpose_both:
			__mat_matrix_mul_add_launch(A, B, C, A.cols, A.rows, B.rows,
			                            _mat_mul_case3_kernel());
			return;
		}
	}

	void __mat_matrix_mul(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		mat_set_all(0, C);
		__mat_matrix_mul_add(A, B, C, tr);
	}

	mat_arr mat_matrix_mul_add(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		if (C == nullptr)
		{
			mat_arr tempC = mat_arr(std::max(A.count, B.count),
			                        (tr == transpose_A || tr == transpose_both) ? A.cols : A.rows,
			                        (tr == transpose_B || tr == transpose_both) ? B.rows : B.cols);
			__mat_matrix_mul_add(A, B, &tempC, tr);
			return tempC;
		}
		__mat_matrix_mul_add(A, B, C, tr);

		return *C;
	}

	mat_arr mat_matrix_mul(const mat_arr& A, const mat_arr& B, mat_arr* C, const mat_tr tr)
	{
		if (C == nullptr)
		{
			mat_arr tempC = mat_arr(std::max(A.count, B.count),
			                        (tr == transpose_A || tr == transpose_both) ? A.cols : A.rows,
			                        (tr == transpose_B || tr == transpose_both) ? B.rows : B.cols);
			__mat_matrix_mul(A, B, &tempC, tr);
			return tempC;
		}
		__mat_matrix_mul(A, B, C, tr);

		return *C;
	}

	void __transpose_size_check(const unsigned count_a, const unsigned rows_a, const unsigned cols_a,
	                            const unsigned count_c, const unsigned rows_c, const unsigned cols_c)
	{
		if (count_a != count_c)
		{
			throw std::runtime_error("Wrong array sizes");
		}

		if (rows_a != cols_c || cols_a != rows_c)
		{
			throw std::runtime_error("Wrong matrix dimensions");
		}
	}

	void __mat_transpose(const mat_arr& A, mat_arr* C)
	{
		__transpose_size_check(A.count, A.rows, A.cols,
		                       C->count, C->rows, C->cols);

		const float* a_start = A.start();
		float* c_start = C->start();

		const unsigned count = C->count;
		const unsigned rows = C->rows;
		const unsigned cols = C->cols;

		for (unsigned mat_no = 0; mat_no < count; mat_no++)
		{
			const unsigned offset = mat_no * rows * cols;
			const float* a = a_start + offset;
			float* c = c_start + offset;

			unsigned i_normal = 0;
			unsigned i_transposed = 0;
			for (unsigned i = 0; i < rows; ++i)
			{
				for (unsigned col = 0; col < cols; ++col)
				{
					c[i_normal] = a[i_transposed];
					i_normal++;
					i_transposed += rows;
				}
				i_transposed -= (cols * rows - 1);
			}
		}
	}

	mat_arr mat_transpose(const mat_arr& A, mat_arr* C)
	{
		if (C == nullptr)
		{
			mat_arr tempC = mat_arr(A.count, A.cols, A.rows);
			__mat_transpose(A, C);
			return tempC;
		}
		__mat_transpose(A, C);
		return *C;
	}

	mat_arr mat_set_all(float val, mat_arr* C)
	{
		if (C == nullptr)
		{
			throw std::runtime_error("C is nullptr");
		}
		const unsigned size = C->size();
		float* c = C->start();
		for (unsigned i = 0; i < size; i++)
		{
			c[i] = val;
		}
		return *C;
	}

	void __mat_concat_size_check(const std::vector<mat_arr>& mats, mat_arr* C)
	{
		const auto mats_count = static_cast<unsigned>(mats.size());
		unsigned count_sum = 0;
		for (unsigned i = 0; i < mats_count; i++)
		{
			if (C->rows != mats[i].rows)
			{
				throw std::runtime_error("Row count does not fit");
			}

			if (C->cols != mats[i].cols)
			{
				throw std::runtime_error("Column count does not fit");
			}

			count_sum += mats[i].count;
		}

		if (C->count != count_sum)
		{
			throw std::runtime_error("Array sizes do not fit");
		}
	}

	void __mat_concat_mats(const std::vector<mat_arr>& mats, mat_arr* C)
	{
		__mat_concat_size_check(mats, C);
		const auto mat_arr_count = static_cast<unsigned>(mats.size());
		const unsigned row_col = C->rows * C->cols;
		float* c = C->start();

		for (unsigned mat_arr_no = 0; mat_arr_no < mat_arr_count; mat_arr_no++)
		{
			const unsigned size = row_col * mats[mat_arr_no].count;
			const float* m = mats[mat_arr_no].start();

			for (unsigned i = 0; i < size; i++)
			{
				*c = *m;
				c++;
				m++;
			}
		}
	}

	mat_arr mat_concat_mats(const std::vector<mat_arr>& mats, mat_arr* C)
	{
		if (C == nullptr)
		{
			unsigned count = 0;
			for (const auto& mat : mats)
			{
				count += mat.count;
			}
			mat_arr tempC = mat_arr(count, mats[0].rows, mats[0].cols);
			__mat_concat_mats(mats, &tempC);
			return tempC;
		}
		__mat_concat_mats(mats, C);
		return *C;
	}

	void __mat_select_mats_size_check(const mat_arr& A, const std::vector<unsigned>& indices, mat_arr* C)
	{
		if (C->rows != A.rows)
		{
			throw std::runtime_error("Row count does not fit");
		}

		if (C->cols != A.cols)
		{
			throw std::runtime_error("Column count does not fit");
		}

		if (static_cast<unsigned>(indices.size()) != C->count)
		{
			throw std::runtime_error("Array sizes do not fit");
		}
	}

	void __mat_select_mats(const mat_arr& A, const std::vector<unsigned>& indices, mat_arr* C)
	{
		__mat_select_mats_size_check(A, indices, C);

		const auto mat_count = static_cast<unsigned>(indices.size());
		const unsigned row_col = C->rows * C->cols;

		const float* a_start = A.start();
		float* c_start = C->start();

		for (unsigned mat_no = 0; mat_no < mat_count; mat_no++)
		{
			const float* a = a_start + indices[mat_no] * row_col;
			float* c = c_start + (mat_no * row_col);

			for (unsigned i = 0; i < row_col; i++)
			{
				c[i] = a[i];
			}
		}
	}

	mat_arr mat_select_mats(const mat_arr& A, const std::vector<unsigned>& indices, mat_arr* C)
	{
		if (C == nullptr)
		{
			mat_arr tempC = mat_arr(static_cast<unsigned>(indices.size()), A.rows, A.cols);
			__mat_select_mats(A, indices, &tempC);
			return tempC;
		}
		__mat_select_mats(A, indices, C);
		return *C;
	}

	mat_arr mat_random_gaussian(float mean, float sigma, std::random_device* rnd, mat_arr* C)
	{
		std::normal_distribution<float> distr(mean, sigma);
		return mat_element_wise_operation(*C, C, [&](float a)
		{
			return distr(*rnd);
		});
	}
}
