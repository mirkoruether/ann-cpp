#include "mat_arr.h"
#include <utility>

using namespace linalg;

mat_arr::mat_arr(shared_ptr<vector<double>> vector, unsigned offset,
                 unsigned count, unsigned rows, unsigned cols)
	: vec(std::move(vector)),
	  offset(offset),
	  count(count),
	  rows(rows),
	  cols(cols)
{
}

mat_arr::mat_arr(unsigned count, unsigned rows, unsigned cols)
	: vec(make_shared<vector<double>>(count * rows * cols)),
	  offset(0),
	  count(count),
	  rows(rows),
	  cols(cols)
{
}

mat_arr::mat_arr(array<unsigned, 3> dim)
	: mat_arr(dim[0], dim[1], dim[2])
{
}

mat_arr mat_arr::get_mat(unsigned index)
{
	return get_mats(index, 1);
}

const mat_arr mat_arr::get_mat(unsigned index) const
{
	return get_mats(index, 1);
}

mat_arr mat_arr::get_mats(unsigned start, unsigned count)
{
	if (count + start > this->count)
	{
		throw runtime_error("Out of bounds");
	}

	return mat_arr(vec, start * rows * cols, count, rows, cols);
}

const mat_arr mat_arr::get_mats(unsigned start, unsigned count) const
{
	if (count + start > this->count)
	{
		throw runtime_error("Out of bounds");
	}

	return mat_arr(vec, start * rows * cols, count, rows, cols);
}

unsigned mat_arr::index(unsigned index, unsigned row, unsigned col) const
{
	return index * rows * cols + row * cols + col;
}

array<unsigned, 3> mat_arr::dim() const
{
	return array<unsigned, 3>
	{
		count, rows, cols
	};
}

unsigned mat_arr::size() const
{
	return count * rows * cols;
}

double* mat_arr::start()
{
	return vec->data() + offset;
}

const double* mat_arr::start() const
{
	return vec->data() + offset;
}

const double& mat_arr::operator[](unsigned index) const
{
	return *(start() + index);
}

double& mat_arr::operator[](unsigned index)
{
	return *(start() + index);
}

mat_arr mat_arr::duplicate() const
{
	mat_arr result(dim());
	const double* a = start();
	std::copy(a, a + size(), result.start());
	return result;
}

bool mat_arr::only_real() const
{
	const unsigned s = size();
	const double* a = start();
	for (unsigned i = 0; i < s; i++)
	{
		const double val = *(a + i);
		if (!isnormal(val) && val != 0.0)
		{
			return false;
		}
	}
	return true;
}
