#include "mat_arr.h"
#include <utility>
#include <cmath>
#include <memory>
#include <vector>

using namespace linalg;

mat_arr::mat_arr(std::shared_ptr<synced_vectors<float>> vector, unsigned offset,
                 unsigned count, unsigned rows, unsigned cols)
	: vec(std::move(vector)),
	  offset(offset),
	  count(count),
	  rows(rows),
	  cols(cols)
{
}

mat_arr::mat_arr(unsigned count, unsigned rows, unsigned cols)
	: vec(std::make_shared<synced_vectors<float>>(count * rows * cols)),
	  offset(0),
	  count(count),
	  rows(rows),
	  cols(cols)
{
}

mat_arr::mat_arr(std::array<unsigned, 3> dim)
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
		throw std::runtime_error("Out of bounds");
	}

	return mat_arr(vec, start * rows * cols, count, rows, cols);
}

const mat_arr mat_arr::get_mats(unsigned start, unsigned count) const
{
	if (count + start > this->count)
	{
		throw std::runtime_error("Out of bounds");
	}

	return mat_arr(vec, start * rows * cols, count, rows, cols);
}

unsigned mat_arr::index(unsigned index, unsigned row, unsigned col) const
{
	return index * rows * cols + row * cols + col;
}

std::array<unsigned, 3> mat_arr::dim() const
{
	return std::array<unsigned, 3>
	{
		count, rows, cols
	};
}

unsigned mat_arr::size() const
{
	return count * rows * cols;
}

float* mat_arr::start()
{
	return vec->host_data() + offset;
}

const float* mat_arr::start() const
{
	return vec->host_data() + offset;
}

float* mat_arr::dev_start()
{
	return vec->dev_data() + offset;
}

const float* mat_arr::dev_start() const
{
	return vec->dev_data() + offset;
}

const float& mat_arr::operator[](unsigned index) const
{
	return *(start() + index);
}

float& mat_arr::operator[](unsigned index)
{
	return *(start() + index);
}

mat_arr mat_arr::duplicate(bool try_device_copy) const
{
	const auto new_vec = std::make_shared<synced_vectors<float>>(*vec, offset, size(), try_device_copy);
	return mat_arr(new_vec, 0, count, rows, cols);
}

bool mat_arr::only_real() const
{
	const unsigned s = size();
	const float* a = start();
	for (unsigned i = 0; i < s; i++)
	{
		const float val = *(a + i);
		if (!std::isfinite(val))
		{
			return false;
		}
	}
	return true;
}

void mat_arr::assert_only_real() const
{
	if(!only_real())
	{
		throw std::runtime_error("nan");
	}
}
