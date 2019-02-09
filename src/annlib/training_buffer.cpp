#include "training_buffer.h"
#include "mat_arr_math.h"

using namespace annlib;

training_buffer::training_buffer(training_buffer* buf, unsigned start, unsigned count)
	: mini_batch_size(count)
{
	for (auto& act : buf->activations)
	{
		activations.emplace_back(act.get_mats(start, count));
	}

	for (auto& err : buf->errors)
	{
		errors.emplace_back(err.get_mats(start, count));
	}

	for (auto& lb : buf->lbufs)
	{
		lbufs.emplace_back(lb.get_part(start, count));
	}
}

unsigned training_buffer::layer_count() const
{
	return static_cast<unsigned>(lbufs.size());
}

mat_arr* training_buffer::in(unsigned layer_no)
{
	return &activations[layer_no];
}

mat_arr* training_buffer::out(unsigned layer_no)
{
	return &activations[layer_no + 1];
}

mat_arr* training_buffer::error(unsigned layer_no)
{
	return &errors[layer_no];
}

mat_arr* training_buffer::sol()
{
	return &errors.back();
}

layer_buffer* training_buffer::lbuf(unsigned layer_no)
{
	return &lbufs[layer_no];
}

std::vector<training_buffer> training_buffer::do_split(unsigned part_count)
{
	std::vector<training_buffer> partial_buffers;

	const unsigned part_size = mini_batch_size / part_count;
	const unsigned remainder = mini_batch_size % part_count;
	unsigned current_start = 0;
	for (unsigned i = 0; i < part_count; i++)
	{
		const unsigned current_part_size = i < remainder ? part_size + 1 : part_size;
		partial_buffers.emplace_back(this, current_start, current_part_size);
		current_start += current_part_size;
	}

	return partial_buffers;
}

training_buffer::training_buffer(unsigned mini_batch_size, std::vector<unsigned> sizes)
	: mini_batch_size(mini_batch_size)
{
	const auto layer_count = static_cast<unsigned>(sizes.size() - 1);
	activations.emplace_back(mini_batch_size, 1, sizes[0]);
	for (unsigned i = 1; i < layer_count + 1; i++)
	{
		activations.emplace_back(mini_batch_size, 1, sizes[i]);
		errors.emplace_back(mini_batch_size, 1, sizes[i]);
	}

	for (unsigned i = 0; i < layer_count; i++)
	{
		lbufs.emplace_back(mini_batch_size, activations[i], activations[i + 1], errors[i]);
	}
}

void layer_buffer::add_mini_batch_size(const std::string& key, unsigned rows, unsigned cols)
{
	add(key, mini_batch_size, rows, cols, true);
}

void layer_buffer::add_single(const std::string& key, unsigned rows, unsigned cols)
{
	add_custom_count(key, 1, rows, cols);
}

layer_buffer::layer_buffer(unsigned mini_batch_size, mat_arr in,
                           mat_arr out, mat_arr error)
	: mini_batch_size(mini_batch_size),
	  in(std::move(in)), out(std::move(out)),
	  error(std::move(error))
{
}

void layer_buffer::add_custom_count(const std::string& key, std::array<unsigned, 3> dim)
{
	add_custom_count(key, dim[0], dim[1], dim[2]);
}

void layer_buffer::add_custom_count(const std::string& key, unsigned count, unsigned rows, unsigned cols)
{
	add(key, count, rows, cols, false);
}

buf::buf(bool split, unsigned count, unsigned rows, unsigned cols)
	: split(split),
	  mat(count, rows, cols)
{
}

buf::buf(bool split, mat_arr mat)
	: split(split),
	  mat(std::move(mat))
{
}

void layer_buffer::add(const std::string& key, unsigned count, unsigned rows, unsigned cols, bool split)
{
	m.insert_or_assign(key, std::make_shared<buf>(split, count, rows, cols));
}

void layer_buffer::add(const std::string& key, mat_arr mat, bool split)
{
	m.insert_or_assign(key, std::make_shared<buf>(split, std::move(mat)));
}

void layer_buffer::remove(const std::string& key)
{
	m.erase(key);
}

mat_arr layer_buffer::get_val(const std::string& key)
{
	return m[key]->mat;
}

mat_arr* layer_buffer::get_ptr(const std::string& key)
{
	return &m[key]->mat;
}

layer_buffer layer_buffer::get_part(unsigned start, unsigned count)
{
	layer_buffer res(count,
	                 in.get_mats(start, count),
	                 out.get_mats(start, count),
	                 error.get_mats(start, count));

	for (const auto& e : m)
	{
		if (e.second->split)
		{
			res.add(e.first, e.second->mat.get_mats(start, count), true);
		}
		else
		{
			res.add(e.first, e.second->mat, false);
		}
	}

	return res;
}
