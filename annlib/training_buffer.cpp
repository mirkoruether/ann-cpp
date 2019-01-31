#include "training_buffer.h"
#include "mat_arr_math.h"

using namespace annlib;

mat_arr* training_buffer::in(unsigned layer_no)
{
	return &activations[layer_no];
}

mat_arr* training_buffer::out(unsigned layer_no)
{
	return &activations[layer_no + 1];
}

mat_arr* training_buffer::bpterm(unsigned layer_no)
{
	return &backprop_terms[layer_no];
}

mat_arr* training_buffer::sol()
{
	return &solution;
}

layer_buffer* training_buffer::lbuf(unsigned layer_no)
{
	return &lbufs[layer_no];
}

training_buffer::training_buffer(unsigned mini_batch_size, std::vector<unsigned> sizes)
	: solution(mini_batch_size, 1, sizes[sizes.size() - 1])
{
	const unsigned layer_count = static_cast<unsigned>(sizes.size() - 1);
	activations.emplace_back(mini_batch_size, 1, sizes[0]);
	for (unsigned i = 1; i < layer_count + 1; i++)
	{
		activations.emplace_back(mini_batch_size, 1, sizes[i]);
		backprop_terms.emplace_back(mini_batch_size, 1, sizes[i]);
	}

	for(unsigned i=0; i<layer_count;i++)
	{
		lbufs.emplace_back(mini_batch_size, &activations[i], &activations[i + 1], &backprop_terms[i]);
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

layer_buffer::layer_buffer(unsigned mini_batch_size, const mat_arr* in,
                           const mat_arr* out, const mat_arr* backprop_term)
	: mini_batch_size(mini_batch_size),
	  in(in), out(out),
	  backprop_term(backprop_term)
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

void layer_buffer::add(const std::string& key, unsigned count, unsigned rows, unsigned cols, bool split)
{
	m.insert_or_assign(key, std::make_shared<buf>(split, count, rows, cols));
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
