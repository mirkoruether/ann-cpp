#ifndef TRAINING_BUFFER_H
#define TRAINING_BUFFER_H

#include "mat_arr.h"
#include <vector>
#include <map>

using namespace linalg;

namespace annlib
{
	class layer_buffer;

	class training_buffer
	{
	private:
		std::vector<layer_buffer> lbufs;
		std::vector<mat_arr> activations;
		std::vector<mat_arr> errors;

	public:
		const unsigned mini_batch_size;

		mat_arr* in(unsigned layer_no);

		mat_arr* out(unsigned layer_no);

		mat_arr* error(unsigned layer_no);

		mat_arr* sol();

		layer_buffer* lbuf(unsigned layer_no);

		std::vector<training_buffer> do_split(unsigned part_count);

		training_buffer(unsigned mini_batch_size, std::vector<unsigned> sizes);

		training_buffer(training_buffer* buf, unsigned start, unsigned count);

		unsigned layer_count() const;
	};

	struct buf
	{
	public:
		const bool split;
		mat_arr mat;

		buf(bool split, unsigned count, unsigned rows, unsigned cols);

		buf(bool split, mat_arr mat);
	};

	class layer_buffer
	{
	private:
		std::map<std::string, std::shared_ptr<buf>> m;

		void add(const std::string& key, unsigned count, unsigned rows, unsigned cols, bool split);

	protected:
		void add(const std::string& key, mat_arr mat, bool split);

	public:
		const unsigned mini_batch_size;
		const mat_arr in;
		const mat_arr out;
		const mat_arr error;

		layer_buffer(unsigned mini_batch_size, mat_arr in, mat_arr out, mat_arr error);

		void add_custom_count(const std::string& key, std::array<unsigned, 3> dim);

		void add_custom_count(const std::string& key, unsigned count, unsigned rows, unsigned cols);

		void add_mini_batch_size(const std::string& key, unsigned rows, unsigned cols);

		void add_single(const std::string& key, unsigned rows, unsigned cols);

		void remove(const std::string& key);

		mat_arr get_val(const std::string& key);

		mat_arr* get_ptr(const std::string& key);

		layer_buffer get_part(unsigned start, unsigned count);
	};
}
#endif
