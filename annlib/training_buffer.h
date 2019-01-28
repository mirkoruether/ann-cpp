#ifndef TRAINING_BUFFER_H
#define TRAINING_BUFFER_H

#include "mat_arr.h"
#include <map>

using namespace linalg;

namespace annlib
{
	class partial_training_buffer;

	class training_buffer
	{
	public:
		training_buffer(std::vector<unsigned> sizes, unsigned mini_batch_size, unsigned part_count);

		mat_arr input_rv;
		mat_arr solution_rv;

		std::vector<mat_arr> weighted_inputs_rv;
		std::vector<mat_arr> activations_rv;
		std::vector<mat_arr> activation_dfs_rv;
		std::vector<mat_arr> errors_rv;

		std::vector<mat_arr> gradient_biases_rv_noarr;
		std::vector<mat_arr> gradient_weights_noarr;

		std::vector<partial_training_buffer> partial_buffers;

		std::vector<mat_arr*> all();
		void clear();

		unsigned layer_count() const;
		unsigned part_count() const;
	};

	class layer_buffer
	{
	private:
		std::map<std::string, std::unique_ptr<mat_arr>> m;

	protected:
		void add_custom_count(const std::string& key, std::array<unsigned, 3> dim);
		void add_custom_count(const std::string& key, unsigned count, unsigned rows, unsigned cols);
		
	public:
		const unsigned mini_batch_size;
		const mat_arr* in;
		const mat_arr* out;
		const mat_arr* error;

		layer_buffer(unsigned mini_batch_size, const mat_arr* in, const mat_arr* out, const mat_arr* error)
			: mini_batch_size(mini_batch_size),
			  in(in),
			  out(out),
			  error(error)
		{
		}

		void add_mini_batch_size(const std::string& key, unsigned rows, unsigned cols);
		void add_single(const std::string& key, unsigned rows, unsigned cols);
		
		void remove(const std::string& key);
		mat_arr get_val(const std::string& key);
		mat_arr* get_ptr(const std::string& key);
	};

	class partial_training_buffer
	{
	public:
		partial_training_buffer(training_buffer* buf,
		                        unsigned start, unsigned count);

		mat_arr input_rv;
		mat_arr solution_rv;

		std::vector<mat_arr> weighted_inputs_rv;
		std::vector<mat_arr> activations_rv;
		std::vector<mat_arr> activation_dfs_rv;
		std::vector<mat_arr> errors_rv;
	};
}
#endif
