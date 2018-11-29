#ifndef GENERAL_UTIL_H
#define GENERAL_UTIL_H

#include <functional>

namespace linalg
{
	template <typename in_t, typename out_t>
	std::vector<out_t> vector_select(const std::vector<in_t>& in, std::function<out_t(in_t)> f)
	{
		std::vector<out_t> out(in.size());
		for (size_t i = 0; i < in.size(); i++)
		{
			out[i] = f(in[i]);
		}
		return out;
	}

	template <typename t>
	void add_pointers(std::vector<t>* in, std::vector<t*>* target)
	{
		for (size_t i = 0; i < in->size(); i++)
		{
			target->emplace_back(&in->operator[](i));
		}
	}
}
#endif
