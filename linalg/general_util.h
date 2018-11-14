#ifndef GENERAL_UTIL_H
#define GENERAL_UTIL_H

#include <functional>
#include <array>

using namespace std;

namespace linalg
{
	template <typename in_t, typename out_t, size_t count>
	array<out_t, count> array_select(const array<in_t, count>& in, function<out_t(in_t)> f)
	{
		array<out_t, count> out{};
		for (size_t i = 0; i < count; i++)
		{
			out[i] = f(in[i]);
		}
		return out;
	}

	template <typename in_t, typename out_t>
	vector<out_t> vector_select(const vector<in_t>& in, function<out_t(in_t)> f)
	{
		vector<out_t> out(in.size());
		for (size_t i = 0; i < in.size(); i++)
		{
			out[i] = f(in[i]);
		}
		return out;
	}

	template <typename t>
	void add_pointers(vector<t>* in, vector<t*>* target)
	{
		for (size_t i = 0; i < in->size(); i++)
		{
			target->emplace_back(&in->operator[](i));
		}
	}
}
#endif
