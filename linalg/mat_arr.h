#ifndef MAT_ARR_H
#define MAT_ARR_H

#include <memory>
#include <vector>
#include <array>

namespace linalg
{
	class mat_arr
	{
	private:
		const std::shared_ptr<std::vector<double>> vec;
		const unsigned offset;

	public:
		const unsigned count;
		const unsigned rows;
		const unsigned cols;

	protected:
		mat_arr(std::shared_ptr<std::vector<double>> vector, unsigned offset,
		        unsigned count, unsigned rows, unsigned cols);

	public:
		mat_arr(unsigned count, unsigned rows, unsigned cols);

		explicit mat_arr(std::array<unsigned, 3> dim);

		mat_arr get_mat(unsigned index);

		const mat_arr get_mat(unsigned index) const;

		mat_arr get_mats(unsigned start, unsigned count);

		const mat_arr get_mats(unsigned start, unsigned count) const;

		unsigned index(unsigned index, unsigned row, unsigned col) const;

		std::array<unsigned, 3> dim() const;

		unsigned size() const;

		double* start();

		const double* start() const;

		const double& operator[](unsigned index) const;

		double& operator[](unsigned index);

		mat_arr duplicate() const;

		bool only_real() const;
	};
}

#endif
