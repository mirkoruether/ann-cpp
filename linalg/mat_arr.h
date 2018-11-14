#ifndef MAT_ARR_H
#define MAT_ARR_H

#include <memory>
#include <vector>
#include <array>

using namespace std;

namespace linalg
{
	class mat_arr
	{
	private:
		const shared_ptr<vector<double>> vec;
		const unsigned offset;

	public:
		const unsigned count;
		const unsigned rows;
		const unsigned cols;


	protected:
		mat_arr(shared_ptr<vector<double>> vector, unsigned offset,
		        unsigned count, unsigned rows, unsigned cols);


	public:
		mat_arr(unsigned count, unsigned rows, unsigned cols);

		explicit mat_arr(array<unsigned, 3> dim);

		mat_arr get_mat(unsigned index) const;

		unsigned index(unsigned index, unsigned row, unsigned col) const;

		array<unsigned, 3> dim() const;

		unsigned size() const;

		double* start();

		const double* start() const;

		const double& operator[](unsigned index) const;

		double& operator[](unsigned index);
	};
}

#endif
