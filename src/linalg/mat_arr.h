#ifndef MAT_ARR_H
#define MAT_ARR_H

#ifdef LINALG_DOUBLE
#define fpt double
#else
#define fpt float
#endif

#include <memory>
#include <array>
#include "synced_vectors.h"

namespace linalg
{
	enum mat_tr
	{
		transpose_no,
		transpose_A,
		transpose_B,
		transpose_both
	};

	class mat_arr
	{
	private:
		const std::shared_ptr<synced_vectors<fpt>> vec;
		const unsigned offset;

	public:
		const unsigned count;
		const unsigned rows;
		const unsigned cols;

	protected:
		mat_arr(std::shared_ptr<synced_vectors<fpt>> vector, unsigned offset,
		        unsigned count, unsigned rows, unsigned cols);

	public:
		mat_arr();

		mat_arr(unsigned count, unsigned rows, unsigned cols);

		explicit mat_arr(std::array<unsigned, 3> dim);

		mat_arr get_mat(unsigned index);

		const mat_arr get_mat(unsigned index) const;

		mat_arr get_mats(unsigned start, unsigned count);

		const mat_arr get_mats(unsigned start, unsigned count) const;

		unsigned index(unsigned index, unsigned row, unsigned col) const;

		std::array<unsigned, 3> dim() const;

		unsigned size() const;

		fpt* start();

		const fpt* start() const;

		const fpt& operator[](unsigned index) const;

		fpt& operator[](unsigned index);

		fpt* dev_start();

		const fpt* dev_start() const;

		mat_arr duplicate(bool try_device_copy = true) const;

		bool only_real() const;

		void assert_only_real() const;

		friend std::ostream& operator<<(std::ostream& stream, const mat_arr& matrix);
	};
}

#endif
