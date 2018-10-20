#include "dmatrix.h"
#include <cmath>

using namespace std;

namespace linalg
{
	DMatrix::DMatrix(unsigned rowCount, unsigned columnCount)
		: columnCount(columnCount)
	{
		vec = make_shared<vector<double>>(rowCount * columnCount);
	}

	DMatrix::DMatrix(vec_ptr vec_p, unsigned columnCount)
		: columnCount(columnCount), vec(move(vec_p))
	{
		if (vec == nullptr)
		{
			throw logic_error("vector pointer points to nothing");
		}
		if (vec->size() % columnCount != 0)
		{
			throw logic_error("illegal vector size");
		}
	}

	DMatrix DMatrix::ones(unsigned rows, unsigned cols)
	{
		DMatrix result(rows, cols);
		for (unsigned i = 0; i < rows * cols; i++)
		{
			result[i] = 1.0;
		}
		return result;
	}

	unsigned DMatrix::getRowCount() const
	{
		return getLength() / getColumnCount();
	}

	unsigned DMatrix::getColumnCount() const
	{
		return columnCount;
	}

	unsigned DMatrix::getLength() const
	{
		return static_cast<unsigned>(vec->size());
	}

	DMatrix DMatrix::dup() const
	{
		DMatrix result(getRowCount(), getColumnCount());
		for (unsigned i = 0; i < getLength(); ++i)
		{
			result[i] = vec->operator[](i);
		}
		return result;
	}

	DMatrix DMatrix::transpose() const
	{
		DMatrix result = DMatrix(getColumnCount(), getRowCount());
		for (unsigned i = 0; i < getRowCount(); ++i)
		{
			for (unsigned j = 0; j < getColumnCount(); ++j)
			{
				result.operator()(j, i) = this->operator()(i, j);
			}
		}
		return result;
	}

	pair<unsigned, unsigned> DMatrix::getSize() const
	{
		return pair<unsigned, unsigned>(getRowCount(), getColumnCount());
	}

	double& DMatrix::operator[](unsigned index)
	{
		return vec->operator[](index);
	}

	const double& DMatrix::operator[](unsigned index) const
	{
		return vec->operator[](index);
	}

	unsigned DMatrix::index(unsigned row, unsigned column) const
	{
		return row * getColumnCount() + column;
	}

	double& DMatrix::operator()(unsigned row, unsigned column)
	{
		assertIndex(row, column);
		return vec->operator[](index(row, column));
	}

	double DMatrix::operator()(unsigned row, unsigned column) const
	{
		assertIndex(row, column);
		return vec->operator[](index(row, column));
	}

	void DMatrix::assertIndex(unsigned index) const
	{
		if (index >= getLength())
		{
			throw out_of_range("index is out of range");
		}
	}

	void DMatrix::assertIndex(unsigned row, unsigned column) const
	{
		if (row >= getRowCount())
		{
			throw out_of_range("row is out of range");
		}
		if (column >= getColumnCount())
		{
			throw out_of_range("column is out of range");
		}
	}

	void DMatrix::assertSameSize(const DMatrix& other) const
	{
		if (getRowCount() != other.getRowCount()
			|| getColumnCount() != other.getColumnCount())
		{
			throw runtime_error("sizes are not equal");
		}
	}

	void DMatrix::assertSameLength(const DMatrix& other) const
	{
		if (getLength() != other.getLength())
		{
			throw runtime_error("lengths are not equal");
		}
	}

	DMatrix DMatrix::operator+(const DMatrix& other) const
	{
		return dup().addInPlace(other);
	}

	DMatrix DMatrix::operator+=(const DMatrix& other)
	{
		return addInPlace(other);
	}

	DMatrix DMatrix::addInPlace(const DMatrix& other)
	{
		assertSameSize(other);
		for (unsigned i = 0; i < getLength(); ++i)
		{
			vec->operator[](i) += other[i];
		}
		return *this;
	}

	DMatrix DMatrix::operator-(const DMatrix& other) const
	{
		return dup().subInPlace(other);
	}

	DMatrix DMatrix::operator-=(const DMatrix& other)
	{
		return subInPlace(other);
	}

	DMatrix DMatrix::subInPlace(const DMatrix& other)
	{
		assertSameSize(other);
		for (unsigned i = 0; i < getLength(); ++i)
		{
			vec->operator[](i) -= other[i];
		}
		return *this;
	}

	DMatrix DMatrix::elementWiseMul(const DMatrix& other) const
	{
		return dup().elementWiseMulInPlace(other);
	}

	DMatrix DMatrix::elementWiseMulInPlace(const DMatrix& other)
	{
		assertSameSize(other);
		for (unsigned i = 0; i < getLength(); ++i)
		{
			vec->operator[](i) *= other[i];
		}
		return *this;
	}

	DMatrix DMatrix::elementWiseDiv(const DMatrix& other) const
	{
		return dup().elementWiseDivInPlace(other);
	}

	DMatrix DMatrix::elementWiseDivInPlace(const DMatrix& other)
	{
		assertSameSize(other);
		for (unsigned i = 0; i < getLength(); ++i)
		{
			vec->operator[](i) /= other[i];
		}
		return *this;
	}

	DMatrix DMatrix::operator*(double r) const
	{
		return dup().scalarMulInPlace(r);
	}

	DMatrix DMatrix::operator*=(double r)
	{
		return scalarMulInPlace(r);
	}

	DMatrix DMatrix::scalarMulInPlace(double r)
	{
		for (unsigned i = 0; i < getLength(); ++i)
		{
			vec->operator[](i) *= r;
		}
		return *this;
	}

	DMatrix DMatrix::operator/(double r) const
	{
		return dup().scalarDivInPlace(r);
	}

	DMatrix DMatrix::operator/=(double r)
	{
		return scalarDivInPlace(r);
	}

	DMatrix DMatrix::scalarDivInPlace(double r)
	{
		for (unsigned i = 0; i < getLength(); ++i)
		{
			vec->operator[](i) /= r;
		}
		return *this;
	}

	DMatrix DMatrix::applyFunctionToElements(const function<double(double)>& func) const
	{
		return dup().applyFunctionToElementsInPlace(func);
	}

	DMatrix DMatrix::applyFunctionToElementsInPlace(const function<double(double)>& func)
	{
		for (unsigned i = 0; i < getLength(); ++i)
		{
			vec->operator[](i) = func(vec->operator[](i));
		}
		return *this;
	}

	bool DMatrix::isRowVector() const
	{
		return getRowCount() == 1;
	}

	DRowVector DMatrix::asRowVector() const
	{
		DRowVector rowVector = DRowVector(vec);
		return rowVector;
	}

	DRowVector DMatrix::toRowVectorDuplicate() const
	{
		return dup().asRowVector();
	}

	bool DMatrix::isColumnVector() const
	{
		return getColumnCount() == 1;
	}

	DColumnVector DMatrix::asColumnVector() const
	{
		DColumnVector columnVector = DColumnVector(vec);
		return columnVector;
	}

	DColumnVector DMatrix::toColumnVectorDuplicate() const
	{
		return dup().asColumnVector();
	}

	bool DMatrix::isScalar() const
	{
		return getRowCount() == 1 && getColumnCount() == 1;
	}

	double DMatrix::toScalar() const
	{
		return vec->operator[](0);
	}

	double DMatrix::vector_innerProduct(const DMatrix& other) const
	{
		assertSameLength(other);
		double result = 0.0;
		for (unsigned i = 0; i < getLength(); ++i)
		{
			result += vec->operator[](i) * other[i];
		}
		return result;
	}

	double DMatrix::vector_norm() const
	{
		double result = 0.0;
		for (unsigned i = 0; i < getLength(); ++i)
		{
			result += vec->operator[](i) * vec->operator[](i);
		}
		return sqrt(result);
	}

	DMatrix::operator DRowVector() const
	{
		if (!isRowVector())
			throw logic_error("Implicit conversion not possible, matrix is no row vector");
		return asRowVector();
	}

	DMatrix::operator DColumnVector() const
	{
		if (!isColumnVector())
			throw logic_error("Implicit conversion not possible, matrix is no column vector");
		return asColumnVector();
	}

	DMatrix::operator double() const
	{
		if (!isScalar())
			throw logic_error("Implicit conversion not possible, matrix is no scalar value");
		return toScalar();
	}
}
