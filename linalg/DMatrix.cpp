//
// Created by Mirko on 20.02.2018.
//

#include "DMatrix.h"
#include <cmath>

using namespace std;

namespace linalg {
    DMatrix::DMatrix(unsigned rowCount, unsigned columnCount)
            : rowCount(rowCount), columnCount(columnCount), length(rowCount * columnCount) {
        vec = vector<double>(length);
    }

    unsigned DMatrix::getRowCount() const {
        return rowCount;
    }

    unsigned DMatrix::getColumnCount() const {
        return columnCount;
    }

    unsigned DMatrix::getLength() const {
        return length;
    }

    DMatrix DMatrix::dup() const {
        DMatrix result(rowCount, columnCount);
        for (unsigned i = 0; i < length; ++i) {
            result[i] = vec[i];
        }
        return result;
    }

    DMatrix DMatrix::transpose() const {
        DMatrix result = DMatrix(getColumnCount(), getRowCount());
        for (unsigned i = 0; i < getRowCount(); ++i) {
            for (unsigned j = 0; j < getColumnCount(); ++j) {
                result[j, i] = this->operator()(i, j);
            }
        }
        return result;
    }

    pair<unsigned, unsigned> DMatrix::getSize() const {
        return pair<unsigned, unsigned>(getRowCount(), getColumnCount());
    }

    double &DMatrix::operator[](unsigned index) {
        return vec[index];
    }

    const double &DMatrix::operator[](unsigned index) const {
        return vec[index];
    }

    unsigned DMatrix::index(unsigned row, unsigned column) const {
        return row * getColumnCount() + column;
    }

    double &DMatrix::operator()(unsigned row, unsigned column) {
        assertIndex(row, column);
        return vec[index(row, column)];
    }

    double DMatrix::operator()(unsigned row, unsigned column) const {
        assertIndex(row, column);
        return vec[index(row, column)];
    }

    void DMatrix::assertIndex(unsigned index) const {
        if (index >= length) {
            throw out_of_range("index is out of range");
        }
    }

    void DMatrix::assertIndex(unsigned row, unsigned column) const {
        if (row >= getRowCount()) {
            throw out_of_range("row is out of range");
        }
        if (column >= getColumnCount()) {
            throw out_of_range("column is out of range");
        }
    }

    void DMatrix::assertSameSize(const DMatrix &other) const {
        if (getRowCount() != other.getRowCount()
            || getColumnCount() != other.getColumnCount()) {
            throw runtime_error("sizes are not equal");
        }
    }

    void DMatrix::assertSameLength(const DMatrix &other) const {
        if (length != other.length) {
            throw runtime_error("lengths are not equal");
        }
    }

    DMatrix DMatrix::operator+(const DMatrix &other) const {
        return dup().addInPlace(other);
    }

    DMatrix DMatrix::operator+=(const DMatrix &other) {
        return addInPlace(other);
    }

    DMatrix &DMatrix::addInPlace(const DMatrix &other) {
        assertSameSize(other);
        for (unsigned i = 0; i < length; ++i) {
            vec[i] += other[i];
        }
        return *this;
    }

    DMatrix DMatrix::operator-(const DMatrix &other) const {
        return dup().subInPlace(other);
    }

    DMatrix DMatrix::operator-=(const DMatrix &other) {
        return subInPlace(other);
    }

    DMatrix &DMatrix::subInPlace(const DMatrix &other) {
        assertSameSize(other);
        for (unsigned i = 0; i < length; ++i) {
            vec[i] -= other[i];
        }
        return *this;
    }

    DMatrix DMatrix::elementWiseMul(const DMatrix &other) const {
        return dup().elementWiseMulInPlace(other);
    }

    DMatrix &DMatrix::elementWiseMulInPlace(const DMatrix &other) {
        assertSameSize(other);
        for (unsigned i = 0; i < length; ++i) {
            vec[i] *= other[i];
        }
        return *this;
    }

    DMatrix DMatrix::elementWiseDiv(const DMatrix &other) const {
        return dup().elementWiseDivInPlace(other);
    }

    DMatrix &DMatrix::elementWiseDivInPlace(const DMatrix &other) {
        assertSameSize(other);
        for (unsigned i = 0; i < length; ++i) {
            vec[i] /= other[i];
        }
        return *this;
    }

    DMatrix DMatrix::operator*(double r) const {
        return dup().scalarMulInPlace(r);
    }

    DMatrix DMatrix::operator*=(double r) {
        return scalarMulInPlace(r);
    }

    DMatrix &DMatrix::scalarMulInPlace(double r) {
        for (unsigned i = 0; i < length; ++i) {
            vec[i] *= r;
        }
        return *this;
    }

    DMatrix DMatrix::operator/(double r) const {
        return dup().scalarDivInPlace(r);
    }

    DMatrix DMatrix::operator/=(double r) {
        return scalarDivInPlace(r);
    }

    DMatrix &DMatrix::scalarDivInPlace(double r) {
        for (unsigned i = 0; i < length; ++i) {
            vec[i] /= r;
        }
        return *this;
    }

    DMatrix DMatrix::applyFunctionToElements(const function<double(double)> &func) {
        return dup().applyFunctionToElementsInPlace(func);
    }

    DMatrix &DMatrix::applyFunctionToElementsInPlace(const function<double(double)> &func) {
        for (unsigned i = 0; i < length; ++i) {
            vec[i] = func(vec[i]);
        }
        return *this;
    }

    bool DMatrix::isRowVector() const {
        return rowCount == 1;
    }

    bool DMatrix::isColVector() const {
        return columnCount == 1;
    }

    bool DMatrix::isScalar() const {
        return rowCount == 1 && columnCount == 1;
    }

    double DMatrix::vector_innerProduct(const DMatrix &other) const {
        assertSameLength(other);
        double result = 0.0;
        for (unsigned i = 0; i < length; ++i) {
            result += vec[i] * other[i];
        }
        return result;
    }

    double DMatrix::vector_norm() const {
        double result = 0.0;
        for (unsigned i = 0; i < length; ++i) {
            result += vec[i] * vec[i];
        }
        return sqrt(result);
    }
}