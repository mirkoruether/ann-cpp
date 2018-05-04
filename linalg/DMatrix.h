//
// Created by Mirko on 20.02.2018.
//

#ifndef LINALG_DMATRIX_H
#define LINALG_DMATRIX_H

#include <cstdlib>
#include <vector>
#include <functional>

using namespace std;

namespace linalg {
    class DMatrix {
    private:
        unsigned rowCount;
        unsigned columnCount;
        unsigned length;
        vector<double> vec;

    public:
        DMatrix(unsigned rowCount, unsigned columnCount);

        unsigned getRowCount() const;

        unsigned getColumnCount() const;

        unsigned getLength() const;

        DMatrix dup() const;

        DMatrix transpose() const;

        pair<unsigned, unsigned> getSize() const;

        double &operator[](unsigned index);

        const double &operator[](unsigned index) const;

        unsigned index(unsigned row, unsigned column) const;

        double &operator()(unsigned row, unsigned column);

        double operator()(unsigned row, unsigned column) const;

        void assertIndex(unsigned index) const;

        void assertIndex(unsigned row, unsigned column) const;

        void assertSameSize(const DMatrix &other) const;

        void assertSameLength(const DMatrix &other) const;

        DMatrix operator+(const DMatrix &other) const;

        DMatrix operator+=(const DMatrix &other);

        DMatrix addInPlace(const DMatrix &other);

        DMatrix operator-(const DMatrix &other) const;

        DMatrix operator-=(const DMatrix &other);

        DMatrix subInPlace(const DMatrix &other);

        DMatrix elementWiseMul(const DMatrix &other) const;

        DMatrix elementWiseMulInPlace(const DMatrix &other);

        DMatrix elementWiseDiv(const DMatrix &other) const;

        DMatrix elementWiseDivInPlace(const DMatrix &other);

        DMatrix operator*(double r) const;

        DMatrix operator*=(double r);

        DMatrix scalarMulInPlace(double r);

        DMatrix operator/(double r) const;

        DMatrix operator/=(double r);

        DMatrix scalarDivInPlace(double r);

        DMatrix applyFunctionToElements(const function<double(double)> &func);

        DMatrix applyFunctionToElementsInPlace(const function<double(double)> &func);

        bool isRowVector() const;

        bool isColVector() const;

        bool isScalar() const;

        double vector_innerProduct(const DMatrix &other) const;

        double vector_norm() const;
    };
}


#endif //LINALG_DMATRIX_H
