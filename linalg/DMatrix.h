//
// Created by Mirko on 20.02.2018.
//

#ifndef LINALG_DMATRIX_H
#define LINALG_DMATRIX_H

#include <cstdlib>
#include <vector>
#include <functional>
#include <memory>

using namespace std;
using vec_ptr = shared_ptr<vector<double> >;

namespace linalg {
    class DRowVector;

    class DColumnVector;

    class DMatrix {
    protected:
        unsigned columnCount;
        vec_ptr vec;

    public:
        DMatrix(vec_ptr vec_p, unsigned columnCount);

        DMatrix(unsigned rowCount, unsigned columnCount);

        virtual ~DMatrix();

        unsigned getRowCount() const;

        unsigned getColumnCount() const;

        unsigned getLength() const;

        virtual DMatrix &dup() const;

        virtual DMatrix &transpose() const;

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

        virtual DMatrix &operator+(const DMatrix &other) const;

        virtual DMatrix &operator+=(const DMatrix &other);

        virtual DMatrix &addInPlace(const DMatrix &other);

        virtual DMatrix &operator-(const DMatrix &other) const;

        virtual DMatrix &operator-=(const DMatrix &other);

        virtual DMatrix &subInPlace(const DMatrix &other);

        virtual DMatrix &elementWiseMul(const DMatrix &other) const;

        virtual DMatrix &elementWiseMulInPlace(const DMatrix &other);

        virtual DMatrix &elementWiseDiv(const DMatrix &other) const;

        virtual DMatrix &elementWiseDivInPlace(const DMatrix &other);

        virtual DMatrix &operator*(double r) const;

        virtual DMatrix &operator*=(double r);

        virtual DMatrix &scalarMulInPlace(double r);

        virtual DMatrix &operator/(double r) const;

        virtual DMatrix &operator/=(double r);

        virtual DMatrix &scalarDivInPlace(double r);

        virtual DMatrix &applyFunctionToElements(const function<double(double)> &func) const;

        virtual DMatrix &applyFunctionToElementsInPlace(const function<double(double)> &func);

        bool isRowVector() const;

        explicit operator DRowVector();

        DRowVector &toRowVectorDuplicate() const;

        DRowVector &asRowVector();

        bool isColumnVector() const;

        explicit operator DColumnVector();

        DColumnVector &toColumnVectorDuplicate() const;

        DColumnVector &asColumnVector();

        bool isScalar() const;

        explicit operator double();

        double toScalar() const;

        double vector_innerProduct(const DMatrix &other) const;

        double vector_norm() const;
    };
}


#endif //LINALG_DMATRIX_H
