//
// Created by Mirko on 01.05.2018.
//

#include <stdexcept>
#include "MatrixMulUtil.h"

namespace linalg {
    DMatrix matrix_Mul_Case0(const DMatrix &a, const DMatrix &b);

    DMatrix matrix_Mul_Case1(const DMatrix &a, const DMatrix &b);

    DMatrix matrix_Mul_Case2(const DMatrix &a, const DMatrix &b);

    DMatrix matrix_Mul_Case3(const DMatrix &a, const DMatrix &b);

    DMatrix matrix_Mul(bool transposeA, bool transposeB, const DMatrix &a, const DMatrix &b) {
        int key = transposeA + 2 * transposeB;
        switch (key) {
            case 0:
                return matrix_Mul_Case0(a, b);
            case 1:
                return matrix_Mul_Case1(a, b);
            case 2:
                return matrix_Mul_Case2(a, b);
            case 3:
                return matrix_Mul_Case3(a, b);
            default:
                throw runtime_error("illegal case");
        }
    }

    //A not transposed, B not transposed
    DMatrix matrix_Mul_Case0(const DMatrix &a, const DMatrix &b) {
        if (a.getColumnCount() != b.getRowCount()) {
            throw runtime_error("matrices cannot be multiplied");
        }

        unsigned l = a.getRowCount();
        unsigned m = a.getColumnCount();
        unsigned n = b.getColumnCount();

        DMatrix c = DMatrix(l, n);

        for (unsigned i = 0; i < l; i++) {
            unsigned aRow = i * m;
            unsigned cRow = i * n;
            for (unsigned k = 0; k < m; k++) {
                unsigned bRow = k * n;
                for (unsigned j = 0; j < n; j++) {
                    c[cRow + j] += a[aRow + k] * b[bRow + j];
                }
            }
        }
        return c;
    }

    //A transposed, B not transposed
    DMatrix matrix_Mul_Case1(const DMatrix &a, const DMatrix &b) {
        //TODO optimize
        return matrix_Mul_Case0(a.transpose(), b);
    }

    //A not transposed, B transposed
    DMatrix matrix_Mul_Case2(const DMatrix &a, const DMatrix &b) {
        //TODO optimize
        return matrix_Mul_Case0(a, b.transpose());
    }

    //A transposed, B transposed
    DMatrix matrix_Mul_Case3(const DMatrix &a, const DMatrix &b) {
        //TODO optimize
        return matrix_Mul_Case0(a, b).transpose();
    }
}