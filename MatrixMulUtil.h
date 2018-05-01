//
// Created by Mirko on 01.05.2018.
//

#ifndef ANN_CPP_MATRIXMULUTIL_H
#define ANN_CPP_MATRIXMULUTIL_H

#include "DMatrix.h"

using namespace std;
using namespace linalg;

namespace linalg {
    DMatrix matrix_Mul(bool transposeA, bool transposeB, DMatrix &a, DMatrix &b);

    //A not transposed, B not transposed
    DMatrix matrix_Mul_Case0(const DMatrix &a, const DMatrix &b);

    //A transposed, B not transposed
    DMatrix matrix_Mul_Case1(const DMatrix &a, const DMatrix &b);

    //A not transposed, B transposed
    DMatrix matrix_Mul_Case2(const DMatrix &a, const DMatrix &b);

    //A transposed, B transposed
    DMatrix matrix_Mul_Case3(const DMatrix &a, const DMatrix &b);
}


#endif //ANN_CPP_MATRIXMULUTIL_H
