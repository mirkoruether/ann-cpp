//
// Created by Mirko on 01.05.2018.
//

#ifndef ANN_CPP_MATRIXMULUTIL_H
#define ANN_CPP_MATRIXMULUTIL_H

#include "DMatrix.h"

using namespace std;
using namespace linalg;

namespace linalg {
    DMatrix matrix_Mul(bool transposeA, bool transposeB, const DMatrix &a, const DMatrix &b);
}


#endif //ANN_CPP_MATRIXMULUTIL_H
