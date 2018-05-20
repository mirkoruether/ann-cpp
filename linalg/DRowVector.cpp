//
// Created by Mirko on 20.05.2018.
//

#include "DRowVector.h"

DRowVector::DRowVector(unsigned columnCount)
        : DMatrix(1, columnCount) {
}

DRowVector::DRowVector(vec_ptr vec_p)
        : DMatrix(vec_p, (unsigned) vec_p->size()) {
}



