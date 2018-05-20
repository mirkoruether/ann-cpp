//
// Created by Mirko on 20.05.2018.
//

#include "DColumnVector.h"


DColumnVector::DColumnVector(unsigned rowCount)
        : DMatrix(rowCount, 1) {
}

DColumnVector::DColumnVector(vec_ptr vec_p)
        : DMatrix(move(vec_p), 1) {
}