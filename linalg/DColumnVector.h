//
// Created by Mirko on 20.05.2018.
//

#ifndef ANN_CPP_DCOLUMNVECTOR_H
#define ANN_CPP_DCOLUMNVECTOR_H


#include "DMatrix.h"
#include "DRowVector.h"
#include <vector>
#include <memory>

using namespace linalg;
using namespace std;

using vec_ptr = shared_ptr<vector<double> >;


namespace linalg {
    class DColumnVector : public DMatrix {
    public:
        explicit DColumnVector(vec_ptr vec_p);

        explicit DColumnVector(unsigned rowCount);
    };
}


#endif //ANN_CPP_DCOLUMNVECTOR_H
