//
// Created by Mirko on 20.05.2018.
//

#ifndef ANN_CPP_DROWVECTOR_H
#define ANN_CPP_DROWVECTOR_H

#include "DMatrix.h"
#include "DColumnVector.h"
#include <vector>
#include <memory>

using namespace linalg;
using namespace std;

using vec_ptr = shared_ptr<vector<double> >;


namespace linalg {
    class DRowVector : public DMatrix {
    public:
        explicit DRowVector(vec_ptr vec_p);

        explicit DRowVector(unsigned columnCount);
    };
}


#endif //ANN_CPP_DROWVECTOR_H
