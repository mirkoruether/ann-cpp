//
// Created by Mirko on 16.05.2018.
//

#ifndef ANN_CPP_ACTIVATIONFUNCTION_H
#define ANN_CPP_ACTIVATIONFUNCTION_H

#include <functional>

using namespace std;

namespace annlib {
    class ActivationFunction {
    public:
        const function<double(double)> &f;
        const function<double(double)> &df;

        ActivationFunction(const function<double(double)> &f, const function<double(double)> &df);

        double apply(double d) const;

        double apply_derivative(double d) const;
    };
}


#endif //ANN_CPP_ACTIVATIONFUNCTION_H
