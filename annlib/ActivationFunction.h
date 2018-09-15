//
// Created by Mirko on 16.05.2018.
//

#ifndef ANN_CPP_ACTIVATIONFUNCTION_H
#define ANN_CPP_ACTIVATIONFUNCTION_H

#include <functional>
#include <cmath>

using namespace std;

namespace annlib {
    class ActivationFunction {
    public:
        const function<double(double)> &f;
        const function<double(double)> &df;

        ActivationFunction(const function<double(double)> &f, const function<double(double)> &df) : f(f), df(df) {}

        double apply(double d) const { return f(d); }

        double apply_derivative(double d) const { return df(d); }
    };

    class LogisticActivationFunction : public ActivationFunction {
    public:
        explicit LogisticActivationFunction(double T);
    };
}


#endif //ANN_CPP_ACTIVATIONFUNCTION_H
