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

        ActivationFunction(const function<double(double)> &f, const function<double(double)> &df);

        double apply(double d) const;

        double apply_derivative(double d) const;
    };

    class LogisticActivationFunction : ActivationFunction {
    private:
        double T;

        double f(double x) {
            1 / (1 + exp(-x / T));
        }

        double df(double x) {
            exp(x / T) / (T * pow(exp(x / T) + 1.0, 2.0));
        }

    public:
        explicit LogisticActivationFunction(double T)
                : T(T), ActivationFunction(f, df) {}
    };
}


#endif //ANN_CPP_ACTIVATIONFUNCTION_H
