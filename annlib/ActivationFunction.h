//
// Created by Mirko on 16.05.2018.
//

#ifndef ANN_CPP_ACTIVATIONFUNCTION_H
#define ANN_CPP_ACTIVATIONFUNCTION_H

#include <functional>
#include <cmath>
#include <utility>

using namespace std;

namespace annlib {
    class ActivationFunction {
    public:
        function<double(double)> f;
        function<double(double)> df;

        ActivationFunction(function<double(double)> f, function<double(double)> df) : f(std::move(f)), df(std::move(df)) {}

        double apply(double d) const { return f(d); }

        double apply_derivative(double d) const { return df(d); }
    };

    class LogisticActivationFunction : public ActivationFunction {
    public:
        explicit LogisticActivationFunction(double T);
    };
}


#endif //ANN_CPP_ACTIVATIONFUNCTION_H
