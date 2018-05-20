//
// Created by Mirko on 16.05.2018.
//

#include "ActivationFunction.h"

namespace annlib {
    ActivationFunction::ActivationFunction(const function<double(double)> &f, const function<double(double)> &df)
            : f(f), df(df) {}

    double ActivationFunction::apply(double d) const {
        return f(d);
    }

    double ActivationFunction::apply_derivative(double d) const {
        return df(d);
    }
}