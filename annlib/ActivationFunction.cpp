//
// Created by Mirko on 14.09.2018.
//

#include "ActivationFunction.h"

using namespace annlib;

namespace annlib {
    LogisticActivationFunction::LogisticActivationFunction(double T)
            : ActivationFunction([this, T](double d) { return 1 / (1 + exp(-d / T)); },
                                 [this, T](double d) { return exp(d / T) / (T * pow(exp(d / T) + 1.0, 2.0)); }) {}

}