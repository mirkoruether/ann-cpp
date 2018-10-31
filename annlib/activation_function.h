#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <functional>
#include <cmath>
#include <utility>

using namespace std;

namespace annlib {
    class activation_function {
    public:
        function<double(double)> f;
        function<double(double)> df;

        activation_function(function<double(double)> f, function<double(double)> df) : f(std::move(f)), df(std::move(df)) {}

        double apply(double d) const { return f(d); }

        double apply_derivative(double d) const { return df(d); }
    };

    class logistic_activation_function : public activation_function {
    public:
        explicit logistic_activation_function(double T);
    };
}


#endif
