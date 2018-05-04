//
// Created by Mirko on 04.05.2018.
//

#include <iostream>
#include <DMatrix.h>
#include <NeuralNetwork.h>

using namespace std;
using namespace linalg;

int main() {
    NeuralNetwork nn();

    DMatrix matrix(2, 2);
    matrix(0, 0) = 1;
    matrix(0, 1) = 2;
    matrix(1, 0) = 3;
    matrix(1, 1) = 4;

    cout << matrix.getLength() << endl;
    int i;
    cin >> i;
}