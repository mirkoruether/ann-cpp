cmake_minimum_required(VERSION 3.9)
project(ann_cpp)

set(CMAKE_CXX_STANDARD 17)

add_library(ann_cpp DMatrix.h DMatrix.cpp MatrixMulUtil.cpp MatrixMulUtil.h NetworkLayer.cpp NetworkLayer.h NeuralNetwork.cpp NeuralNetwork.h)