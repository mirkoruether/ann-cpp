cmake_minimum_required(VERSION 3.9)
project(ann_cpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(ann_cpp main.cpp DMatrix.h DMatrix.cpp MatrixMulUtil.cpp MatrixMulUtil.h)