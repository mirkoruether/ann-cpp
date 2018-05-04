cmake_minimum_required(VERSION 3.9)
project(ann_cpp)

set(CMAKE_CXX_STANDARD 17)

include_directories(linalg)
add_subdirectory(linalg)

include_directories(annlib)
add_subdirectory(annlib)

add_executable(ann_cpp main.cpp)

add_subdirectory(test)

target_link_libraries(ann_cpp annlib linalg)