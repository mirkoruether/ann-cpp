cmake_minimum_required(VERSION 3.9)
project(ann_cpp)

include(CheckLanguage)
check_language(CUDA)

enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 17)

include_directories(linalg)
add_subdirectory(linalg)

include_directories(annlib)
add_subdirectory(annlib)

add_executable(ann_cpp main.cpp mnist.h mnist.cpp)

add_dependencies(ann_cpp annlib linalg)

#add_subdirectory(test)

target_link_libraries(ann_cpp annlib linalg)