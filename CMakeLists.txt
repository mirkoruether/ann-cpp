cmake_minimum_required(VERSION 3.9)
project(ann_cpp)

set(CMAKE_CXX_STANDARD 17)

include(CheckLanguage)
option(TRY_ENABLE_CUDA "Search for CUDA?" ON)

if (TRY_ENABLE_CUDA)
    message("Try to find CUDA")
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        if (NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_STANDARD 11)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif ()
        message("-- >> CUDA enabled <<")
    endif (CMAKE_CUDA_COMPILER)
else (TRY_ENABLE_CUDA)
    message("-- >> CUDA ignored <<")
endif (TRY_ENABLE_CUDA)

include_directories(linalg)
add_subdirectory(linalg)

include_directories(annlib)
add_subdirectory(annlib)

add_executable(ann_cpp main.cpp mnist.h mnist.cpp)

add_dependencies(ann_cpp annlib linalg)

target_link_libraries(ann_cpp annlib linalg)