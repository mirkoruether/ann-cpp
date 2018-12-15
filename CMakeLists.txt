cmake_minimum_required(VERSION 3.9)
project(ann_cpp)

message("-- >> Compiler Flags")
message("${CMAKE_CXX_FLAGS_RELEASE}")

include(CheckLanguage)
set(TRY_ENABLE_CUDA TRUE)

if (TRY_ENABLE_CUDA)
    message("Try to find CUDA")
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
		find_package(CUDA REQUIRED)
        if (NOT DEFINED CMAKE_CUDA_STANDARD)
            set(CMAKE_CUDA_STANDARD 11)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        endif ()
		set(ANN_CUDA_ACTIVE TRUE)
		#include_directories(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
		#include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/include")
		include_directories("${CMAKE_CUDA_COMPILER}/../../include")

		add_definitions(-DLINALG_CUDA_SUPPORT)
		add_definitions(-DANNLIB_USE_CUDA)

        message("-- >> CUDA enabled <<")
		message("-- >> CUDA libs:  ${CUDA_LIBRARIES}<<")
    endif (CMAKE_CUDA_COMPILER)
else (TRY_ENABLE_CUDA)
	set(ANN_CUDA_ACTIVE FALSE)
    message("-- >> CUDA ignored <<")
endif (TRY_ENABLE_CUDA)

include_directories(linalg)
add_subdirectory(linalg)

include_directories(annlib)
add_subdirectory(annlib)

add_executable(ann_cpp main.cpp mnist.h mnist.cpp)

add_dependencies(ann_cpp annlib linalg)

target_link_libraries(ann_cpp annlib linalg)
if(ANN_CUDA_ACTIVE)
	target_link_libraries(ann_cpp annlib_cudaops linalg_cudaops ${CUDA_LIBRARIES})
endif(ANN_CUDA_ACTIVE)
