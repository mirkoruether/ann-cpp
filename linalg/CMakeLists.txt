add_library(linalg 
mat_arr.h mat_arr.cpp
mat_arr_math_t.h
mat_arr_math.h mat_arr_math.cpp
dev_vector.h)

install(TARGETS linalg DESTINATION lib)
install(FILES mat_arr.h mat_arr_math.h mat_arr_math_t.h DESTINATION include)

if(ANN_CUDA_ACTIVE)
	include_directories(cuda)
    add_subdirectory(cuda)
endif(ANN_CUDA_ACTIVE)
