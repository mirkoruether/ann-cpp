add_library(linalg 
mat_arr.h mat_arr.cpp
mat_arr_math.h mat_arr_math.cpp
dmatrix.h dmatrix.cpp 
matrixmulutil.h matrixmulutil.cpp)

install(TARGETS linalg DESTINATION lib)
install(FILES dmatrix.h matrixmulutil.h mat_arr.h Mat_arr_math.h DESTINATION include)