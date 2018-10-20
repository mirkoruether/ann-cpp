add_library(linalg 
dmatrix.h dmatrix.cpp 
matrixmulutil.h matrixmulutil.cpp)

install(TARGETS linalg DESTINATION lib)
install(FILES dmatrix.h matrixmulutil.h DESTINATION include)