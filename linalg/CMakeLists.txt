add_library(linalg DMatrix.h DMatrix.cpp MatrixMulUtil.h MatrixMulUtil.cpp)

install(TARGETS linalg DESTINATION lib)
install(FILES DMatrix.h MatrixMulUtil.h DESTINATION include)