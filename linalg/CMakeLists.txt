add_library(linalg 
mat_arr.h mat_arr.cpp
mat_arr_math.h mat_arr_math.cpp
general_util.h)

install(TARGETS linalg DESTINATION lib)
install(FILES mat_arr.h mat_arr_math.h general_util.h DESTINATION include)