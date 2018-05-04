include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(linalgtest DMatrix_Test.cpp)

target_link_libraries(linalgtest gtest gtest_main)
target_link_libraries(linalgtest annlib linalg)