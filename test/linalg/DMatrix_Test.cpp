//
// Created by Mirko on 04.05.2018.
//

#include "gtest/gtest.h"
#include <DMatrix.h>

using namespace linalg;

TEST(DMatrix_Test, constructor) {
    DMatrix mat1(3, 2);
    EXPECT_EQ(mat1.getRowCount(), 3);
    EXPECT_EQ(mat1.getColumnCount(), 2);
    EXPECT_EQ(mat1.getLength(), 6);
}

TEST(DMatrix_Test, plus) {
    DMatrix mat1(3, 2);
    mat1[0,0] = 1;
    mat1[0,1] = 3;
    mat1[1,0] = 4;
    mat1[1,1] = 7;
    mat1[2,0] = 9;
    mat1[2,1] = 4;

    DMatrix mat2(3, 2);
    mat2[0,0] = 5;
    mat2[0,1] = 6;
    mat2[1,0] = 3;
    mat2[1,1] = 4;
    mat2[2,0] = 8;
    mat2[2,1] = 0;

    DMatrix mat3 = mat1 + mat2;
    EXPECT_EQ(mat3.getRowCount(), 3);
    EXPECT_EQ(mat3.getColumnCount(), 2);

    EXPECT_DOUBLE_EQ((mat3[0,0]), 6);
    EXPECT_DOUBLE_EQ((mat3[0,1]), 9);
    EXPECT_DOUBLE_EQ((mat3[1,0]), 7);
    EXPECT_DOUBLE_EQ((mat3[1,1]), 11);
    EXPECT_DOUBLE_EQ((mat3[2,0]), 17);
    EXPECT_DOUBLE_EQ((mat3[2,1]), 4);
}