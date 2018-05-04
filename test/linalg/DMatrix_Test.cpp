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