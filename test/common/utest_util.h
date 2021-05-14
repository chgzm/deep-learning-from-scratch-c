#ifndef UTEST_UTIL_H
#define UTEST_UTIL_H

#include <vector>
extern "C" {
#include <matrix.h>
}

Vector* create_vector_from_stdvec(const std::vector<double>& vec);
Matrix* create_matrix_from_stdvec(const std::vector<std::vector<double>>& vec);

void EXPECT_VECTOR_EQ(const std::vector<double>& e, const Vector* v);
void EXPECT_MATRIX_EQ(const std::vector<std::vector<double>>& E, const Matrix* M);

void EXPECT_VECTOR_NEAR(const std::vector<double>& e, const Vector* v);
void EXPECT_MATRIX_NEAR(const std::vector<std::vector<double>>& E, const Matrix* M);

#endif
