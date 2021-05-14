#include "utest_util.h"

#include "gtest/gtest.h"


Vector* create_vector_from_stdvec(const std::vector<double>& vec) {
    Vector* v = create_vector(vec.size());

    for (int i = 0; i < v->size; ++i) {
        v->elements[i] = vec[i];
    }

    return v;
}

Matrix* create_matrix_from_stdvec(const std::vector<std::vector<double>>& vec) {
    Matrix* M = create_matrix(vec.size(), vec[0].size());

    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            M->elements[i][j] = vec[i][j];
        }
    }

    return M;
}

void EXPECT_VECTOR_EQ(const std::vector<double>& e, const Vector* v) {
    EXPECT_EQ(e.size(), v->size);

    for (int i = 0; i < v->size; ++i) {
        EXPECT_DOUBLE_EQ(e[i], v->elements[i]);
    }
}

void EXPECT_MATRIX_EQ(const std::vector<std::vector<double>>& E, const Matrix* M) {
    EXPECT_EQ(E.size(), M->rows);
    EXPECT_EQ(E[0].size(), M->cols);

    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            EXPECT_DOUBLE_EQ(E[i][j], M->elements[i][j]);
        }
    }
}
