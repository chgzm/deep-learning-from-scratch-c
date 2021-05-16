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

Matrix4d* create_matrix4d_from_stdvec(const std::vector<std::vector<std::vector<std::vector<double>>>>& vec) {
    Matrix4d* M = create_matrix_4d(vec.size(), vec[0].size(), vec[0][0].size(), vec[0][0][0].size());

    for (int i = 0; i < M->sizes[0]; ++i) {
        for (int j = 0; j < M->sizes[1]; ++j) {
            for (int k = 0; k < M->sizes[2]; ++k) {
                for (int l = 0; l < M->sizes[3]; ++l) {
                    M->elements[i][j][k][l] = vec[i][j][k][l];
                }
            }
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

void EXPECT_MATRIX4D_EQ(const std::vector<std::vector<std::vector<std::vector<double>>>>& E, const Matrix4d* M) {
    EXPECT_EQ(E.size(), M->sizes[0]);
    EXPECT_EQ(E[0].size(), M->sizes[1]);
    EXPECT_EQ(E[0][0].size(), M->sizes[2]);
    EXPECT_EQ(E[0][0][0].size(), M->sizes[3]);

    for (int i = 0; i < M->sizes[0]; ++i) {
        for (int j = 0; j < M->sizes[1]; ++j) {
            for (int k = 0; k < M->sizes[2]; ++k) {
                for (int l = 0; l < M->sizes[3]; ++l) {
                    EXPECT_DOUBLE_EQ(E[i][j][k][l], M->elements[i][j][k][l]); 
                }
            }
        }
    }
}

void EXPECT_VECTOR_NEAR(const std::vector<double>& e, const Vector* v) {
    EXPECT_EQ(e.size(), v->size);

    for (int i = 0; i < v->size; ++i) {
        EXPECT_NEAR(e[i], v->elements[i], 10e-8);
    }
}

void EXPECT_MATRIX_NEAR(const std::vector<std::vector<double>>& E, const Matrix* M) {
    EXPECT_EQ(E.size(), M->rows);
    EXPECT_EQ(E[0].size(), M->cols);

    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            EXPECT_NEAR(E[i][j], M->elements[i][j], 10e-8);
        }
    }
}
