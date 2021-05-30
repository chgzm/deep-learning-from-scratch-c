#include "gtest/gtest.h"

#include "utest_util.h"

extern "C" {
#include <function.h>
}

TEST(vector_sigmoid, success) {
    Vector* v = create_vector(3);
    v->elements[0] = -1;
    v->elements[1] = 1;
    v->elements[2] = 2;

    Vector* u = vector_sigmoid(v);
    EXPECT_NE(nullptr, u);

    EXPECT_DOUBLE_EQ(sigmoid(-1), u->elements[0]);
    EXPECT_DOUBLE_EQ(sigmoid(1),  u->elements[1]);
    EXPECT_DOUBLE_EQ(sigmoid(2),  u->elements[2]);

    free_vector(v);
    free_vector(u);
}

TEST(matrix_sigmoid, success) {
    Matrix* M = create_matrix(2, 3);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            M->elements[i][j] = i * 3 + j - 2;
        }
    }

    Matrix* N = matrix_sigmoid(M);
    EXPECT_NE(nullptr, N);

    for (int i = 0; i < N->rows; ++i) {
        for (int j = 0; j < N->cols; ++j) {
            EXPECT_DOUBLE_EQ(sigmoid(M->elements[i][j]), N->elements[i][j]);
        }
     }
    
     free_matrix(M);
     free_matrix(N);
}

TEST(sigmoid_grad, success) {
    Matrix* M = create_matrix(2, 3);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            M->elements[i][j] = i * 3 + j - 2;
        }
    }

    Matrix* N = sigmoid_grad(M);
 
    double ans[2][3] = {{0.10499359, 0.19661193, 0.25}, {0.19661193, 0.10499359, 0.04517666}};

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(ans[i][j], N->elements[i][j], 10e-8);
        }
    }
   
    free_matrix(M);
    free_matrix(N);
}

TEST(vector_softmax, success) {
    Vector* v = create_vector(3);
    v->elements[0] = 1;
    v->elements[1] = 2;
    v->elements[2] = 3;

    Vector* u = vector_softmax(v);
    EXPECT_NEAR(0.09003057, u->elements[0], 10e-8);
    EXPECT_NEAR(0.24472847, u->elements[1], 10e-8);
    EXPECT_NEAR(0.66524096, u->elements[2], 10e-8);

    free_vector(v);
    free_vector(u);
}

TEST(matrix_softmax, success) {
    Matrix* M = create_matrix(2, 3);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            M->elements[i][j] = i * 3 + j - 3;
        }
    }

    Matrix* N = matrix_softmax(M);
    double ans[2][3] = {{0.09003057, 0.24472847, 0.66524096}, {0.09003057, 0.24472847, 0.66524096}};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_NEAR(ans[i][j], N->elements[i][j], 10e-8);
        }
    }
    
    free_matrix(M);
    free_matrix(N);
}

TEST(argmax, success) {
    double v[] = {2, 1, 5, 3, -1};
    EXPECT_EQ(2, argmax(v, 5));
}

TEST(vector_argmax, success) {
    Vector* v = create_vector(5);
    v->elements[0] = 2;
    v->elements[1] = 1;
    v->elements[2] = 5;
    v->elements[3] = 3;

    EXPECT_EQ(2, vector_argmax(v));

    free_vector(v);
}

TEST(matrix_argmax_row, success) {
    Matrix* M = create_matrix_from_stdvec({{1, 2, 3}, {4, 6, 5}, {9, 8, 7}});
    int* arg_max = matrix_argmax_row(M);

    int ans[] = {2, 1, 0};
    for (int i = 0; i < M->rows; ++i) {
        EXPECT_EQ(ans[i], arg_max[i]);
    }

    free_matrix(M);
    free(arg_max);
}




