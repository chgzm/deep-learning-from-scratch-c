#include "gtest/gtest.h"

extern "C" {
#include <matrix.h>
}

// static const double EPS = 1e-10;

TEST(create_vector, success) {
    const int size = 5;
    Vector* v = create_vector(size);

    EXPECT_EQ(size, v->size);
    for (int i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(0.0, v->elements[i]);
    }

    free_vector(v);
}

TEST(create_vector_initval, success) {
    const int size = 5;
    const int init_val = 0.1;
    Vector* v = create_vector_initval(size, init_val);

    EXPECT_EQ(size, v->size);
    for (int i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(init_val, v->elements[i]);
    }

    free_vector(v);
}

TEST(create_matrix, success) {
    const int rows = 2;
    const int cols = 3;
    Matrix* M = create_matrix(2, 3);

    EXPECT_EQ(rows, M->rows);
    EXPECT_EQ(cols, M->cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            EXPECT_DOUBLE_EQ(0.0, M->elements[i][j]);
        }
    }

    free_matrix(M);
}

TEST(create_matrix_4d, success) {
    const int sizes[] = {2, 3, 4, 5};
    Matrix4d* M = create_matrix_4d(2, 3, 4, 5);

    EXPECT_EQ(sizes[0], M->sizes[0]);
    EXPECT_EQ(sizes[1], M->sizes[1]);
    EXPECT_EQ(sizes[2], M->sizes[2]);
    EXPECT_EQ(sizes[3], M->sizes[3]);
 
    for (int i = 0; i < sizes[0]; ++i) {
        for (int j = 0; j < sizes[1]; ++j) {
            for (int k = 0; k < sizes[2]; ++k) {
                for (int l = 0; l < sizes[3]; ++l) {
                    EXPECT_DOUBLE_EQ(0.0, M->elements[i][j][k][l]);
                }
            }
        }
    }

    free_matrix_4d(M);
}

TEST(init_vector_from_file, success) {
    Vector* v = create_vector(50);

    EXPECT_EQ(0, init_vector_from_file(v, "../../dataset/b1.csv"));

    EXPECT_DOUBLE_EQ(-0.067503154277801513671875, v->elements[0]);
    EXPECT_DOUBLE_EQ(0.10716812312602996826171875, v->elements[49]);

    free_vector(v);
}

TEST(init_matrix_from_file, success) {
    Matrix* M = create_matrix(50, 100);
    EXPECT_EQ(0, init_matrix_from_file(M, "../../dataset/W2.csv"));

    EXPECT_DOUBLE_EQ(-0.10694038867950439453125,  M->elements[0][0]);
    EXPECT_DOUBLE_EQ(0.009080098010599613189697265625, M->elements[49][99]);

    free_matrix(M);
}

TEST(copy_vector, success) {
    Vector* v = create_vector(50);
    EXPECT_EQ(0, init_vector_from_file(v, "../../dataset/b1.csv"));

    Vector* v2 = create_vector(50);
    copy_vector(v2, v);

    EXPECT_EQ(v->size, v2->size);
    EXPECT_DOUBLE_EQ(-0.067503154277801513671875, v2->elements[0]);
    EXPECT_DOUBLE_EQ(0.10716812312602996826171875, v2->elements[49]);

    free_vector(v);
    free_vector(v2);
}

TEST(copy_matrix, success) {
    Matrix* M = create_matrix(50, 100);
    EXPECT_EQ(0, init_matrix_from_file(M, "../../dataset/W2.csv"));

    Matrix* M2 = create_matrix(50, 100);
    copy_matrix(M2, M);

    EXPECT_EQ(M->rows, M2->rows);
    EXPECT_EQ(M->cols, M2->cols);

    EXPECT_DOUBLE_EQ(-0.10694038867950439453125,  M2->elements[0][0]);
    EXPECT_DOUBLE_EQ(0.009080098010599613189697265625, M2->elements[49][99]);

    free_matrix(M);
    free_matrix(M2);
}

TEST(init_matrix_random, success) {
    Matrix* M = create_matrix(100, 100);
    init_matrix_random(M);

    srand(100);
    double sum = 0.0;
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            EXPECT_LT(M->elements[i][j], 4.0);
            EXPECT_LT(-4.0, M->elements[i][j]);
            sum += M->elements[i][j];
        }
    }

    const double mean = sum / (100.0 * 100.0);
    EXPECT_LT(mean, 0.1);
    EXPECT_LT(-0.1, mean);

    free_matrix(M);
}

TEST(init_matrix_rand, success) {
    Matrix* M = create_matrix(100, 100);
    init_matrix_rand(M);

    srand(100);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            EXPECT_LT(M->elements[i][j], 1.0);
            EXPECT_LT(0.0, M->elements[i][j]);
        }
    }

    free_matrix(M);
}

TEST(init_vector_from_array, success) {
    double vals[] = {0, 1, 2, 3, 4};
    Vector* v = create_vector(5);

    init_vector_from_array(v, (double*)(&vals)); 

    for (int i = 0; i < v->size; ++i) {
        EXPECT_DOUBLE_EQ(vals[i], v->elements[i]);
    }

    free_vector(v);
}

TEST(add_vector, success) {
    Vector* a = create_vector(3);
    Vector* b = create_vector(3);

    for (int i = 0; i < 3; ++i) {
        a->elements[i] = b->elements[i] = i + 1;
    }

    Vector* v = add_vector(a, b);
    EXPECT_NE(nullptr, v);

    double ans[] = {2, 4, 6};
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(ans[i], v->elements[i]);
    }

    free_vector(a);
    free_vector(b);
    free_vector(v);
}

TEST(add_vector, error) {
    Vector* a = create_vector(3);
    Vector* b = create_vector(4);

    Vector* v = add_vector(a, b);
    EXPECT_EQ(nullptr, v);

    free_vector(a);
    free_vector(b);
}

TEST(dot_vector_matrix, success) {
    Vector* v = create_vector(3);
    for (int i = 0; i < v->size; ++i) {
        v->elements[i] = i + 1;
    }

    Matrix* M = create_matrix(3, 3);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            M->elements[i][j] = i * 3 + j + 1;
        }
    }

    Vector* u = dot_vector_matrix(v, M);
    EXPECT_EQ(3, u->size);

    double ans[] = {30, 36, 42};
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(ans[i], u->elements[i]);
    }

    free_vector(v);
    free_vector(u);
    free_matrix(M);
}


TEST(dot_vector_matrix, error) {
    Vector* v = create_vector(4);
    Matrix* M = create_matrix(3, 3);
    Vector* u = dot_vector_matrix(v, M);
    EXPECT_EQ(nullptr, u);

    free_vector(v);
    free_matrix(M);
}

TEST(dot_matrix, success) {
    Matrix* M = create_matrix(2, 3);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            M->elements[i][j] = i * 3 + j + 1;
        }
    }

    Matrix* N = create_matrix(3, 4);
    for (int i = 0; i < N->rows; ++i) {
        for (int j = 0; j < N->cols; ++j) {
            N->elements[i][j] = i * 4 + j + 1;
        }
    }

    Matrix* P = dot_matrix(M, N);
    EXPECT_NE(nullptr, P);

    double ans[2][4] = {{38, 44, 50, 56}, {83, 98, 113, 128}};

    for (int i = 0; i < P->rows; ++i) {
        for (int j = 0; j < P->cols; ++j) {
            EXPECT_DOUBLE_EQ(ans[i][j], P->elements[i][j]);
        }
    }

    free_matrix(M);
    free_matrix(N);
    free_matrix(P);
}

TEST(dot_matrix, error) {
    Matrix* M = create_matrix(2, 3);
    Matrix* N = create_matrix(4, 4);
    Matrix* P = dot_matrix(M, N);
    EXPECT_EQ(nullptr, P);

    free_matrix(M);
    free_matrix(N);
}

TEST(product_vector_matrix, success) {
    Vector* v = create_vector(3);
    for (int i = 0; i < v->size; ++i) {
        v->elements[i] = i + 1;
    }

    Matrix* M = create_matrix(3, 3);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            M->elements[i][j] = i * 3 + j + 1;
        }
    }

    Matrix* N = product_vector_matrix(v, M);
    EXPECT_NE(nullptr, N);

    double ans[3][3] = {{1, 4, 9}, {4, 10, 18}, {7, 16, 27}};
    for (int i = 0; i < N->rows; ++i) {
        for (int j = 0; j < N->cols; ++j) {
            EXPECT_DOUBLE_EQ(ans[i][j], N->elements[i][j]);
        }
    }

    free_vector(v);
    free_matrix(M);
    free_matrix(N);
}

TEST(product_vector_matrix, error) {
    Vector* v = create_vector(2);
    Matrix* M = create_matrix(2, 3);
    Matrix* N = product_vector_matrix(v, M);
    EXPECT_EQ(nullptr, N);

    free_vector(v);
    free_matrix(M);
}

TEST(product_matrix, success) {
    Matrix* M = create_matrix(3, 4);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            M->elements[i][j] = i * 4 + j + 1;
        }
    }

    Matrix* N = create_matrix(3, 4);
    for (int i = 0; i < N->rows; ++i) {
        for (int j = 0; j < N->cols; ++j) {
            N->elements[i][j] = i * 4 + j + 1;
        }
    }

    Matrix* A = product_matrix(M, N);
    EXPECT_NE(nullptr, N);

    double ans[3][4] = {{1, 4, 9, 16}, {25, 36, 49, 64}, {81, 100, 121, 144}};
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            EXPECT_DOUBLE_EQ(ans[i][j], A->elements[i][j]);
        }
    }

    free_matrix(M);
    free_matrix(N);
    free_matrix(A);
}

TEST(product_matrix, error) {
    Matrix* M = create_matrix(3, 4);
    Matrix* N = create_matrix(4, 4);
    Matrix* A = product_matrix(M, N);

    EXPECT_EQ(nullptr, A);

    free_matrix(M);
    free_matrix(N);
}

TEST(product_vector, success) {
    Vector* v = create_vector(3);
    for (int i = 0; i < v->size; ++i) {
        v->elements[i] = i + 1;
    }

    Vector* u = create_vector(3);
    for (int i = 0; i < u->size; ++i) {
        u->elements[i] = i + 1;
    }

    Vector* w = product_vector(v, u);
    EXPECT_NE(nullptr, w);

    double ans[3] = {1, 4, 9};
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(ans[i], w->elements[i]);
    }

    free_vector(v);
    free_vector(u);
    free_vector(w);
}

TEST(product_vector, error) {
    Vector* v = create_vector(3);
    Vector* u = create_vector(4);

    Vector* w = product_vector(v, u);
    EXPECT_EQ(nullptr, w);

    free_vector(v);
    free_vector(u);
}

TEST(matrix_col_mean, success) {
    Matrix* M = create_matrix(3, 4);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            M->elements[i][j] = i * 4 + j + 1;
        }
    }

    Vector* v = matrix_col_mean(M);
    double ans[4] = {5, 6, 7, 8};
    
    for (int i = 0; i < 4; ++i) {
        EXPECT_DOUBLE_EQ(ans[i], v->elements[i]);
    }

    free_matrix(M);
    free_vector(v);
}

TEST(scalar_matrix, success) {
    Matrix* M = create_matrix(3, 3);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            M->elements[i][j] = i * 3 + j + 1;
        }
    }

    scalar_matrix(M, 3);
    double ans[3][3] = {{3, 6, 9}, {12, 15, 18}, {21, 24, 27}};
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            EXPECT_DOUBLE_EQ(ans[i][j], M->elements[i][j]);
        }
    }

    free_matrix(M);
}

TEST(scalar_vector, success) {
    Vector* v = create_vector(3);
    for (int i = 0; i < v->size; ++i) {
        v->elements[i] = i + 1;
    }

    scalar_vector(v, 3);
    double ans[3] = {3, 6, 9};
    for (int i = 0; i < v->size; ++i) {
        EXPECT_DOUBLE_EQ(ans[i], v->elements[i]);
    }

    free_vector(v);
}

TEST(transpose, success) {
    Matrix* M = create_matrix(3, 3);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            M->elements[i][j] = i * 3 + j + 1;
        }
    }

    Matrix* M_T = transpose(M);

    double ans[3][3] = {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}};
    for (int i = 0; i < M_T->rows; ++i) {
        for (int j = 0; j < M_T->cols; ++j) {
            EXPECT_DOUBLE_EQ(ans[i][j], M_T->elements[i][j]);
        }
    }

    free_matrix(M);
    free_matrix(M_T);
}

