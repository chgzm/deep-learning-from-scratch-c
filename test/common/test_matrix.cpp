#include "gtest/gtest.h"

extern "C" {
#include <matrix.h>
#include <mnist.h>
}

#include "utest_util.h"

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

    srand(100);
    init_matrix_random(M);
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

TEST(init_matrix_4d_random, success) {
    Matrix4d* M = create_matrix_4d(10, 10, 10, 10);

    srand(100);
    init_matrix_4d_random(M);
    double sum = 0.0;
    for (int i = 0; i < M->sizes[0]; ++i) {
        for (int j = 0; j < M->sizes[1]; ++j) {
            for (int k = 0; k < M->sizes[2]; ++k) {
                for (int l = 0; l < M->sizes[3]; ++l) {
                    EXPECT_LT(M->elements[i][j][k][l], 5.0);
                    EXPECT_LT(-5.0, M->elements[i][j][k][l]);
                    sum += M->elements[i][j][k][l];
                }
            }
        }
    }

    const double mean = sum / (10 * 10 * 10 * 10);
    EXPECT_LT(mean, 0.1);
    EXPECT_LT(-0.1, mean);

    free_matrix_4d(M);
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
    Vector* a = create_vector_from_stdvec({1, 2, 3});
    Vector* b = create_vector_from_stdvec({1, 2, 3});

    Vector* v = add_vector(a, b);
    EXPECT_NE(nullptr, v);
    EXPECT_VECTOR_EQ({2, 4, 6}, v);

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
    Vector* v = create_vector_from_stdvec({1, 2, 3});
    Matrix* M = create_matrix_from_stdvec({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

    Vector* u = dot_vector_matrix(v, M);
    EXPECT_VECTOR_EQ({30, 36, 42}, u);

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
    Matrix* M = create_matrix_from_stdvec({{1, 2, 3}, {4, 5, 6}});
    Matrix* N = create_matrix_from_stdvec({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});

    Matrix* P = dot_matrix(M, N);
    EXPECT_NE(nullptr, P);
    EXPECT_MATRIX_EQ({{38, 44, 50, 56}, {83, 98, 113, 128}}, P);

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
    Vector* v = create_vector_from_stdvec({1, 2, 3});
    Matrix* M = create_matrix_from_stdvec({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

    Matrix* N = product_vector_matrix(v, M);
    EXPECT_NE(nullptr, N);
    EXPECT_MATRIX_EQ({{1, 4, 9}, {4, 10, 18}, {7, 16, 27}}, N);

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
    Matrix* M = create_matrix_from_stdvec({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
    Matrix* N = create_matrix_from_stdvec({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});

    Matrix* A = product_matrix(M, N);
    EXPECT_NE(nullptr, N);
    EXPECT_MATRIX_EQ({{1, 4, 9, 16}, {25, 36, 49, 64}, {81, 100, 121, 144}}, A);

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
    Vector* v = create_vector_from_stdvec({1, 2, 3});
    Vector* u = create_vector_from_stdvec({1, 2, 3});

    Vector* w = product_vector(v, u);
    EXPECT_NE(nullptr, w);
    EXPECT_VECTOR_EQ({1, 4, 9}, w);

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
    Matrix* M = create_matrix_from_stdvec({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});

    Vector* v = matrix_col_mean(M);
   
    EXPECT_VECTOR_EQ({5, 6, 7, 8}, v);

    free_matrix(M);
    free_vector(v);
}

TEST(matrix_col_sum, success) {
    Matrix* M = create_matrix_from_stdvec({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}});
    
    Vector* v = matrix_col_sum(M);
    EXPECT_VECTOR_EQ({15, 18, 21, 24}, v);

    free_matrix(M);
    free_vector(v);
}

TEST(matrix_row_max, success) {
    Matrix* M = create_matrix_from_stdvec({{4, 2, 3, 1}, {5, 8, 7, 6}, {9, 10, 11, 12}});
    
    Vector* v = matrix_row_max(M);
    EXPECT_VECTOR_EQ({4, 8, 12}, v);

    free_matrix(M);
    free_vector(v);
}

TEST(scalar_matrix, success) {
    Matrix* M = create_matrix_from_stdvec({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

    scalar_matrix(M, 3);
    EXPECT_MATRIX_EQ({{3, 6, 9}, {12, 15, 18}, {21, 24, 27}}, M);

    free_matrix(M);
}

TEST(scalar_matrix_4d, success) {
     Matrix4d* M = create_matrix4d_from_stdvec({
       {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
       }, 
       {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
       }
    });

    scalar_matrix_4d(M, 3.0);

    std::vector<std::vector<std::vector<std::vector<double>>>> ans = {  
       {
            {{ 3,  6}, { 9, 12}}, 
            {{15, 18}, {21, 24}},
            {{27, 30}, {33, 36}},
            {{39, 42}, {45, 48}}
       }, 
       {
            {{ 3,  6}, { 9, 12}}, 
            {{15, 18}, {21, 24}},
            {{27, 30}, {33, 36}},
            {{39, 42}, {45, 48}}
       } 
    };
    EXPECT_MATRIX4D_EQ(ans, M);

    free_matrix_4d(M);
}

TEST(scalar_vector, success) {
    Vector* v = create_vector_from_stdvec({1, 2, 3});

    scalar_vector(v, 3);
    EXPECT_VECTOR_EQ({3, 6, 9}, v);

    free_vector(v);
}

TEST(transpose, success) {
    Matrix* M = create_matrix_from_stdvec({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

    Matrix* M_T = transpose(M);
    EXPECT_MATRIX_EQ({{1, 4, 7}, {2, 5, 8}, {3, 6, 9}}, M_T);

    free_matrix(M);
    free_matrix(M_T);
}

TEST(matrix_4d_transpose, success) {
     Matrix4d* M = create_matrix4d_from_stdvec({
       {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
       }, 
       {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
       }
    });

    Matrix4d* MT = matrix_4d_transpose(M, 0, 3, 1, 2);   

    std::vector<std::vector<std::vector<std::vector<double>>>> ans = {  
        {
            {
                { 1,  3},
                { 5,  7},
                { 9, 11},
                {13, 15}
            },

            {
                { 2,  4},
                { 6,  8},
                {10, 12},
                {14, 16}
            }
        },

        {
            {
                { 1,  3},
                { 5,  7},
                { 9, 11},
                {13, 15}
            },

            {
                { 2,  4},
                { 6,  8},
                {10, 12},
                {14, 16}
            }
        }
    };

    EXPECT_MATRIX4D_EQ(ans, MT);

    free_matrix_4d(M);
    free_matrix_4d(MT);
}

TEST(matrix_4d_transpose, success2) {
     Matrix4d* M = create_matrix4d_from_stdvec({
       {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
       }, 
       {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
       }
    });

    Matrix4d* MT = matrix_4d_transpose(M, 0, 3, 2, 1);   

    std::vector<std::vector<std::vector<std::vector<double>>>> ans = {  
        {
            {
                {1, 5,  9, 13},
                {3, 7, 11, 15}
            },
            {
                {2, 6, 10, 14},
                {4, 8, 12, 16}
            }
        },
        {
            {
                {1, 5,  9, 13},
                {3, 7, 11, 15}
            },
            {
                {2, 6, 10, 14},
                {4, 8, 12, 16}
            }
        }
    };

    EXPECT_MATRIX4D_EQ(ans, MT);

    free_matrix_4d(M);
    free_matrix_4d(MT);
}

TEST(vector_reshape_to_4d, success) {
    Vector* v = create_vector_from_stdvec({
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 
        7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12
    });

    Matrix4d* M = vector_reshape_to_4d(v, 4, 3, 2, 2); 

    const std::vector<std::vector<std::vector<std::vector<double>>>> ans = {  
        {
            {{1, 1}, {1, 1}}, 
            {{2, 2}, {2, 2}}, 
            {{3, 3}, {3, 3}}, 
        }, 
        {
            {{4, 4}, {4, 4}}, 
            {{5, 5}, {5, 5}},
            {{6, 6}, {6, 6}}
        }, 
        {
            {{7, 7}, {7, 7}}, 
            {{8, 8}, {8, 8}},
            {{9, 9}, {9, 9}}
        }, 
        {
            {{10, 10}, {10, 10}}, 
            {{11, 11}, {11, 11}},
            {{12, 12}, {12, 12}}
        }
    };
    EXPECT_MATRIX4D_EQ(ans, M);

    free_vector(v);
    free_matrix_4d(M);
}   

TEST(matrix_reshape, success) {
    Matrix* M = create_matrix_from_stdvec(
    {
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16},
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16},
    });

    Matrix* R = matrix_reshape(M, 4, 8);

    std::vector<std::vector<double>> ans = {
        { 1,  2,  3,  4, 5,  6,  7,  8},
        { 9, 10, 11, 12, 13, 14, 15, 16},
        { 1,  2,  3,  4, 5,  6,  7,  8},
        { 9, 10, 11, 12, 13, 14, 15, 16},
    };
    EXPECT_MATRIX_EQ(ans, R);

    free_matrix(M);
    free_matrix(R);
}

TEST(matrix_reshape, success2) {
    Matrix* M = create_matrix_from_stdvec(
    {
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16},
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16},
    });

    Matrix* R = matrix_reshape(M, -1, 8);

    std::vector<std::vector<double>> ans = {
        { 1,  2,  3,  4, 5,  6,  7,  8},
        { 9, 10, 11, 12, 13, 14, 15, 16},
        { 1,  2,  3,  4, 5,  6,  7,  8},
        { 9, 10, 11, 12, 13, 14, 15, 16},
    };
    EXPECT_MATRIX_EQ(ans, R);

    free_matrix(M);
    free_matrix(R);
}

TEST(matrix_reshape, success3) {
    Matrix* M = create_matrix_from_stdvec(
    {
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16},
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16},
    });

    Matrix* R = matrix_reshape(M, 4, -1);

    std::vector<std::vector<double>> ans = {
        { 1,  2,  3,  4, 5,  6,  7,  8},
        { 9, 10, 11, 12, 13, 14, 15, 16},
        { 1,  2,  3,  4, 5,  6,  7,  8},
        { 9, 10, 11, 12, 13, 14, 15, 16},
    };
    EXPECT_MATRIX_EQ(ans, R);

    free_matrix(M);
    free_matrix(R);
}

TEST(matrix_reshape_to_2d, success) {
    Matrix4d* M = create_matrix4d_from_stdvec({
       {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
       }, 
       {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
       }
    });

    Matrix* N = matrix_reshape_to_2d(M, 8, 4);

    std::vector<std::vector<double>> ans = {
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16},
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16},
    };

    EXPECT_MATRIX_EQ(ans, N);

    free_matrix_4d(M);
    free_matrix(N);
}

TEST(matrix_reshape_to_2d, success2) {
    Matrix4d* M = create_matrix4d_from_stdvec({
       {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
       }, 
       {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
       }
    });

    Matrix* N = matrix_reshape_to_2d(M, 8, -1);

    std::vector<std::vector<double>> ans = {
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16},
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16},
    };

    EXPECT_MATRIX_EQ(ans, N);

    free_matrix_4d(M);
    free_matrix(N);
}

TEST(matrix_reshape_to_2d, success3) {
    Matrix4d* M = create_matrix4d_from_stdvec({
       {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
       }, 
       {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
       }
    });

    Matrix* N = matrix_reshape_to_2d(M, -1, 4);

    std::vector<std::vector<double>> ans = {
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16},
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16},
    };

    EXPECT_MATRIX_EQ(ans, N);

    free_matrix_4d(M);
    free_matrix(N);
}


TEST(matrix_reshape_to_4d, success) {
    Matrix* M = create_matrix_from_stdvec(
    {
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16},
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16},
    });

    Matrix4d* N = matrix_reshape_to_4d(M, 2, 4, 2, 2);

    std::vector<std::vector<std::vector<std::vector<double>>>> ans = {  
       {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
       }, 
       {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
       }
    };

    EXPECT_MATRIX4D_EQ(ans, N);

    free_matrix(M);
    free_matrix_4d(N);
}

TEST(matrix_reshape_to_4d, success2) {
    Matrix* M = create_matrix_from_stdvec(
    {
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16},
        { 1,  2,  3,  4},
        { 5,  6,  7,  8},
        { 9, 10, 11, 12},
        {13, 14, 15, 16},
    });

    Matrix4d* N = matrix_reshape_to_4d(M, 2, 4, 2, -1);

    std::vector<std::vector<std::vector<std::vector<double>>>> ans = {  
       {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
       }, 
       {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
       }
    };

    EXPECT_MATRIX4D_EQ(ans, N);

    free_matrix(M);
    free_matrix_4d(N);
}

TEST(matrix_4d_flatten, success) {
    Matrix4d* M = create_matrix4d_from_stdvec({
        {
             {{1,   2}, {3,   4}}, 
             {{5,   6}, {7,   8}},
             {{9,  10}, {11, 12}},
             {{13, 14}, {15, 16}}
        }, 
        {
             {{1,   2}, {3,   4}}, 
             {{5,   6}, {7,   8}},
             {{9,  10}, {11, 12}},
             {{13, 14}, {15, 16}}
        }
    });

    Vector* v = matrix_4d_flatten(M);

    const std::vector<double> ans = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    };
    EXPECT_VECTOR_EQ(ans, v);

    free_vector(v);
    free_matrix_4d(M);  
}

TEST(matrix_sum, success) {
    Matrix* M = create_matrix_from_stdvec({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

    const double sum =  matrix_sum(M);
    EXPECT_DOUBLE_EQ(sum, 45);

    free_matrix(M);
}

TEST(vector_div_vector, success) {
    Vector* v = create_vector_from_stdvec({5, 10, 15});
    Vector* u = create_vector_from_stdvec({5, 5, 5});
    
    Vector* w = vector_div_vector(v, u);
    EXPECT_NE(nullptr, w);
    EXPECT_VECTOR_EQ({1, 2, 3}, w);

    free_vector(v);
    free_vector(u);
    free_vector(w);
}

TEST(vector_div_vector, error) {
    Vector* v = create_vector(3);
    Vector* u = create_vector(4);
    Vector* w = vector_div_vector(v, u);
    EXPECT_EQ(nullptr, w);

    free_vector(v);
    free_vector(u);
}

TEST(matrix_add_vector, success) {
    Matrix* M = create_matrix_from_stdvec({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    Vector* v = create_vector_from_stdvec({1, 10, 100});

    Matrix* N = matrix_add_vector(M, v);
    EXPECT_NE(nullptr, N);
    EXPECT_MATRIX_EQ({{2, 12, 103}, {5, 15, 106}, {8, 18, 109}}, N);

    free_matrix(M);
    free_vector(v);
    free_matrix(N);
}

TEST(matrix_add_vector, error) {
    Matrix* M = create_matrix(3, 3);
    Vector* v = create_vector(4);
    Matrix* N = matrix_add_vector(M, v);
    EXPECT_EQ(nullptr, N);

    free_matrix(M);
    free_vector(v);
}

TEST(matrix_add_matrix, success) {
    Matrix* M = create_matrix_from_stdvec({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    Matrix* N = create_matrix_from_stdvec({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

    Matrix* A = matrix_add_matrix(M, N);
    EXPECT_NE(nullptr, A);

    EXPECT_MATRIX_EQ({{2, 4, 6}, {8, 10, 12}, {14, 16, 18}}, A);
   
    free_matrix(M);
    free_matrix(N);
    free_matrix(A);
}

TEST(matrix_add_matrix, error) {
    Matrix* M = create_matrix(3, 4);
    Matrix* N = create_matrix(3, 3);
    Matrix* A = matrix_add_matrix(M, N);
    EXPECT_EQ(nullptr, A);

    free_matrix(M);
    free_matrix(N);
}

TEST(matrix_sub_vector, success) {
    Matrix* M = create_matrix_from_stdvec({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    Vector* v = create_vector_from_stdvec({1, 10, 100});

    Matrix* N = matrix_sub_vector(M, v);
    EXPECT_NE(nullptr, N);

    EXPECT_MATRIX_EQ({{0, -8, -97}, {3, -5, -94}, {6, -2, -91}}, N);

    free_matrix(M);
    free_vector(v);
    free_matrix(N);
}

TEST(matrix_sub_vector, error) {
    Matrix* M = create_matrix(3, 3);
    Vector* v = create_vector(4);
    Matrix* N = matrix_sub_vector(M, v);
    EXPECT_EQ(nullptr, N);

    free_matrix(M);
    free_vector(v);
}

TEST(matrix_div_vector, success) {
    Matrix* M = create_matrix_from_stdvec({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    Vector* v = create_vector_from_stdvec({1, 10, 100});

    Matrix* N = matrix_div_vector(M, v);
    EXPECT_NE(nullptr, N);

    EXPECT_MATRIX_EQ({{1, 0.2, 0.03}, {4, 0.5, 0.06}, {7, 0.8, 0.09}}, N);

    free_matrix(M);
    free_vector(v);
    free_matrix(N);
}

TEST(matrix_div_vector, error) {
    Matrix* M = create_matrix(3, 3);
    Vector* v = create_vector(4);
    Matrix* N = matrix_div_vector(M, v);
    EXPECT_EQ(nullptr, N);

    free_matrix(M);
    free_vector(v);
}

TEST(vector_add_scalar, success) {
    Vector* v = create_vector_from_stdvec({1, 10, 100});

    Vector* u = vector_add_scalar(v, 1000);
    EXPECT_NE(nullptr, u);

    EXPECT_VECTOR_EQ({1001, 1010, 1100}, u);

    free_vector(v);
    free_vector(u);
}

TEST(pow_matrix, success) {
    Matrix* M = create_matrix_from_stdvec({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

    Matrix* N = pow_matrix(M, 2);
    EXPECT_MATRIX_EQ({{1, 4, 9}, {16, 25, 36}, {49, 64, 81}}, N);

    free_matrix(M);
    free_matrix(N);
}

TEST(sqrt_vector, success) {
    Vector* v = create_vector_from_stdvec({1, 4, 9});

    Vector* u = sqrt_vector(v);

    EXPECT_VECTOR_EQ({1, 2, 3}, u);

    free_vector(v);
    free_vector(u);
}

TEST(im2col, success) {
    Matrix4d* M = create_matrix4d_from_stdvec(
    {
        {
            {{1, 1}, {1, 1}}, //R
            {{2, 2}, {2, 2}}, //G
            {{3, 3}, {3, 3}}, //B
        }, 
        {
            {{4, 4}, {4, 4}}, 
            {{5, 5}, {5, 5}},
            {{6, 6}, {6, 6}}
        }, 
        {
            {{7, 7}, {7, 7}}, 
            {{8, 8}, {8, 8}},
            {{9, 9}, {9, 9}}
        }, 
        {
            {{10, 10}, {10, 10}}, 
            {{11, 11}, {11, 11}},
            {{12, 12}, {12, 12}}
        }
    });

    Matrix* N = im2col(M, 1, 1, 1, 0);
    std::vector<std::vector<double>> ans = {
        { 1,  2,  3},
        { 1,  2,  3},
        { 1,  2,  3},
        { 1,  2,  3},
        { 4,  5,  6},
        { 4,  5,  6},
        { 4,  5,  6},
        { 4,  5,  6},
        { 7,  8,  9},
        { 7,  8,  9},
        { 7,  8,  9},
        { 7,  8,  9},
        {10, 11, 12},
        {10, 11, 12},
        {10, 11, 12},
        {10, 11, 12}
    };

    EXPECT_MATRIX_EQ(ans, N); 

    free_matrix_4d(M);
    free_matrix(N);
}

TEST(im2col, success2) {
    Matrix4d* M = create_matrix4d_from_stdvec(
    {
        {
            {{1, 1}, {1, 1}}, //R
            {{2, 2}, {2, 2}}, //G
            {{3, 3}, {3, 3}}, //B
        }, 
        {
            {{4, 4}, {4, 4}}, 
            {{5, 5}, {5, 5}},
            {{6, 6}, {6, 6}}
        }, 
        {
            {{7, 7}, {7, 7}}, 
            {{8, 8}, {8, 8}},
            {{9, 9}, {9, 9}}
        }, 
        {
            {{10, 10}, {10, 10}}, 
            {{11, 11}, {11, 11}},
            {{12, 12}, {12, 12}}
        }
    });

    Matrix* N = im2col(M, 2, 2, 1, 0);
    std::vector<std::vector<double>> ans = {
       { 1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3},
       { 4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6},
       { 7,  7,  7,  7,  8,  8,  8,  8,  9,  9,  9,  9},
       {10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12}
    };

    EXPECT_MATRIX_EQ(ans, N); 

    free_matrix_4d(M);
    free_matrix(N);
}

TEST(im2col, success3) {
    Matrix4d* M = create_matrix4d_from_stdvec(
    {
        {
            {{1, 1}, {1, 1}}, 
            {{2, 2}, {2, 2}}, 
            {{3, 3}, {3, 3}}, 
        }, 
        {
            {{4, 4}, {4, 4}}, 
            {{5, 5}, {5, 5}},
            {{6, 6}, {6, 6}}
        }, 
        {
            {{7, 7}, {7, 7}}, 
            {{8, 8}, {8, 8}},
            {{9, 9}, {9, 9}}
        }, 
        {
            {{10, 10}, {10, 10}}, 
            {{11, 11}, {11, 11}},
            {{12, 12}, {12, 12}}
        }
    });

    Matrix* N = im2col(M, 1, 1, 2, 0);
    std::vector<std::vector<double>> ans = {
        { 1,  2,  3},
        { 4,  5,  6},
        { 7,  8,  9},
        {10, 11, 12},
    };

    EXPECT_MATRIX_EQ(ans, N); 

    free_matrix_4d(M);
    free_matrix(N);
}

TEST(im2col, success4) {
    Matrix4d* M = create_matrix4d_from_stdvec(
    {
        {
            {{1, 1}, {1, 1}},
            {{2, 2}, {2, 2}},
            {{3, 3}, {3, 3}},
        }, 
        {
            {{4, 4}, {4, 4}}, 
            {{5, 5}, {5, 5}},
            {{6, 6}, {6, 6}}
        }, 
        {
            {{7, 7}, {7, 7}}, 
            {{8, 8}, {8, 8}},
            {{9, 9}, {9, 9}}
        }, 
        {
            {{10, 10}, {10, 10}}, 
            {{11, 11}, {11, 11}},
            {{12, 12}, {12, 12}}
        }
    });

    Matrix* N = im2col(M, 2, 2, 2, 0);
    std::vector<std::vector<double>> ans = {
       { 1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3},
       { 4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6},
       { 7,  7,  7,  7,  8,  8,  8,  8,  9,  9,  9,  9},
       {10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12}
    };

    EXPECT_MATRIX_EQ(ans, N); 

    free_matrix_4d(M);
    free_matrix(N);
}

TEST(im2col, success5) {
    Matrix4d* M = create_matrix4d_from_stdvec(
    {
        {
            {{1, 1}, {1, 1}},
            {{2, 2}, {2, 2}},
            {{3, 3}, {3, 3}},
        }, 
        {
            {{4, 4}, {4, 4}}, 
            {{5, 5}, {5, 5}},
            {{6, 6}, {6, 6}}
        }, 
        {
            {{7, 7}, {7, 7}}, 
            {{8, 8}, {8, 8}},
            {{9, 9}, {9, 9}}
        }, 
        {
            {{10, 10}, {10, 10}}, 
            {{11, 11}, {11, 11}},
            {{12, 12}, {12, 12}}
        }
    });

    Matrix* N = im2col(M, 1, 1, 2, 1);
    std::vector<std::vector<double>> ans = {
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 1,  2,  3},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 4,  5,  6},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 7,  8,  9},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        {10, 11, 12},
    };

    EXPECT_MATRIX_EQ(ans, N); 

    free_matrix_4d(M);
    free_matrix(N);
}

TEST(im2col, success6) {
    Matrix4d* M = create_matrix4d_from_stdvec(
    {
        {
            {{1, 1}, {1, 1}},
            {{2, 2}, {2, 2}},
            {{3, 3}, {3, 3}},
        }, 
        {
            {{4, 4}, {4, 4}}, 
            {{5, 5}, {5, 5}},
            {{6, 6}, {6, 6}}
        }, 
        {
            {{7, 7}, {7, 7}}, 
            {{8, 8}, {8, 8}},
            {{9, 9}, {9, 9}}
        }, 
        {
            {{10, 10}, {10, 10}}, 
            {{11, 11}, {11, 11}},
            {{12, 12}, {12, 12}}
        }
    });

    Matrix* N = im2col(M, 1, 1, 2, 2);
    std::vector<std::vector<double>> ans = {
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 1,  2,  3},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 4,  5,  6},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 7,  8,  9},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        {10, 11, 12},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0}
    };

    EXPECT_MATRIX_EQ(ans, N); 

    free_matrix_4d(M);
    free_matrix(N);
}

TEST(col2im, success) {
    Matrix* M = create_matrix_from_stdvec(
    {
        { 1,  2,  3},
        { 1,  2,  3},
        { 1,  2,  3},
        { 1,  2,  3},
        { 4,  5,  6},
        { 4,  5,  6},
        { 4,  5,  6},
        { 4,  5,  6},
        { 7,  8,  9},
        { 7,  8,  9},
        { 7,  8,  9},
        { 7,  8,  9},
        {10, 11, 12},
        {10, 11, 12},
        {10, 11, 12},
        {10, 11, 12}
   });

    int sizes[] = {4, 3, 2, 2};
    Matrix4d* N = col2im(M, sizes, 1, 1, 1, 0);
    std::vector<std::vector<std::vector<std::vector<double>>>> ans = {  
        {
            {{1, 1}, {1, 1}}, 
            {{2, 2}, {2, 2}},
            {{3, 3}, {3, 3}},
        }, 
        {
            {{4, 4}, {4, 4}}, 
            {{5, 5}, {5, 5}},
            {{6, 6}, {6, 6}}
        }, 
        {
            {{7, 7}, {7, 7}}, 
            {{8, 8}, {8, 8}},
            {{9, 9}, {9, 9}}
            
        }, 
        {
            {{10, 10}, {10, 10}}, 
            {{11, 11}, {11, 11}},
            {{12, 12}, {12, 12}}
        }
    };

    EXPECT_MATRIX4D_EQ(ans, N); 

    free_matrix(M);
    free_matrix_4d(N);
}

TEST(col2im, success2) {
    Matrix* M = create_matrix_from_stdvec(
    {
       { 1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3},
       { 4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6},
       { 7,  7,  7,  7,  8,  8,  8,  8,  9,  9,  9,  9},
       {10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12}
    });

    int sizes[] = {4, 3, 2, 2};
    Matrix4d* N = col2im(M, sizes,  2, 2, 1, 0);
    std::vector<std::vector<std::vector<std::vector<double>>>> ans = {  
        {
            {{1, 1}, {1, 1}}, 
            {{2, 2}, {2, 2}},
            {{3, 3}, {3, 3}},
        }, 
        {
            {{4, 4}, {4, 4}}, 
            {{5, 5}, {5, 5}},
            {{6, 6}, {6, 6}}
        }, 
        {
            {{7, 7}, {7, 7}}, 
            {{8, 8}, {8, 8}},
            {{9, 9}, {9, 9}}
            
        }, 
        {
            {{10, 10}, {10, 10}}, 
            {{11, 11}, {11, 11}},
            {{12, 12}, {12, 12}}
        }
    };

    EXPECT_MATRIX4D_EQ(ans, N); 

    free_matrix(M);
    free_matrix_4d(N);
}

TEST(col2im, success3) {
    Matrix* M = create_matrix_from_stdvec(
    {
        { 1,  2,  3},
        { 4,  5,  6},
        { 7,  8,  9},
        {10, 11, 12},
   });

    int sizes[] = {4, 3, 2, 2};
    Matrix4d* N = col2im(M, sizes, 1, 1, 2, 0);

    std::vector<std::vector<std::vector<std::vector<double>>>> ans = {  
        {
            {{1, 0}, {0, 0}}, 
            {{2, 0}, {0, 0}},
            {{3, 0}, {0, 0}},
        }, 
        {
            {{4, 0}, {0, 0}}, 
            {{5, 0}, {0, 0}},
            {{6, 0}, {0, 0}}
        }, 
        {
            {{7, 0}, {0, 0}}, 
            {{8, 0}, {0, 0}},
            {{9, 0}, {0, 0}}
            
        }, 
        {
            {{10, 0}, {0, 0}}, 
            {{11, 0}, {0, 0}},
            {{12, 0}, {0, 0}}
        }
    };

    EXPECT_MATRIX4D_EQ(ans, N); 

    free_matrix(M);
    free_matrix_4d(N);
}

TEST(col2im, success4) {
    Matrix* M = create_matrix_from_stdvec(
    {
       { 1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3},
       { 4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6},
       { 7,  7,  7,  7,  8,  8,  8,  8,  9,  9,  9,  9},
       {10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12}
    });

    int sizes[] = {4, 3, 2, 2};
    Matrix4d* N = col2im(M, sizes,  2, 2, 2, 0);
    std::vector<std::vector<std::vector<std::vector<double>>>> ans = {  
        {
            {{1, 1}, {1, 1}}, 
            {{2, 2}, {2, 2}},
            {{3, 3}, {3, 3}},
        }, 
        {
            {{4, 4}, {4, 4}}, 
            {{5, 5}, {5, 5}},
            {{6, 6}, {6, 6}}
        }, 
        {
            {{7, 7}, {7, 7}}, 
            {{8, 8}, {8, 8}},
            {{9, 9}, {9, 9}}
            
        }, 
        {
            {{10, 10}, {10, 10}}, 
            {{11, 11}, {11, 11}},
            {{12, 12}, {12, 12}}
        }
    };

    EXPECT_MATRIX4D_EQ(ans, N); 

    free_matrix(M);
    free_matrix_4d(N);
}

TEST(col2im, success5) {
    Matrix* M = create_matrix_from_stdvec({
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 1,  2,  3},
        { 1,  2,  3},
        { 0,  0,  0},
        { 0,  0,  0},
        { 1,  2,  3},
        { 1,  2,  3},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 4,  5,  6},
        { 4,  5,  6},
        { 0,  0,  0},
        { 0,  0,  0},
        { 4,  5,  6},
        { 4,  5,  6},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 7,  8,  9},
        { 7,  8,  9},
        { 0,  0,  0},
        { 0,  0,  0},
        { 7,  8,  9},
        { 7,  8,  9},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        {10, 11, 12},
        {10, 11, 12},
        { 0,  0,  0},
        { 0,  0,  0},
        {10, 11, 12},
        {10, 11, 12},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0}
    });

    int sizes[] = {4, 3, 2, 2};
    Matrix4d* N = col2im(M, sizes, 1, 1, 1, 1);
    std::vector<std::vector<std::vector<std::vector<double>>>> ans = {  
        {
            {{1, 1}, {1, 1}}, 
            {{2, 2}, {2, 2}},
            {{3, 3}, {3, 3}},
        }, 
        {
            {{4, 4}, {4, 4}}, 
            {{5, 5}, {5, 5}},
            {{6, 6}, {6, 6}}
        }, 
        {
            {{7, 7}, {7, 7}}, 
            {{8, 8}, {8, 8}},
            {{9, 9}, {9, 9}}
            
        }, 
        {
            {{10, 10}, {10, 10}}, 
            {{11, 11}, {11, 11}},
            {{12, 12}, {12, 12}}
        }
    };

    EXPECT_MATRIX4D_EQ(ans, N); 

    free_matrix(M);
    free_matrix_4d(N);
}

TEST(col2im, success6) {
    Matrix* M = create_matrix_from_stdvec({
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 1,  2,  3},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 4,  5,  6},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        { 7,  8,  9},
        { 0,  0,  0},
        { 0,  0,  0},
        { 0,  0,  0},
        {10, 11, 12},
    });

    int sizes[] = {4, 3, 2, 2};
    Matrix4d* N = col2im(M, sizes, 1, 1, 2, 1);
    std::vector<std::vector<std::vector<std::vector<double>>>> ans = {  
        {
            {{0, 0}, {0, 1}}, 
            {{0, 0}, {0, 2}},
            {{0, 0}, {0, 3}},
        }, 
        {
            {{0, 0}, {0, 4}}, 
            {{0, 0}, {0, 5}},
            {{0, 0}, {0, 6}}
        }, 
        {
            {{0, 0}, {0, 7}}, 
            {{0, 0}, {0, 8}},
            {{0, 0}, {0, 9}}
            
        }, 
        {
            {{0, 0}, {0, 10}}, 
            {{0, 0}, {0, 11}},
            {{0, 0}, {0, 12}}
        }
    };

    EXPECT_MATRIX4D_EQ(ans, N); 

    free_matrix(M);
    free_matrix_4d(N);
}

TEST(matrix_4d_pad, success) {
    Matrix4d* M = create_matrix4d_from_stdvec(
    {
        {
            {{1, 1}, {1, 1}},
            {{2, 2}, {2, 2}},
            {{3, 3}, {3, 3}},
        }, 
        {
            {{4, 4}, {4, 4}}, 
            {{5, 5}, {5, 5}},
            {{6, 6}, {6, 6}}
        }, 
        {
            {{7, 7}, {7, 7}}, 
            {{8, 8}, {8, 8}},
            {{9, 9}, {9, 9}}
        }, 
        {
            {{10, 10}, {10, 10}}, 
            {{11, 11}, {11, 11}},
            {{12, 12}, {12, 12}}
        }
    });

    Matrix4d* N = matrix_4d_pad(M, 1);

    std::vector<std::vector<std::vector<std::vector<double>>>> ans = {
        {
            {{0, 0, 0, 0}, {0, 1, 1, 0},   {0, 1, 1, 0},   {0, 0, 0, 0}},
            {{0, 0, 0, 0}, {0, 2, 2, 0},   {0, 2, 2, 0},   {0, 0, 0, 0}},
            {{0, 0, 0, 0}, {0, 3, 3, 0},   {0, 3, 3, 0},   {0, 0, 0, 0}},
        }, 
        {
            {{0, 0, 0, 0}, {0, 4, 4, 0},   {0, 4, 4, 0},   {0, 0, 0, 0}}, 
            {{0, 0, 0, 0}, {0, 5, 5, 0},   {0, 5, 5, 0},   {0, 0, 0, 0}},
            {{0, 0, 0, 0}, {0, 6, 6, 0},   {0, 6, 6, 0},   {0, 0, 0, 0}}
        }, 
        {
            {{0, 0, 0, 0}, {0, 7, 7, 0},   {0, 7, 7, 0},   {0, 0, 0, 0}}, 
            {{0, 0, 0, 0}, {0, 8, 8, 0},   {0, 8, 8, 0},   {0, 0, 0, 0}},
            {{0, 0, 0, 0}, {0, 9, 9, 0},   {0, 9, 9, 0},   {0, 0, 0, 0}}
        }, 
        {
            {{0, 0, 0, 0}, {0, 10, 10, 0}, {0, 10, 10, 0}, {0, 0, 0, 0}}, 
            {{0, 0, 0, 0}, {0, 11, 11, 0}, {0, 11, 11, 0}, {0, 0, 0, 0}},
            {{0, 0, 0, 0}, {0, 12, 12, 0}, {0, 12, 12, 0}, {0, 0, 0, 0}}
        }
    };
    EXPECT_MATRIX4D_EQ(ans, N); 

    free_matrix_4d(M);
    free_matrix_4d(N);
}

TEST(matrix_4d_pad, success2) {
    Matrix4d* M = create_matrix4d_from_stdvec(
    {
        {
            {{1, 1}, {1, 1}},
            {{2, 2}, {2, 2}},
            {{3, 3}, {3, 3}},
        }, 
        {
            {{4, 4}, {4, 4}}, 
            {{5, 5}, {5, 5}},
            {{6, 6}, {6, 6}}
        }, 
        {
            {{7, 7}, {7, 7}}, 
            {{8, 8}, {8, 8}},
            {{9, 9}, {9, 9}}
        }, 
        {
            {{10, 10}, {10, 10}}, 
            {{11, 11}, {11, 11}},
            {{12, 12}, {12, 12}}
        }
    });

    Matrix4d* N = matrix_4d_pad(M, 2);

    std::vector<std::vector<std::vector<std::vector<double>>>> ans = {
        {
            {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0},   {0, 0, 1, 1, 0, 0},   {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},
            {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 2, 2, 0, 0},   {0, 0, 2, 2, 0, 0},   {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},
            {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 3, 3, 0, 0},   {0, 0, 3, 3, 0, 0},   {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},
        }, 
        {
            {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 4, 4, 0, 0},   {0, 0, 4, 4, 0, 0},   {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}}, 
            {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 5, 5, 0, 0},   {0, 0, 5, 5, 0, 0},   {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},
            {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 6, 6, 0, 0},   {0, 0, 6, 6, 0, 0},   {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}}
        }, 
        {
            {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 7, 7, 0, 0},   {0, 0, 7, 7, 0, 0},   {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}}, 
            {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 8, 8, 0, 0},   {0, 0, 8, 8, 0, 0},   {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},
            {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 9, 9, 0, 0},   {0, 0, 9, 9, 0, 0},   {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}}
        }, 
        {
            {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 10, 10, 0, 0}, {0, 0, 10, 10, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}}, 
            {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 11, 11, 0, 0}, {0, 0, 11, 11, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},
            {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 12, 12, 0, 0}, {0, 0, 12, 12, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}}
        }
    };
    EXPECT_MATRIX4D_EQ(ans, N); 

    free_matrix_4d(M);
    free_matrix_4d(N);
}


TEST(create_image_batch, success) {
    double** images = (double**)malloc(sizeof(double*) * 5);
    for (int i = 0; i < 5; ++i) {
        images[i] = (double*)malloc(sizeof(double) * NUM_OF_PIXELS);
        for (int j = 0; j < NUM_OF_PIXELS; ++j) {
            images[i][j] = (i + 1) * j; 
        }
    }
    int batch_index[] = {2, 3, 4};

    Matrix* M = create_image_batch(images,(const int*)batch_index, 3);
    EXPECT_NE(nullptr, M);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < NUM_OF_PIXELS; ++j) {
            EXPECT_DOUBLE_EQ((i + 3) * j, M->elements[i][j]);
        }
    }

    for (int i = 0; i < 5; ++i) {
        free(images[i]);
    }
    free(images);
    free_matrix(M);
}

TEST(create_image_batch_4d, success) {
    double**** images = (double****)malloc(sizeof(double***) * 5);
    for (int i = 0; i < 5; ++i) {
        images[i] = (double***)malloc(sizeof(double**));
        images[i][0] = (double**)malloc(sizeof(double*) * NUM_OF_ROWS);
        for (int j = 0; j < NUM_OF_ROWS; ++j) {
            images[i][0][j] = (double*)malloc(sizeof(double) * NUM_OF_COLS);
            for (int k = 0; k < NUM_OF_COLS; ++k) {
                images[i][0][j][k] = (i + 1) * (j * NUM_OF_ROWS + k);
            }
        }
    }
    int batch_index[] = {2, 3, 4};

    Matrix4d* M = create_image_batch_4d(images,(const int*)batch_index, 3);
    EXPECT_NE(nullptr, M);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < NUM_OF_ROWS; ++j) {
            for (int k = 0; k < NUM_OF_COLS; ++k) {
                EXPECT_DOUBLE_EQ((i + 3) * (j * NUM_OF_ROWS + k), M->elements[i][0][j][k]);
            }
        }
    }

    for (int i = 0; i < 5; ++i) {
        free(images[i]);
    }
    free(images);
    free_matrix_4d(M);
}
