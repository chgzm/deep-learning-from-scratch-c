#include "gtest/gtest.h"

#include "utest_util.h"

extern "C" {
#include <layer.h>
}

TEST(affine_forward_backward, success) {
    Matrix* W = create_matrix_from_stdvec({{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}, {1.0, 1.1, 1.2}});
    Vector* b = create_vector_from_stdvec({1, 2, 3});
    Affine* A = create_affine(W, b);

    Matrix* X = create_matrix_from_stdvec({{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2}});
    Matrix* M = affine_forward(A, X);
    EXPECT_MATRIX_NEAR({{1.7, 2.8, 3.9}, {2.58,3.84, 5.1}, {3.46, 4.88, 6.3}}, M);
    
    Matrix* B =  create_matrix_from_stdvec({{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}});
    Matrix* N = affine_backward(A, B);
    EXPECT_MATRIX_NEAR({{0.14, 0.32, 0.5, 0.68}, {0.32, 0.77, 1.22, 1.67}, {0.5,  1.22, 1.94, 2.66}}, N);

    free_affine(A);
    free_matrix(X);
    free_matrix(M);
    free_matrix(B);
    free_matrix(N);
}

TEST(relu_forward_backward, success) {
    Relu* R = create_relu();

    Matrix* M = create_matrix_from_stdvec({{-1, 1, -1}, {1, -1, 1}});
    Matrix* A = relu_forward(R, M);
    
    EXPECT_MATRIX_EQ({{0, 1, 0}, {1, 0, 1}}, A);

    Matrix* N = create_matrix_from_stdvec({{1, 1, 1}, {2, 2, 2}});
    Matrix* B = relu_backward(R, N);

    EXPECT_MATRIX_EQ({{0, 1, 0}, {2, 0, 2}}, B);

    free_matrix(M);
    free_matrix(N);
    free_matrix(A);
    free_matrix(B);
    free_relu(R);
}

TEST(softmax_with_loss_forward_backward, success) {
    SoftmaxWithLoss* S = create_softmax_with_loss();
    
    Matrix* X = create_matrix_from_stdvec({{1, 2, 2}, {4, 5, 6}});
    Vector* t = create_vector_from_stdvec({0, 1});

    const double loss = softmax_with_loss_forward(S, X, t);
    EXPECT_DOUBLE_EQ(loss, 1.6347998581152146);

    Matrix* M = softmax_with_loss_backward(S);
    EXPECT_MATRIX_NEAR({{-0.4223188, 0.2111594, 0.2111594}, {0.04501529, -0.37763576, 0.33262048}}, M);

    free_matrix(X);
    free_vector(t);
    free_matrix(M);
    free_softmax_with_loss(S);
}

TEST(batch_normalization_forward_backward, success) {
    Vector* g = create_vector_initval(4, 1);
    Vector* b = create_vector(4);
 
    BatchNormalization* B = create_batch_normalization(g, b, 0.9);
   
    Matrix* X = create_matrix_from_stdvec({{1, 3, 2, 4}, {8, 6, 7, 5}, {10, 9, 11, 12}});
    Matrix* M = batch_normalization_forward(B, X);
    EXPECT_MATRIX_NEAR({{-1.38218943, -1.22474477, -1.2675004, -0.8429272}, {0.4319342, 0, 0.09053574, -0.56195146}, {0.95025524, 1.22474477, 1.17696466, 1.40487866}}, M);

    scalar_matrix(X, 10000000);
    Matrix* N = batch_normalization_backward(B, X);
    EXPECT_MATRIX_NEAR({{-0.92833612, -2.04124094, -0.93504121, -0.66546879}, {0.29010504, 0, 0.06678866, -0.44364586}, {0.63823109, 2.04124094, 0.86825255, 1.10911464}}, N);

    free_batch_normalization(B);
    free_matrix(X);
    free_matrix(M);
    free_matrix(N);
}

TEST(dropout_forward_backward, success) {
    Dropout* D = create_dropout(0.5);
    
    Matrix* X = create_matrix_from_stdvec({{1, 3, 2, 4}, {8, 6, 7, 5}, {10, 9, 11, 12}});

    std::vector<int> zero_index;
    Matrix* M = dropout_forward(D, X);
    Matrix* N = dropout_backward(D, X);

    free_dropout(D);
    free_matrix(X);
    free_matrix(M);
    free_matrix(N);
}

TEST(convolution_forward_backward, success) {
    Matrix4d* M = create_matrix4d_from_stdvec({
        {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
        }, 
        {
             {{10,   20}, {30,   40}}, 
             {{50,   60}, {70,   80}},
             {{90,  100}, {110, 120}},
             {{130, 140}, {150, 160}}
        }
    });
    Vector* v = create_vector_from_stdvec({1, 2});

    Convolution* Conv = create_convolution(M, v, 1, 0);

    Matrix4d* X = create_matrix4d_from_stdvec({
        {
            {{1,   2}, {3,   4}}, 
            {{5,   6}, {7,   8}},
            {{9,  10}, {11, 12}},
            {{13, 14}, {15, 16}}
        }, 
        {
             {{10,   20}, {30,   40}}, 
             {{50,   60}, {70,   80}},
             {{90,  100}, {110, 120}},
             {{130, 140}, {150, 160}}
        }
    });

    Matrix4d* F = convolution_forward(Conv, X);

    const std::vector<std::vector<std::vector<std::vector<double>>>> ans = {  
        {
            {{1497}},
            {{14962}},
        },
        {
            {{14961}},
            {{149602}},
        }
    };
    EXPECT_MATRIX4D_EQ(ans, F);

    Matrix4d* B = convolution_backward(Conv, F);
    const std::vector<std::vector<std::vector<std::vector<double>>>> ans2 = {  
        {
            {{  151117,   302234.},
             {  453351,   604468.}},
        
            {{  755585,   906702.},
             { 1057819,  1208936.}},
        
            {{ 1360053,  1511170.},
             { 1662287,  1813404.}},
        
            {{ 1964521,  2115638.},
             { 2266755,  2417872.}},
        },
        {  
            {{ 1510981,  3021962.},
             { 4532943,  6043924.}},
        
            {{ 7554905,  9065886.},
             {10576867, 12087848.}},
        
            {{13598829, 15109810.},
             {16620791, 18131772.}},
        
            {{19642753, 21153734.},
             {22664715, 24175696.}},
        }
    };
    EXPECT_MATRIX4D_EQ(ans2, B);

    free_matrix_4d(F);
    free_matrix_4d(B);

    free_convolution(Conv);
}

