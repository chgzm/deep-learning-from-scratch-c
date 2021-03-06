#include "deep_convnet.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mnist.h>
#include <function.h>

DeepConvNet* create_deep_convnet(int* input_dim, ConvParam* params, int hidden_size, int output_size) {
    DeepConvNet* net = malloc(sizeof(DeepConvNet));

    const int pre_node_nums[] = {1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size};
    int weight_init_scales[8];

    for (int i = 0; i < 8; ++i) {
        weight_init_scales[i] = sqrt(2.0 / pre_node_nums[i]);
    }

    // Convolution - Relu
    int pre_channel_num = input_dim[0];
    for (int i = 0; i < 6; ++i) {
        Matrix4d* W4d = create_matrix_4d(params[i].filter_num, pre_channel_num, params[i].filter_size, params[i].filter_size);
        init_matrix_4d_random(W4d);
        scalar_matrix_4d(W4d, weight_init_scales[i]);
        Vector* b = create_vector(params[i].filter_num);
        pre_channel_num = params[i].filter_num;

        net->C[i] = create_convolution(W4d, b, params[i].stride, params[i].pad);
        net->R4d[i] = create_relu_4d();
    }

    // Pooling
    net->P[0] = create_pooling(2, 2, 2, 0);
    net->P[1] = create_pooling(2, 2, 2, 0);
    net->P[2] = create_pooling(2, 2, 2, 0);

    // Affine 
    Matrix* W0 = create_matrix(64*4*4, hidden_size);
    init_matrix_random(W0);
    scalar_matrix(W0, weight_init_scales[6]);
    Vector* b0 = create_vector(hidden_size);
    net->A[0] = create_affine(W0, b0);
  
    Matrix* W1 = create_matrix(hidden_size, output_size);
    init_matrix_random(W1);
    scalar_matrix(W1, weight_init_scales[7]);
    Vector* b2 = create_vector(output_size);
    net->A[1] = create_affine(W1, b2);
    
    // Relu
    net->R = create_relu();

    // Dropout
    net->D[0] = create_dropout(0.5);
    net->D[1] = create_dropout(0.5);

    // SoftmaxWithLoss
    net->S = create_softmax_with_loss();

    return net;
}

void free_deep_convnet(DeepConvNet* net) {
    for (int i = 0; i < 6; ++i) {
        free_convolution(net->C[i]);
        free_relu_4d(net->R4d[i]);
    }

    for (int i = 0; i < 3; ++i) {
        free_pooling(net->P[i]);
    }

    for (int i = 0; i < 2; ++i) {
        free_affine(net->A[i]);
        free_dropout(net->D[i]);
    }

    free_softmax_with_loss(net->S);
    free_relu(net->R);

    free(net);
}

int deep_convnet_load_params(DeepConvNet* net) {
    if (init_matrix_4d_from_file(net->C[0]->W, "./data/W1.csv")) { fprintf(stderr, "Failed to load ./data/W1.csv\n"); }
    if (init_matrix_4d_from_file(net->C[1]->W, "./data/W2.csv")) { fprintf(stderr, "Failed to load ./data/W2.csv\n"); }
    if (init_matrix_4d_from_file(net->C[2]->W, "./data/W3.csv")) { fprintf(stderr, "Failed to load ./data/W3.csv\n"); }
    if (init_matrix_4d_from_file(net->C[3]->W, "./data/W4.csv")) { fprintf(stderr, "Failed to load ./data/W4.csv\n"); }
    if (init_matrix_4d_from_file(net->C[4]->W, "./data/W5.csv")) { fprintf(stderr, "Failed to load ./data/W5.csv\n"); }
    if (init_matrix_4d_from_file(net->C[5]->W, "./data/W6.csv")) { fprintf(stderr, "Failed to load ./data/W6.csv\n"); }
    if (init_matrix_from_file(net->A[0]->W,    "./data/W7.csv")) { fprintf(stderr, "Failed to load ./data/W7.csv\n"); }
    if (init_matrix_from_file(net->A[1]->W,    "./data/W8.csv")) { fprintf(stderr, "Failed to load ./data/W8.csv\n"); }

    if (init_vector_from_file(net->C[0]->b,    "./data/b1.csv")) { fprintf(stderr, "Failed to load ./data/b1.csv\n"); }
    if (init_vector_from_file(net->C[1]->b,    "./data/b2.csv")) { fprintf(stderr, "Failed to load ./data/b2.csv\n"); }
    if (init_vector_from_file(net->C[2]->b,    "./data/b3.csv")) { fprintf(stderr, "Failed to load ./data/b3.csv\n"); }
    if (init_vector_from_file(net->C[3]->b,    "./data/b4.csv")) { fprintf(stderr, "Failed to load ./data/b4.csv\n"); }
    if (init_vector_from_file(net->C[4]->b,    "./data/b5.csv")) { fprintf(stderr, "Failed to load ./data/b5.csv\n"); }
    if (init_vector_from_file(net->C[5]->b,    "./data/b6.csv")) { fprintf(stderr, "Failed to load ./data/b6.csv\n"); }
    if (init_vector_from_file(net->A[0]->b,    "./data/b7.csv")) { fprintf(stderr, "Failed to load ./data/b7.csv\n"); }
    if (init_vector_from_file(net->A[1]->b,    "./data/b8.csv")) { fprintf(stderr, "Failed to load ./data/b8.csv\n"); }

    return 0; 
}

Matrix* deep_convnet_predict(const DeepConvNet* net, Matrix4d* X, bool train_flg) {
    Matrix4d* T   = convolution_forward(net->C[0], X);
    Matrix4d* T2  = relu_4d_forward(net->R4d[0], T);
    Matrix4d* T3  = convolution_forward(net->C[1], T2);
    Matrix4d* T4  = relu_4d_forward(net->R4d[1], T3);
    Matrix4d* T5  = pooling_forward(net->P[0], T4);

    Matrix4d* T6  = convolution_forward(net->C[2], T5);
    Matrix4d* T7  = relu_4d_forward(net->R4d[2], T6);
    Matrix4d* T8  = convolution_forward(net->C[3], T7);
    Matrix4d* T9  = relu_4d_forward(net->R4d[3], T8);
    Matrix4d* T10 = pooling_forward(net->P[1], T9);

    Matrix4d* T11 = convolution_forward(net->C[4], T10);
    Matrix4d* T12 = relu_4d_forward(net->R4d[4], T11);
    Matrix4d* T13 = convolution_forward(net->C[5], T12);
    Matrix4d* T14 = relu_4d_forward(net->R4d[5], T13);
    Matrix4d* T15 = pooling_forward(net->P[2], T14);

    Matrix*   T16 = affine_4d_forward(net->A[0], T15);
    Matrix*   T17 = relu_forward(net->R, T16);
    Matrix*   T18 = dropout_forward(net->D[0], T17, train_flg);
    Matrix*   T19 = affine_forward(net->A[1], T18);
    Matrix*   Y   = dropout_forward(net->D[1], T19, train_flg);

    free_matrix_4d(T);
    free_matrix_4d(T2);
    free_matrix_4d(T3);
    // free_matrix_4d(T4);
    free_matrix_4d(T5);
    free_matrix_4d(T6);
    free_matrix_4d(T7);
    free_matrix_4d(T8);
    // free_matrix_4d(T9);
    free_matrix_4d(T10);
    free_matrix_4d(T11);
    free_matrix_4d(T12);
    free_matrix_4d(T13);
    // free_matrix_4d(T14);
    free_matrix_4d(T15);
    free_matrix(T16);
    free_matrix(T17);
    free_matrix(T18);
    free_matrix(T19);

    return Y;
}

double deep_convnet_loss(DeepConvNet* net, Matrix4d* X, const Vector* t) {
    Matrix* Y = deep_convnet_predict(net, X, true);
    const double v = softmax_with_loss_forward(net->S, Y, t); 

    free_matrix(Y);
    return v;
}

void deep_convnet_gradient(DeepConvNet* net, Matrix4d* X, const Vector* t) {
    // TODO
}

double deep_convnet_accuracy(const DeepConvNet* net, double**** images, uint8_t* labels, int size, int num_channels) {
    Matrix4d* X = create_matrix_4d(size, num_channels, NUM_OF_ROWS, NUM_OF_COLS);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < num_channels; ++j) {
            for (int k = 0; k < NUM_OF_ROWS; ++k) {
                for (int l = 0; l < NUM_OF_COLS; ++l) {
                    X->elements[i][j][k][l] = images[i][j][k][l];
                }
            }
        } 
    }

    Matrix* Y = deep_convnet_predict(net, X, false);
    int cnt = 0;
    for (int i = 0; i < Y->rows; ++i) {
        const int max_index = argmax(Y->elements[i], Y->cols);
        if (max_index == labels[i]) {
            ++cnt;
        }
    }

    free_matrix_4d(X);
    free_matrix(Y);

    return (double) cnt / size;
}

