#include "simple_convnet.h"

#include "mnist.h"
#include "function.h"

#include <stdio.h>
#include <stdlib.h>

SimpleConvNet* create_simple_convnet(
    int input_dim1,
    int input_dim2,
    int input_dim3,
    int filter_num,
    int filter_size,
    int pad,
    int stride,
    int hidden_size,
    int output_size,
    double weight
) {
    SimpleConvNet* net = malloc(sizeof(SimpleConvNet));

    const int input_size = input_dim2;
    const int conv_output_size = (input_size - filter_size + 2 * pad) / stride + 1;
    const int pool_output_size = (int)(filter_num * (conv_output_size / 2.0) * (conv_output_size /2.0));
   
    Matrix4d* W0 = create_matrix_4d(filter_num, input_dim1, filter_size, filter_size);
    Vector* b0 = create_vector(filter_num);

    Matrix* W1 = create_matrix(pool_output_size, hidden_size);
    Vector* b1 = create_vector(hidden_size);

    Matrix* W2 = create_matrix(hidden_size, output_size);
    Vector* b2 = create_vector(output_size);

    net->C    = create_convolution(W0, b0, stride, pad);
    net->R4d  = create_relu_4d();
    net->P    = create_pooling(2, 2, 2, 0);
    net->A[0] = create_affine(W1, b1);
    net->R    = create_relu();
    net->A[1] = create_affine(W2, b2);
    net->S    = create_softmax_with_loss();

    // init weight
    init_matrix_4d_random(net->C->W); 
    scalar_matrix_4d(net->C->W, weight);
    
    init_matrix_random(net->A[0]->W);
    scalar_matrix(net->A[0]->W, weight);

    init_matrix_random(net->A[1]->W);
    scalar_matrix(net->A[1]->W, weight);

    return net;
}

void free_simple_convnet(SimpleConvNet* net) {
    free_convolution(net->C);
    free_pooling(net->P);

    free_relu_4d(net->R4d);
    free_relu(net->R);

    free_affine(net->A[0]);
    free_affine(net->A[1]);

    free_softmax_with_loss(net->S);

    free(net); 
}

int simple_convnet_load_params(SimpleConvNet* net) {
    if (init_matrix_4d_from_file(net->C->W, "./data/W1.csv")) { fprintf(stderr, "Failed to load ./data/W1.csv\n"); }
    if (init_matrix_from_file(net->A[0]->W, "./data/W2.csv")) { fprintf(stderr, "Failed to load ./data/W2.csv\n"); }
    if (init_matrix_from_file(net->A[1]->W, "./data/W3.csv")) { fprintf(stderr, "Failed to load ./data/W3.csv\n"); }

    if (init_vector_from_file(net->C->b,    "./data/b1.csv")) { fprintf(stderr, "Failed to load ./data/b1.csv\n"); }
    if (init_vector_from_file(net->A[0]->b, "./data/b2.csv")) { fprintf(stderr, "Failed to load ./data/b2.csv\n"); }
    if (init_vector_from_file(net->A[1]->b, "./data/b3.csv")) { fprintf(stderr, "Failed to load ./data/b3.csv\n"); }

    return 0; 
}

static Matrix* predict(const SimpleConvNet* net, Matrix4d* X) {
    Matrix4d* T  = convolution_forward(net->C, X);
    Matrix4d* T2 = relu_4d_forward(net->R4d, T);
    Matrix4d* T3 = pooling_forward(net->P, T2);
    Matrix*   T4 = affine_4d_forward(net->A[0], T3);
    Matrix*   T5 = relu_forward(net->R, T4);
    Matrix*   Y  = affine_forward(net->A[1], T5);

    free_matrix_4d(T);
    // free_matrix_4d(T2);
    free_matrix_4d(T3);
    free_matrix(T4);
    free_matrix(T5);

    return Y;
}

double simple_convnet_loss(SimpleConvNet* net, Matrix4d* X, const Vector* t) {
    Matrix* Y = predict(net, X);
    const double v = softmax_with_loss_forward(net->S, Y, t); 

    free_matrix(Y);
    return v;
}

void simple_convnet_gradient(SimpleConvNet* net, Matrix4d* X, const Vector* t) {
    simple_convnet_loss(net, X, t);

    Matrix* T  = softmax_with_loss_backward(net->S);
    Matrix* T2 = affine_backward(net->A[1], T);  
    Matrix* T3 = relu_backward(net->R, T2);  
    Matrix4d* T4 = affine_4d_backward(net->A[0], T3);  
    Matrix4d* T5 = pooling_backward(net->P, T4);  
    Matrix4d* T6 = relu_4d_backward(net->R4d, T5);  
    Matrix4d* T7 = convolution_backward(net->C, T6);  

    free_matrix(T);
    free_matrix(T2);
    free_matrix(T3);

    free_matrix_4d(T4);
    free_matrix_4d(T5);
    free_matrix_4d(T6);
    free_matrix_4d(T7);
}

double simple_convnet_accuracy(const SimpleConvNet* net, double**** images, uint8_t* labels, int size, int num_channels) {
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

    Matrix* Y = predict(net, X);
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

