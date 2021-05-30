#include "twolayernet.h"
#include <function.h>
#include <mnist.h>
#include <stdlib.h>
#include <math.h>

TwoLayerNet* create_two_layer_net(int input_size, int hidden_size, int output_size, int batch_size) {
    TwoLayerNet* net = malloc(sizeof(TwoLayerNet));

    net->W1 = create_matrix(input_size, hidden_size);
    net->b1 = create_vector(hidden_size);
    net->W2 = create_matrix(hidden_size, output_size);
    net->b2 = create_vector(output_size);

    init_matrix_random(net->W1);
    scalar_matrix(net->W1, 0.01);
    init_matrix_random(net->W2);
    scalar_matrix(net->W2, 0.01);

    net->A1 = create_affine(net->W1, net->b1);
    net->R  = create_relu();
    net->A2 = create_affine(net->W2, net->b2);
    net->S  = create_softmax_with_loss();

    return net;
}

static Matrix* predict(const TwoLayerNet* net, const Matrix* X) {
    Matrix* X1 = affine_forward(net->A1, (Matrix*)X);
    Matrix* X2 = relu_forward(net->R, X1);
    Matrix* X3 = affine_forward(net->A2, X2);

    free_matrix(X1);
    free_matrix(X2);
    return X3;
}

static double loss(TwoLayerNet* net, const Matrix* X, const Vector* t) {
    Matrix* Y = predict(net, X);
    const double v = softmax_with_loss_forward(net->S, Y, (Vector*)t); 

    free_matrix(Y);
    return v;
}

void two_layer_net_gradient(TwoLayerNet* net, const Matrix* X, const Vector* t) {
    loss(net, X, t);

    Matrix* X1 = softmax_with_loss_backward(net->S);
    Matrix* X2 = affine_backward(net->A2, X1);  
    Matrix* X3 = relu_backward(net->R, X2);  
    Matrix* X4 = affine_backward(net->A1, X3);  

    free_matrix(X1);
    free_matrix(X2);
    free_matrix(X3);
    free_matrix(X4);
}

double two_layer_net_accuracy(const TwoLayerNet* net, double** images, uint8_t* labels, int size) {
    Matrix* X = create_matrix(size, NUM_OF_PIXELS);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < NUM_OF_PIXELS; ++j) {
           X->elements[i][j] = images[i][j];
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

    free_matrix(X);
    free_matrix(Y);
    return (double)cnt / size;
}
