#ifndef TWOLAYERNET_H
#define TWOLAYERNET_H

#include <matrix.h>

typedef struct TwoLayerNet TwoLayerNet;
struct TwoLayerNet {
    Matrix* W1;
    Vector* b1;
    Matrix* W2;
    Vector* b2;
    // grads
    Matrix* dW1;
    Vector* db1;
    Matrix* dW2;
    Vector* db2;
};

TwoLayerNet* create_two_layer_net(int input_size, int hidden_size, int output_size);
void two_layer_net_numerical_gradient(TwoLayerNet* net, const Matrix* X, const Vector* t);
void two_layer_net_gradient(TwoLayerNet* net, const Matrix* X, const Vector* t);
double two_layer_net_accuracy(const TwoLayerNet* net, double** images, uint8_t* labels, int size);

#endif
