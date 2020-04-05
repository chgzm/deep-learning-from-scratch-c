#ifndef TWOLAYERNET_H
#define TWOLAYERNET_H

#include <matrix.h>

#define LEARNING_RATE 0.1

typedef struct TwoLayerNet TwoLayerNet;
struct TwoLayerNet {
    Matrix* W1;
    Vector* b1;
    Matrix* W2;
    Vector* b2;
};

TwoLayerNet* create_two_layer_net(int input_size, int hidden_size, int output_size);
void numerical_gradient(TwoLayerNet* net, const Matrix* X, const Vector* t);
void gradient(TwoLayerNet* net, const Matrix* X, const Vector* t);

double accuracy_two_layer_net(const TwoLayerNet* net, double** images, uint8_t* labels, int size);

#endif
