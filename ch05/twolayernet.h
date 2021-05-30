#ifndef TWOLAYERNET_H
#define TWOLAYERNET_H

#include <matrix.h>
#include <layer.h>

typedef struct TwoLayerNet TwoLayerNet;
struct TwoLayerNet {
    Matrix*         W1;
    Vector*         b1;
    Matrix*         W2;
    Vector*         b2;
    Affine*         A1;
    Relu*            R;
    Affine*         A2;
    SoftmaxWithLoss* S;
};

TwoLayerNet* create_two_layer_net(int input_size, int hidden_size, int output_size, int batch_size);
void two_layer_net_gradient(TwoLayerNet* net, const Matrix* X, const Vector* t);
double two_layer_net_accuracy(const TwoLayerNet* net, double** images, uint8_t* labels, int size);

#endif
