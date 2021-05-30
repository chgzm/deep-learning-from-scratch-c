#ifndef SIMPLE_CONVNET_H
#define SIMPLE_CONVNET_H

#include "matrix.h"
#include "layer.h"

typedef struct SimpleConvNet SimpleConvNet;
struct SimpleConvNet {
    Convolution* C;
    Relu4d*      R4d;
    Pooling*     P;
    Relu*        R;
    Affine*      A[2];
    SoftmaxWithLoss* S;
};

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
);

void free_simple_convnet(SimpleConvNet* net);

double simple_convnet_loss(SimpleConvNet* net, Matrix4d* X, const Vector* t);
void simple_convnet_gradient(SimpleConvNet* net, Matrix4d* X, const Vector* t);
double simple_convnet_accuracy(const SimpleConvNet* net, double**** images, uint8_t* labels, int size, int num_channels);

#endif
