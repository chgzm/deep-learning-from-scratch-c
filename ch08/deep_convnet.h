#ifndef DEEP_CONVNET_H
#define DEEP_CONVNET_H

#include "matrix.h"
#include "layer.h"

typedef struct ConvParam ConvParam;
struct ConvParam {
    int filter_num;
    int filter_size;
    int pad;
    int stride;
};

typedef struct DeepConvNet DeepConvNet;
struct DeepConvNet {
    Convolution* C[6];
    Relu4d* R4d[6];
    Pooling* P[3];
    Affine* A[2];
    Relu* R;
    Dropout* D[2];
    SoftmaxWithLoss* S;
    Matrix4d* W4d[6]; 
    Matrix* W[2];
    Vector* b[8]; 
};

DeepConvNet* create_deep_convnet(int* intput_dim, ConvParam* params, int hidden_size, int output_size); 
void free_deep_convnet(DeepConvNet* net);
int deep_convnet_load_params(DeepConvNet* net);

Matrix* deep_convnet_predict(const DeepConvNet* net, Matrix4d* X, bool train_flg);
double deep_convnet_loss(DeepConvNet* net, Matrix4d* X, const Vector* t);
void deep_convnet_gradient(DeepConvNet* net, Matrix4d* X, const Vector* t);
double deep_convnet_accuracy(const DeepConvNet* net, double**** images, uint8_t* labels, int size, int num_channels);

#endif
