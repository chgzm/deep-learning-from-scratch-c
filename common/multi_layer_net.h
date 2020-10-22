#ifndef MULTILAYERNET_H
#define MULTILAYERNET_H

#include "matrix.h"
#include "layer.h"

typedef struct MultiLayerNet MultiLayerNet;
struct MultiLayerNet {
    Matrix**         W;
    Vector**         b;
    Affine**         A;
    Relu**           R;
    SoftmaxWithLoss* S;
    int hidden_layer_num;
};

MultiLayerNet* create_multi_layer_net(
    int input_size, 
    int hidden_layer_num, 
    int hidden_size, 
    int output_size,
    int batch_size
);

void gradient(MultiLayerNet* net, const Matrix* X, const Vector* t);

double accuracy_multi_layer_net(const MultiLayerNet* net, double** images, uint8_t* labels, int size);

#endif
