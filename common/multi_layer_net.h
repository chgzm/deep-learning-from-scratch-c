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
    int input_size;
    int hidden_size;
    int hidden_layer_num;
    double weight_decay_lambda;
};

enum WEIGHT_TYPE {
    STD,
    Xavier,
    He, 
};

MultiLayerNet* create_multi_layer_net(
    int input_size, 
    int hidden_layer_num, 
    int hidden_size, 
    int output_size,
    int batch_size,
    int weight_type,
    double weight,
    double weight_decay_lambda
);

void free_multi_layer_net(MultiLayerNet* net);

void multi_layer_net_gradient(MultiLayerNet* net, const Matrix* X, const Vector* t);
double multi_layer_net_loss(MultiLayerNet* net, const Matrix* X, const Vector* t);
double multi_layer_net_accuracy(const MultiLayerNet* net, double** images, uint8_t* labels, int size);

#endif
