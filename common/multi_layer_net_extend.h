#ifndef MULTILAYERNETEXTEND_H
#define MULTILAYERNETEXTEND_H

#include "matrix.h"
#include "layer.h"
#include "multi_layer_net.h"

typedef struct MultiLayerNetExtend MultiLayerNetExtend;
struct MultiLayerNetExtend {
    Matrix**             W;
    Vector**             b;
    Affine**             A;
    Vector**             gamma;
    Vector**             beta; 
    BatchNormalization** B;
    Relu**               R;
    SoftmaxWithLoss*     S;
    int input_size;
    int hidden_size;
    int hidden_layer_num;
};

MultiLayerNetExtend* create_multi_layer_net_extend(
    int input_size, 
    int hidden_layer_num, 
    int hidden_size, 
    int output_size,
    int batch_size,
    int weight_type,
    double weight
);

void multi_layer_net_extend_gradient(MultiLayerNetExtend* net, const Matrix* X, const Vector* t);
double multi_layer_net_extend_loss(MultiLayerNetExtend* net, const Matrix* X, const Vector* t);
double multi_layer_net_extend_accuracy(const MultiLayerNetExtend* net, double** images, uint8_t* labels, int size);

#endif
