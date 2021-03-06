#include "multi_layer_net_extend.h"
#include "function.h"
#include "mnist.h"

#include <stdlib.h>
#include <math.h>

MultiLayerNetExtend* create_multi_layer_net_extend(
    int input_size, 
    int hidden_layer_num, 
    int hidden_size,
    int output_size,
    int batch_size,
    int weight_type,
    double weight,
    bool use_dropout,
    double dropout_ratio
) {
    MultiLayerNetExtend* net = malloc(sizeof(MultiLayerNetExtend));

    net->W = malloc(sizeof(Matrix*) * hidden_layer_num + 1);
    net->b = malloc(sizeof(Vector*) * hidden_layer_num + 1);
    net->A = malloc(sizeof(Affine*) * hidden_layer_num + 1);

    net->gamma = malloc(sizeof(Vector*)             * hidden_layer_num);
    net->beta  = malloc(sizeof(Vector*)             * hidden_layer_num);
    net->B     = malloc(sizeof(BatchNormalization*) * hidden_layer_num);

    net->R = malloc(sizeof(Relu*) * hidden_layer_num);

    if (use_dropout) {
        net->D = malloc(sizeof(Dropout*) * hidden_layer_num);
    }

    net->W[0] = create_matrix(input_size, hidden_size);
    net->b[0] = create_vector(hidden_size);
    net->A[0] = create_affine(net->W[0], net->b[0]);

    net->gamma[0] = create_vector_initval(hidden_size, 1);
    net->beta[0]  = create_vector(hidden_size);
    net->B[0]     = create_batch_normalization(net->gamma[0], net->beta[0], 0.9);

    net->R[0] = create_relu();

    if (use_dropout) {
        net->D[0] = create_dropout(dropout_ratio);
    }

    for (int i = 1; i < hidden_layer_num + 1; ++i) {
        if (i == hidden_layer_num) {
            net->W[i] = create_matrix(hidden_size, output_size);
            net->b[i] = create_vector(output_size);
            net->A[i] = create_affine(net->W[i], net->b[i]);
        } else {
            net->W[i] = create_matrix(hidden_size, hidden_size);
            net->b[i] = create_vector(hidden_size);
            net->A[i] = create_affine(net->W[i], net->b[i]);
            
            net->gamma[i] = create_vector_initval(hidden_size, 1);
            net->beta[i]  = create_vector(hidden_size);
            net->B[i]     = create_batch_normalization(net->gamma[i], net->beta[i], 0.9);

            net->R[i] = create_relu();

            if (use_dropout) {
                net->D[i] = create_dropout(dropout_ratio);
            }
        }
    }

    net->S = create_softmax_with_loss();
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->hidden_layer_num = hidden_layer_num;
    net->use_dropout = use_dropout;

    // init weight
    switch (weight_type) {
    case He: {
        for (int i = 0; i < hidden_layer_num + 1; ++i) {
            const double scale = (i == 0) ? (sqrt(2.0 / input_size)) : (sqrt(2.0 / hidden_size));
            init_matrix_random(net->W[i]);
            scalar_matrix(net->W[i], scale);
        }
        break;
    }
    case Xavier: {
        for (int i = 0; i < hidden_layer_num + 1; ++i) {
            const double scale = (i == 0) ? (sqrt(1.0 / input_size)) : (sqrt(1.0 / hidden_size));
            init_matrix_random(net->W[i]);
            scalar_matrix(net->W[i], scale);
        }
        break;
    }
    case STD: {
        for (int i = 0; i < hidden_layer_num + 1; ++i) {
            init_matrix_random(net->W[i]);
            scalar_matrix(net->W[i], weight);
        }
        break;
    }
    default: {
        break;
    }
    }

    return net;
}

static Matrix* predict(const MultiLayerNetExtend* net, const Matrix* X) {
    Matrix* X_tmp = NULL;
    for (int i = 0; i < net->hidden_layer_num + 1; ++i) {
        Matrix* X1 = NULL;
        if (i == 0) {
            X1 = affine_forward(net->A[i], X);
        } else {
            X1 = affine_forward(net->A[i], X_tmp);
            free_matrix(X_tmp);
        }

        if (i != net->hidden_layer_num) {
            Matrix* X2 = batch_normalization_forward(net->B[i], X1); 
            X_tmp = relu_forward(net->R[i], X2);

            if (net->use_dropout) {
                Matrix* X3 = X_tmp;
                X_tmp = dropout_forward(net->D[i], X3, true);
                free_matrix(X3);
            }

            free_matrix(X1);
            free_matrix(X2);
        } else {
            return X1;
        }
    }

    return NULL;
}

double multi_layer_net_extend_loss(MultiLayerNetExtend* net, const Matrix* X, const Vector* t) {
    Matrix* Y = predict(net, X);
    const double v = softmax_with_loss_forward(net->S, Y, t); 

    free(Y);
    return v;
}

void multi_layer_net_extend_gradient(MultiLayerNetExtend* net, const Matrix* X, const Vector* t) {
    multi_layer_net_extend_loss(net, X, t);

    Matrix* X1 = softmax_with_loss_backward(net->S);
    Matrix* X2 = affine_backward(net->A[net->hidden_layer_num], X1);  
    free_matrix(X1);

    for (int i = net->hidden_layer_num - 1; i >= 0; --i) {
        if (net->use_dropout) {
            Matrix* X_tmp = X2;
            X2 = dropout_backward(net->D[i], X_tmp);
            free_matrix(X_tmp); 
        }


        Matrix* X3 = relu_backward(net->R[i], X2);  
        free_matrix(X2);

        Matrix* X4 = batch_normalization_backward(net->B[i], X3);
        free_matrix(X3);

        X2 = affine_backward(net->A[i], X4);  
        free_matrix(X4);
    }

    free_matrix(X2);
}

double multi_layer_net_extend_accuracy(const MultiLayerNetExtend* net, double** images, uint8_t* labels, int size) {
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
