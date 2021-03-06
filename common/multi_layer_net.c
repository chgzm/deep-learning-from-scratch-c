#include "multi_layer_net.h"
#include "function.h"
#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

MultiLayerNet* create_multi_layer_net(
    int input_size, 
    int hidden_layer_num, 
    int hidden_size,
    int output_size,
    int batch_size,
    int weight_type,
    double weight,
    double weight_decay_lambda
) {
    MultiLayerNet* net = malloc(sizeof(MultiLayerNet));

    net->W = malloc(sizeof(Matrix*) * hidden_layer_num + 1);
    net->b = malloc(sizeof(Vector*) * hidden_layer_num + 1);
    net->A = malloc(sizeof(Affine*) * hidden_layer_num + 1);
    net->R = malloc(sizeof(Relu*) * hidden_layer_num);

    net->W[0] = create_matrix(input_size, hidden_size);
    net->b[0] = create_vector(hidden_size);
    net->A[0] = create_affine(net->W[0], net->b[0]);
    net->R[0] = create_relu();

    for (int i = 1; i < hidden_layer_num + 1; ++i) {
        if (i == hidden_layer_num) {
            net->W[i] = create_matrix(hidden_size, output_size);
            net->b[i] = create_vector(output_size);
            net->A[i] = create_affine(net->W[i], net->b[i]);
        } else {
            net->W[i] = create_matrix(hidden_size, hidden_size);
            net->b[i] = create_vector(hidden_size);
            net->A[i] = create_affine(net->W[i], net->b[i]);
            net->R[i] = create_relu();
        }
    }

    net->S = create_softmax_with_loss();
    net->input_size = input_size;
    net->hidden_size = hidden_size;
    net->hidden_layer_num = hidden_layer_num;
    net->weight_decay_lambda = weight_decay_lambda;

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

void free_multi_layer_net(MultiLayerNet* net) {
    for (int i = 0; i < net->hidden_layer_num + 1; ++i) {
        free_affine(net->A[i]);

        if (i != net->hidden_layer_num) {
            free_relu(net->R[i]);
        }
    }

    free_softmax_with_loss(net->S);
    free(net->W);
    free(net->b);
    free(net->A);
    free(net->R);

    free(net);
}

static Matrix* predict(const MultiLayerNet* net, const Matrix* X) {
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
            X_tmp = relu_forward(net->R[i], X1);
            free_matrix(X1);
        } else {
            return X1;
        }
    }

    return NULL;
}

double multi_layer_net_loss(MultiLayerNet* net, const Matrix* X, const Vector* t) {
    Matrix* Y = predict(net, X);
    const double v = softmax_with_loss_forward(net->S, Y, t); 

    double weight_decay = 0;
    for (int i = 0; i < net->hidden_layer_num + 1; ++i) {
        Matrix* tmp = pow_matrix(net->W[i], 2);
        const double sum = matrix_sum(tmp);

        weight_decay += 0.5 * net->weight_decay_lambda * sum;
        free_matrix(tmp);
    }

    free_matrix(Y);
    return v + weight_decay;
}

void multi_layer_net_gradient(MultiLayerNet* net, const Matrix* X, const Vector* t) {
    multi_layer_net_loss(net, X, t);

    Matrix* X1 = softmax_with_loss_backward(net->S);
    Matrix* X2 = affine_backward(net->A[net->hidden_layer_num], X1);  
    free_matrix(X1);

    for (int i = net->hidden_layer_num - 1; i >= 0; --i) {
        Matrix* X3 = relu_backward(net->R[i], X2);  
        free_matrix(X2);

        X2 = affine_backward(net->A[i], X3);  
        free_matrix(X3);
    }

    free_matrix(X2);
}

double multi_layer_net_accuracy(const MultiLayerNet* net, double** images, uint8_t* labels, int size) {
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

    return (double) cnt / size;
}
