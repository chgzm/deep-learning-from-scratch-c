#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include <stdbool.h>

typedef struct Affine Affine;
struct Affine {
    Matrix* W;
    Vector* b;
    Matrix* X;
    Matrix* dW;
    Vector* db;
};

typedef struct Mask Mask;
struct Mask {
    int    rows;
    int    cols;
    bool** elements;
};

typedef struct Relu Relu;
struct Relu {
    Mask* mask;
};

typedef struct SoftmaxWithLoss SoftmaxWithLoss;
struct SoftmaxWithLoss {
    double  loss;
    Matrix*    Y;
    Vector*    t;
};

Affine* create_affine(Matrix* W, Vector* b);
Matrix* affine_forward(Affine* A, Matrix* X);
Matrix* affine_backward(Affine* A, const Matrix* D);

Relu* create_relu(int rows, int cols);
Matrix* relu_forward(Relu* R, const Matrix* X);
Matrix* relu_backward(Relu* R, const Matrix* D);

SoftmaxWithLoss* create_softmax_with_loss();
double softmax_with_loss_forward(SoftmaxWithLoss* sft, const Matrix* X, Vector* t);
Matrix* softmax_with_loss_backward(SoftmaxWithLoss* sft);

#endif
