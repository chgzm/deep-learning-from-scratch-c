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

typedef struct BatchNormalization BatchNormalization;
struct BatchNormalization {
    Vector* g;
    Vector* b;
    Vector* dg;
    Vector* db;  

    Matrix* xc;
    Matrix* xn;
    Vector* std;
    Vector* running_mean;
    Vector* running_var;

    double momentum;
    int batch_size;
};

typedef struct Dropout Dropout;
struct Dropout {
    double dropout_ratio;
    Mask* mask;
};

Affine* create_affine(Matrix* W, Vector* b);
Matrix* affine_forward(Affine* A, const Matrix* X);
Matrix* affine_backward(Affine* A, const Matrix* D);

Relu* create_relu();
Matrix* relu_forward(Relu* R, const Matrix* X);
Matrix* relu_backward(Relu* R, const Matrix* D);

SoftmaxWithLoss* create_softmax_with_loss();
double softmax_with_loss_forward(SoftmaxWithLoss* sft, const Matrix* X, const Vector* t);
Matrix* softmax_with_loss_backward(const SoftmaxWithLoss* sft);

BatchNormalization* create_batch_normalization(Vector* g, Vector* b, double momentum);
Matrix* batch_normalization_forward(BatchNormalization* B, const Matrix* X);
Matrix* batch_normalization_backward(BatchNormalization* B, const Matrix* D);

Dropout* create_dropout(double dropout_ratio);
Matrix* dropout_forward(Dropout* D, const Matrix* X);
Matrix* dropout_backward(const Dropout* D, const Matrix* X);

#endif
