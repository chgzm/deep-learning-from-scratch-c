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

typedef struct Convolution Convolution;
struct Convolution {
    Matrix4d* W;
    Vector* b;
    int stride;
    int pad;
    Matrix4d* x;
    Matrix* col;
    Matrix* col_W;
    Vector* db;
    Matrix4d* dW;
};

typedef struct Pooling Pooling;
struct Pooling {
    int pool_h;
    int pool_w;
    int stride;
    int pad;
    Matrix4d* x;
    int* arg_max;
};

Affine* create_affine(Matrix* W, Vector* b);
void free_affine(Affine* A);
Matrix* affine_forward(Affine* A, const Matrix* X);
Matrix* affine_backward(Affine* A, const Matrix* D);

Relu* create_relu();
void free_relu(Relu* R);
Matrix* relu_forward(Relu* R, const Matrix* X);
Matrix* relu_backward(Relu* R, const Matrix* D);

SoftmaxWithLoss* create_softmax_with_loss();
void free_softmax_with_loss(SoftmaxWithLoss* S);
double softmax_with_loss_forward(SoftmaxWithLoss* sft, const Matrix* X, const Vector* t);
Matrix* softmax_with_loss_backward(const SoftmaxWithLoss* sft);

BatchNormalization* create_batch_normalization(Vector* g, Vector* b, double momentum);
void free_batch_normalization(BatchNormalization* B);
Matrix* batch_normalization_forward(BatchNormalization* B, const Matrix* X);
Matrix* batch_normalization_backward(BatchNormalization* B, const Matrix* D);

Dropout* create_dropout(double dropout_ratio);
void free_dropout(Dropout* D);
Matrix* dropout_forward(Dropout* D, const Matrix* X);
Matrix* dropout_backward(const Dropout* D, const Matrix* X);

Convolution* create_convolution(Matrix4d* W, Vector* b, int stride, int pad);
void free_convolution(Convolution* C);
Matrix4d* convolution_forward(Convolution* C, Matrix4d* X);
Matrix4d* convolution_backward(Convolution* C, const Matrix4d* X);

Pooling* create_pooling(int pool_h, int pool_w, int stride, int pad);
void free_pooling(Pooling* P);
Matrix4d* pooling_forward(Pooling* P, Matrix4d* X);
Matrix4d* pooling_backward(const Pooling* P, const Matrix4d* X);

#endif
