#ifndef FUNCTION_H
#define FUNCTION_H

#include "matrix.h"

double sigmoid(double x);
Vector* vector_sigmoid(const Vector* v);
Matrix* sigmoid_grad(const Matrix* M);

Vector* vector_softmax(const Vector* v);
Matrix* matrix_softmax(const Matrix* M);

int argmax(const Vector* v);

#endif
