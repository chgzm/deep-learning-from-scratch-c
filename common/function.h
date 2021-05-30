#ifndef FUNCTION_H
#define FUNCTION_H

#include "matrix.h"

double sigmoid(double x);
Vector* vector_sigmoid(const Vector* v);
Matrix* matrix_sigmoid(const Matrix* M);

Matrix* sigmoid_grad(const Matrix* M);

Vector* vector_softmax(const Vector* v);
Matrix* matrix_softmax(const Matrix* M);

int vector_argmax(const Vector* v);
int* matrix_argmax_row(const Matrix* M);

#endif
