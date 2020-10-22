#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "matrix.h"

void SGD_update(double* x, double dx, double lr);
void SGD_update_matrix(Matrix* A, const Matrix* dA, double lr);
void SGD_update_vector(Vector* V, const Vector* dV, double lr);

void Momentum_update(double* x, double dx, double lr, double momentum, double* v);
void Momentum_update_matrix(Matrix* A, const Matrix* dA, double lr);
void Momentum_update_vector(Vector* V, const Vector* dV, double lr);

void AdaGrad_update(double* x, double dx, double lr, double* h);
void AdaGrad_update_matrix(Matrix* A, const Matrix* dA, double lr);
void AdaGrad_update_vector(Vector* V, const Vector* dV, double lr);

void Adam_update(double* x, double dx, double lr, double beta1, double beta2, double* m, double* v, int iter);
void Adam_update_matrix(Matrix* A, const Matrix* dA, double lr);
void Adam_update_vector(Vector* V, const Vector* dV, double lr);

#endif
