#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "matrix.h"

enum {
    SGD,
    Momentum,
    AdaGrad,
    Adam 
};

void SGD_update(double* x, double dx, double lr);
void SGD_update_vector(Vector* V, const Vector* dV, double lr);
void SGD_update_matrix(Matrix* A, const Matrix* dA, double lr);
void SGD_update_matrix_4d(Matrix4d* A, const Matrix4d* dA, double lr);

void Momentum_update(double* x, double dx, double lr, double momentum, double* v);
void Momentum_update_vector(Vector* V, const Vector* dV, double lr, double momentum, Vector* v);
void Momentum_update_matrix(Matrix* A, const Matrix* dA, double lr, double momentum, Matrix* v);

void AdaGrad_update(double* x, double dx, double lr, double* h);
void AdaGrad_update_vector(Vector* V, const Vector* dV, double lr, Vector* h);
void AdaGrad_update_matrix(Matrix* A, const Matrix* dA, double lr, Matrix* h);

void Adam_update(double* x, double dx, double lr, double beta1, double beta2, double* m, double* v, int iter);
void Adam_update_vector(Vector* V, const Vector* dV, double lr, double beta1, double beta2, Vector* m, Vector* v, int iter);
void Adam_update_matrix(Matrix* A, const Matrix* dA, double lr, double beta1, double beta2, Matrix* m, Matrix* v, int iter);
void Adam_update_matrix_4d(Matrix4d* A, const Matrix4d* dA, double lr, double beta1, double beta2, Matrix4d* m, Matrix4d* v, int iter);

#endif
