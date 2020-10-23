#include "optimizer.h"

#include <math.h>

//
// SGD
//

void SGD_update(double* x, double dx, double lr) {
    *x -= lr * dx;
}

void SGD_update_vector(Vector* V, const Vector* dV, double lr) {
    for (int i = 0; i < V->size; ++i) {
        SGD_update(&(V->elements[i]), dV->elements[i], lr);
    }
}

void SGD_update_matrix(Matrix* A, const Matrix* dA, double lr) {
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            SGD_update(&(A->elements[i][j]), dA->elements[i][j], lr);
        }
    }
}

//
// Momentum
//

void Momentum_update(double* x, double dx, double lr, double momentum, double* v) {
    *v = momentum * (*v) - lr * dx;
    *x += *v;
}

void Momentum_update_vector(Vector* V, const Vector* dV, double lr, double momentum, Vector* v) {
    for (int i = 0; i < V->size; ++i) {
        Momentum_update(&(V->elements[i]), dV->elements[i], lr, momentum, &(v->elements[i]));
    }
}

void Momentum_update_matrix(Matrix* A, const Matrix* dA, double lr, double momentum, Matrix* v) {
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            Momentum_update(&(A->elements[i][j]), dA->elements[i][j], lr, momentum, &(v->elements[i][j]));
        }
    } 
}

//
// AdaGrad
//

void AdaGrad_update(double* x, double dx, double lr, double* h) {
    *h += dx * dx;
    *x -= lr * dx / (sqrt(*h) + 1e-7);
}

void AdaGrad_update_vector(Vector* V, const Vector* dV, double lr, Vector* h) {
    for (int i = 0; i < V->size; ++i) {
        AdaGrad_update(&(V->elements[i]), dV->elements[i], lr, &(h->elements[i]));
    }
}

void AdaGrad_update_matrix(Matrix* A, const Matrix* dA, double lr, Matrix* h) {
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            AdaGrad_update(&(A->elements[i][j]), dA->elements[i][j], lr, &(h->elements[i][j]));
        }
    } 
}

//
// Adam
//

void Adam_update(double* x, double dx, double lr, double beta1, double beta2, double* m, double* v, int iter) {
    const double lr_t = lr * sqrt(1.0 - pow(beta2, iter + 1)) / (1.0 - pow(beta1, iter + 1)); 

    *m += (1 - beta1) * (dx - *m);
    *v += (1 - beta2) * (pow(dx, 2) - *v);

    *x -= lr_t * (*m) / (sqrt(*v) + 1e-7);
}

void Adam_update_vector(Vector* V, const Vector* dV, double lr, double beta1, double beta2, Vector* m, Vector* v, int iter) {
    for (int i = 0; i < V->size; ++i) {
        Adam_update(&(V->elements[i]), dV->elements[i], lr, beta1, beta2, &(m->elements[i]), &(v->elements[i]), iter); 
    }
}

void Adam_update_matrix(Matrix* A, const Matrix* dA, double lr, double beta1, double beta2, Matrix* m, Matrix* v, int iter) {
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            Adam_update(&(A->elements[i][j]), dA->elements[i][j], lr, beta1, beta2, &(m->elements[i][j]), &(v->elements[i][j]), iter); 
        }
    } 
}
