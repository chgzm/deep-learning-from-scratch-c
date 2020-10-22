#include "optimizer.h"

#include <math.h>

void SGD_update(double* x, double dx, double lr) {
    *x -= lr * dx;
}

void Momentum_update(double* x, double dx, double lr, double momentum, double* v) {
    *v = momentum * (*v) - lr * dx;
    *x += *v;
}

void AdaGrad_update(double* x, double dx, double lr, double* h) {
    *h += dx * dx;
    *x -= lr * dx / (sqrt(*h) + 1e-7);
}

void Adam_update(double* x, double dx, double lr, double beta1, double beta2, double* m, double* v, int iter) {
    const double lr_t = lr * sqrt(1.0 - pow(beta2, iter + 1)) / (1.0 - pow(beta1, iter + 1)); 

    *m += (1 - beta1) * (dx - *m);
    *v += (1 - beta2) * (pow(dx, 2) - *v);

    *x -= lr_t * (*m) / (sqrt(*v) + 1e-7);
} 

