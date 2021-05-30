#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <util.h>
#include <matrix.h>
#include <function.h>
#include "optimizer.h"

static double df_x(double x) {
    return x / 10.0;
}

static double df_y(double y) {
    return 2.0 * y;
}

static void run_SGD() {
    const double lr = 0.95;
    double x[30], y[30];
    double param_x = -7.0, param_y = 2.0;
    double grads_x = 0.0, grads_y = 0.0;

    FILE* fp = fopen("naive_SGD.txt", "w");
    if (!fp) {
        fprintf(stderr, "Failed to open file.\n");
        return;
    }

    for (int i = 0; i < 30; ++i) {
        x[i] = param_x;
        y[i] = param_y;
        fprintf(fp, "%lf %lf\n", x[i], y[i]);
        
        grads_x = df_x(x[i]);
        grads_y = df_y(y[i]);

        SGD_update(&param_x, grads_x, lr);
        SGD_update(&param_y, grads_y, lr);
    }

    fclose(fp);
}

static void run_Momentum() {
    const double lr = 0.1;
    const double momentum = 0.9;

    double x[30], y[30];
    double param_x = -7.0, param_y = 2.0;
    double grads_x = 0.0, grads_y = 0.0;
    double v_x = 0.0, v_y = 0.0;

    FILE* fp = fopen("naive_Momentum.txt", "w");
    if (!fp) {
        fprintf(stderr, "Failed to open file.\n");
        return;
    }

    for (int i = 0; i < 30; ++i) {
        x[i] = param_x;
        y[i] = param_y;
        fprintf(fp, "%lf %lf\n", x[i], y[i]);
        
        grads_x = df_x(x[i]);
        grads_y = df_y(y[i]);

        Momentum_update(&param_x, grads_x, lr, momentum, &v_x);
        Momentum_update(&param_y, grads_y, lr, momentum, &v_y);
    }

    fclose(fp);
}

static void run_AdaGrad() {
    const double lr = 1.5;
    double x[30], y[30];
    double param_x = -7.0, param_y = 2.0;
    double grads_x = 0.0, grads_y = 0.0;
    double h_x = 0.0, h_y = 0.0;

    FILE* fp = fopen("naive_AdaGrad.txt", "w");
    if (!fp) {
        fprintf(stderr, "Failed to open file.\n");
        return;
    }

    for (int i = 0; i < 30; ++i) {
        x[i] = param_x;
        y[i] = param_y;
        fprintf(fp, "%lf %lf\n", x[i], y[i]);
        
        grads_x = df_x(x[i]);
        grads_y = df_y(y[i]);

        AdaGrad_update(&param_x, grads_x, lr, &h_x);
        AdaGrad_update(&param_y, grads_y, lr, &h_y);
    }

    fclose(fp);
}

static void run_Adam() {
    const double lr = 0.3, beta1 = 0.9, beta2 = 0.999;
    double x[30], y[30];
    double param_x = -7.0, param_y = 2.0;
    double grads_x = 0.0, grads_y = 0.0;
    double m_x = 0.0, m_y = 0.0;
    double v_x = 0.0, v_y = 0.0;

    FILE* fp = fopen("naive_Adam.txt", "w");
    if (!fp) {
        fprintf(stderr, "Failed to open file.\n");
        return;
    }

    for (int i = 0; i < 30; ++i) {
        x[i] = param_x;
        y[i] = param_y;
        fprintf(fp, "%lf %lf\n", x[i], y[i]);
        
        grads_x = df_x(x[i]);
        grads_y = df_y(y[i]);

        Adam_update(&param_x, grads_x, lr, beta1, beta2, &m_x, &v_x, i);
        Adam_update(&param_y, grads_y, lr, beta1, beta2, &m_y, &v_y, i);
    }

    fclose(fp);
}

int main() {
    run_SGD();
    run_Momentum();
    run_AdaGrad();
    run_Adam();
    plot_gpfile("plot_optimization_naive.gp");

    return 0;   
}
