#include "function.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

Vector* vector_sigmoid(const Vector* v) {
    Vector* r = create_vector(v->size);
    for (int i = 0; i < r->size; ++i) {
        r->elements[i] = sigmoid(v->elements[i]);
    }

    return r;
}

Matrix* matrix_sigmoid(const Matrix* M) {
    Matrix* R = create_matrix(M->rows, M->cols);
    
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            R->elements[i][j] = sigmoid(M->elements[i][j]);
        }
    }

    return R;
}

Matrix* sigmoid_grad(const Matrix* M) {
    Matrix* R = create_matrix(M->rows, M->cols);
    
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            R->elements[i][j] = (1.0 - sigmoid(M->elements[i][j])) * sigmoid(M->elements[i][j]);
        }
    }

    return R;
}

Vector* vector_softmax(const Vector* v) {
    Vector* r = create_vector(v->size);

    double max = 0.0;
    for (int i = 0; i < v->size; ++i) {
        max = ((max > v->elements[i]) ? max : v->elements[i]);
    }   

    double sum = 0.0;
    for (int i = 0; i < v->size; ++i) {
        sum += exp(v->elements[i] - max);
    }   

    for (int i = 0; i < v->size; ++i) {
        r->elements[i] = exp(v->elements[i] - max) / sum;
    }   

    return r;
}

Matrix* matrix_softmax(const Matrix* M) {
    Matrix* N = create_matrix(M->rows, M->cols);

    Vector* max_vals = create_vector(M->rows);
    for (int i = 0; i < M->rows; ++i) {
        double max = M->elements[i][0];
        for (int j = 1; j < M->cols; ++j) {
            if (max < M->elements[i][j]) {
                max =  M->elements[i][j];
            }
        }
        max_vals->elements[i] = max;
    }

    for (int i = 0; i < M->rows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < M->cols; ++j) {
            N->elements[i][j] = exp(M->elements[i][j] - max_vals->elements[i]);
            sum += N->elements[i][j];
        }

        for (int j = 0; j < M->cols; ++j) {
            N->elements[i][j] /= sum;
        }
    }

    free_vector(max_vals);
    return N;
}

int argmax(const double* v, int size) {
    int index = 0;
    double max = v[0];
    for (int i = 1; i < size; ++i) {
        if (max < v[i]) {
            index = i;
            max = v[i];
        }
    }

    return index;
}

int vector_argmax(const Vector* v) {
    int index = 0;
    double max = v->elements[0];
    for (int i = 1; i < v->size; ++i) {
        if (max < v->elements[i]) {
            index = i;
            max = v->elements[i];
        }
    }

    return index;
}

int* matrix_argmax_row(const Matrix* M) {
    int* arg_max = (int*)malloc(sizeof(int) * M->rows);

    for (int i = 0; i < M->rows; ++i) {
        int idx = 0;
        double max = - DBL_MAX;
        for (int j = 0; j < M->cols; ++j) {
            if (max < M->elements[i][j]) {
                idx = j;
                max = M->elements[i][j];
            }
        }

        arg_max[i] = idx;
    }

    return arg_max;
}
