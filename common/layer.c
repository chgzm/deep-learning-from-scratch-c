#include "layer.h"
#include "function.h"
#include <stdlib.h>
#include <math.h>

Affine* create_affine(Matrix* W, Vector* b) {
    Affine* A = malloc(sizeof(Affine));
    A->W  = W;
    A->b  = b;
    A->X  = NULL;
    A->dW = NULL;
    A->db = NULL;
    return A;
}

Matrix* affine_forward(Affine* A, const Matrix* X) { 
    if (A->X != NULL) {
        free_matrix(A->X);
    }
    A->X = create_matrix(X->rows, X->cols);
    copy_matrix(A->X, X);

    Matrix* B = dot_matrix(X, A->W);
    for (int i = 0; i < B->rows; ++i) {
        for (int j = 0; j < B->cols; ++j) {
            B->elements[i][j] += A->b->elements[j];
        }
    }
    return B;
}

Matrix* affine_backward(Affine* A, const Matrix* D) { 
    Matrix* W_T = transpose(A->W);
    Matrix* X_T = transpose(A->X);

    Matrix* dX = dot_matrix(D, W_T);
    if (A->dW != NULL) {
        free_matrix(A->dW);
    }

    A->dW = dot_matrix(X_T, D);

    if (A->db != NULL) {
        free_vector(A->db);
    }
    A->db = create_vector(D->cols);

    for (int i = 0; i < D->cols; ++i) {
        double sum = 0;
        for (int j = 0; j < D->rows; ++j) {
            sum += D->elements[j][i];
        }
        A->db->elements[i] = sum;
    }

    free_matrix(W_T);
    free_matrix(X_T);
    return dX;
}

static Mask* create_mask(int rows, int cols) {
    Mask* m     = malloc(sizeof(Mask));
    m->rows     = rows;
    m->cols     = cols;
    m->elements = calloc(rows, sizeof(bool*));
    for (int i = 0; i < rows; ++i) {
        m->elements[i] = calloc(cols, sizeof(bool));
    }

    return m;
}

static void free_mask(Mask* m) {
    for (int i = 0; i < m->rows; ++i) {
        free(m->elements[i]);
    }
    free(m->elements);
    free(m);
}

Relu* create_relu(int rows, int cols) {
    Relu* r = malloc(sizeof(Relu));
    r->mask = NULL;
    return r;
}

Matrix* relu_forward(Relu* R, const Matrix* X) {
    if (R->mask != NULL) {
        free_mask(R->mask);
    }
    R->mask = create_mask(X->rows, X->cols);
    Matrix* M = create_matrix(X->rows, X->cols);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            if (X->elements[i][j] <= 0) {
                R->mask->elements[i][j] = true;
                M->elements[i][j] = 0;
            } else {
                R->mask->elements[i][j] = false;
                M->elements[i][j] = X->elements[i][j];
            }
        }
    }

    return M;
}

Matrix* relu_backward(Relu* R, const Matrix* D) {
    Matrix* M = create_matrix(D->rows, D->cols);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            if (R->mask->elements[i][j]) {
                M->elements[i][j] = 0;
            } else {
                M->elements[i][j] = D->elements[i][j];
            }
        }
    }

    return M;
}

SoftmaxWithLoss* create_softmax_with_loss() {
    SoftmaxWithLoss* sft = calloc(1, sizeof(SoftmaxWithLoss));
    return sft;
}

static double cross_entropy_error(const Matrix* Y, const Vector* t) {
    static const double delta = 1e-7;
   
    double sum = 0.0;
    for (int i = 0; i < Y->rows; ++i) {
        for (int j = 0; j < Y->cols; ++j) {
            if (j != t->elements[i]) {
                continue;
            }

            double d = Y->elements[i][j] + delta;
            sum += log(d);
        }
    }

    return -1.0 * sum / Y->rows;
}

double softmax_with_loss_forward(SoftmaxWithLoss* sft, const Matrix* X, const Vector* t) {
    if (sft->t != NULL) {
        free_vector(sft->t);
    }
    sft->t = create_vector(t->size);
    copy_vector(sft->t, t);

    if (sft->Y != NULL) {
        free_matrix(sft->Y);
    }
    sft->Y = matrix_softmax(X);

    return cross_entropy_error(sft->Y, t);
}

Matrix* softmax_with_loss_backward(const SoftmaxWithLoss* sft) {
    Matrix* dX = create_matrix(sft->Y->rows, sft->Y->cols);
    for (int i = 0; i < dX->rows; ++i) {
        for (int j = 0; j < dX->cols; ++j) {
            dX->elements[i][j] = sft->Y->elements[i][j];
            if (j == sft->t->elements[i]) {
                dX->elements[i][j] -= 1.0;
            }
            dX->elements[i][j] /= sft->t->size;
        }
    }

    return dX;
}
