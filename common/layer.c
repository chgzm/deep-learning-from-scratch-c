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

BatchNormalization* create_batch_normalization(Vector* g, Vector* b, double momentum) {
    BatchNormalization* B = malloc(sizeof(BatchNormalization));
    B->g  = g;
    B->b  = b;

    B->dg  = NULL;
    B->db  = NULL;
    B->xc  = NULL;
    B->xn  = NULL;
    B->std = NULL;
    B->running_mean = NULL;
    B->running_var  = NULL;
    B->momentum = momentum;

    return B;
}

Matrix* batch_normalization_forward(BatchNormalization* B, const Matrix* X) {
    if (B->xc != NULL) {
        free_matrix(B->xc);
        free_matrix(B->xn);
        free_vector(B->std);
        free_vector(B->running_mean);
        free_vector(B->running_var);
    }
    B->batch_size = X->rows;

    Vector* mu = matrix_col_mean(X);
    B->xc = matrix_sub_vector(X, mu); 
    Matrix* xc_tmp = pow_matrix(B->xc, 2);
    Vector* var = matrix_col_mean(xc_tmp);
    Vector* var_tmp = vector_add_scalar(var, 10e-7);
    B->std = sqrt_vector(var_tmp);
    B->xn = matrix_div_vector(B->xc, B->std);

    Vector* prev_running_mean = B->running_mean;
    Vector* prev_running_var = B->running_var;

    scalar_vector(prev_running_mean, B->momentum);
    scalar_vector(mu, 1.0 - B->momentum);
    B->running_mean = add_vector(prev_running_var, mu);

    scalar_vector(prev_running_var, B->momentum);
    scalar_vector(var, 1.0 - B->momentum);
    B->running_var = add_vector(prev_running_var, var);
   
    Matrix* l = product_vector_matrix(B->g, B->xn);
    Matrix* out = matrix_add_vector(l, B->b);

    free_vector(mu);
    free_vector(var);
    free_vector(var_tmp);
    free_vector(prev_running_mean);
    free_vector(prev_running_var);
    free_matrix(xc_tmp);
    free_matrix(l);

    return out;
}

Matrix* batch_normalization_backward(BatchNormalization* B, const Matrix* D) {
    // dbeta
    Vector* dbeta = matrix_col_sum(D);

    // dgamma  
    Matrix* tmp = product_matrix(B->xn, D);
    Vector* dgamma = matrix_col_sum(tmp);

    // dxn
    Matrix* dxn = product_vector_matrix(B->g, D);

    // dxc
    Matrix* dxc = matrix_div_vector(dxn, B->std);
    
    // dstd 
    Matrix* tmp2 = product_matrix(dxn, B->xc);
    Vector* tmp3 = product_vector(B->std, B->std);
    Matrix* tmp4 = matrix_div_vector(tmp2, tmp3);
    Vector* dstd = matrix_col_sum(tmp4);
    scalar_vector(dstd, -1);

    // dvar
    Vector* tmp5 = create_vector(dstd->size);
    copy_vector(tmp5, dstd);
    scalar_vector(tmp5, 0.5);
    Vector* dvar = vector_div_vector(tmp5, B->std);
    
    // dxc
    Matrix* tmp6 = create_matrix(B->xc->rows, B->xc->cols);
    copy_matrix(tmp6, B->xc);
    scalar_matrix(tmp6, 2.0 / B->batch_size);
    Matrix* tmp7 = product_vector_matrix(dvar, tmp6);
    Matrix* _dxc = matrix_add_matrix(dxc, tmp7);

    // dmu
    Vector* dmu = matrix_col_sum(_dxc); 

    // dx
    Matrix* dx = matrix_sub_vector(_dxc, dmu);
    scalar_matrix(dx, 1.0 / B->batch_size);

    B->dg = dgamma;
    B->db = dbeta;

    free_vector(tmp3);
    free_vector(dstd);
    free_vector(tmp5);
    free_vector(dvar);
    free_vector(dmu); 

    free_matrix(tmp);
    free_matrix(dxn);
    free_matrix(dxc);
    free_matrix(tmp2);
    free_matrix(tmp4);
    free_matrix(tmp6);
    free_matrix(tmp7);
    free_matrix(_dxc);

    return dx;
}
