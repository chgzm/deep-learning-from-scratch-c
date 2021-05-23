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

void free_affine(Affine* A) {
    free_matrix(A->W); 
    free_vector(A->b); 
    free_matrix(A->X);
    free_matrix(A->dW);
    free_vector(A->db);
    free(A);
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

Relu* create_relu() {
    Relu* r = malloc(sizeof(Relu));
    r->mask = NULL;
    return r;
}

void free_relu(Relu* R) {
    free_mask(R->mask);
    free(R);
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

//
// Relu4d
//

static Mask4d* create_mask_4d(const int* sizes) {
    Mask4d* m   = malloc(sizeof(Mask4d));
    m->sizes[0] = sizes[0];
    m->sizes[1] = sizes[1];
    m->sizes[2] = sizes[2];
    m->sizes[3] = sizes[3];
    m->elements = calloc(sizes[0], sizeof(bool***));
    for (int i = 0; i < sizes[0]; ++i) {
        m->elements[i] = calloc(sizes[1], sizeof(bool**));
        for (int j = 0; j < sizes[1]; ++j) {
            m->elements[i][j] = calloc(sizes[2], sizeof(bool*));
            for (int k = 0; k < sizes[2]; ++k) {
                m->elements[i][j][k] = calloc(sizes[3], sizeof(bool));
            }
        }
    }

    return m;
}

static void free_mask_4d(Mask4d* m) {
    for (int i = 0; i < m->sizes[0]; ++i) {
        for (int j = 0; j < m->sizes[1]; ++j) {
            for (int k = 0; k < m->sizes[2]; ++k) {
                free(m->elements[i][j][k]);
            }
            free(m->elements[i][j]);
        }
        free(m->elements[i]);
    }
    free(m->elements);
    free(m);
}

Relu4d* create_relu_4d() {
    Relu4d* r = malloc(sizeof(Relu4d));
    r->mask = NULL;
    return r;
}

void free_relu_4d(Relu4d* R) {
    free_mask_4d(R->mask);
    free(R);
}

Matrix4d* relu_4d_forward(Relu4d* R, const Matrix4d* X) {
    if (R->mask != NULL) {
        free_mask_4d(R->mask);
    }
    R->mask = create_mask_4d(X->sizes);
    Matrix4d* M = create_matrix_4d(X->sizes[0], X->sizes[1], X->sizes[2], X->sizes[3]);
    for (int i = 0; i < M->sizes[0]; ++i) {
        for (int j = 0; j < M->sizes[1]; ++j) {
            for (int k = 0; k < M->sizes[2]; ++k) {
                for (int l = 0; l < M->sizes[3]; ++l) {
                    if (X->elements[i][j][k][l] <= 0) {
                        R->mask->elements[i][j][k][l] = true;
                        M->elements[i][j][k][l] = 0;
                    } else {
                        R->mask->elements[i][j][k][l] = false;
                        M->elements[i][j][k][l] = X->elements[i][j][k][l];
                    }
                }
            }
        }
    }

    return M;
}

Matrix4d* relu_4d_backward(Relu4d* R, const Matrix4d* D) {
    Matrix4d* M = create_matrix_4d(D->sizes[0], D->sizes[1], D->sizes[2], D->sizes[3]);
    for (int i = 0; i < M->sizes[0]; ++i) {
        for (int j = 0; j < M->sizes[1]; ++j) {
            for (int k = 0; k < M->sizes[2]; ++k) {
                for (int l = 0; l < M->sizes[3]; ++l) {
                    if (R->mask->elements[i][j][k][l]) {
                        M->elements[i][j][k][l] = 0;
                    } else {
                        M->elements[i][j][k][l] = D->elements[i][j][k][l];
                    }
                }
            }
        }
    }

    return M;
}

//
// SoftmaxWithLoss
//

SoftmaxWithLoss* create_softmax_with_loss() {
    SoftmaxWithLoss* sft = calloc(1, sizeof(SoftmaxWithLoss));
    return sft;
}

void free_softmax_with_loss(SoftmaxWithLoss* S) {
    free_matrix(S->Y);
    free_vector(S->t);
    free(S);
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

void free_batch_normalization(BatchNormalization* B) {
    free_vector(B->g);
    free_vector(B->b);
    free_vector(B->dg);
    free_vector(B->db);

    free_matrix(B->xc);
    free_matrix(B->xn);
    free_vector(B->std);
    free_vector(B->running_mean);
    free_vector(B->running_var);

    free(B);
}

Matrix* batch_normalization_forward(BatchNormalization* B, const Matrix* X) {
    if (B->xc != NULL) {
        free_matrix(B->xc);
        free_matrix(B->xn);
        free_vector(B->std);
    }

    if (B->running_mean == NULL) {
        B->running_mean = create_vector(B->g->size);
        B->running_var  = create_vector(B->g->size);
    }
    B->batch_size = X->rows;

    // mu
    Vector* mu = matrix_col_mean(X);

    // xc
    B->xc = matrix_sub_vector(X, mu); 

    // var 
    Matrix* xc_tmp = pow_matrix(B->xc, 2);
    Vector* var = matrix_col_mean(xc_tmp);

    // std 
    Vector* var_tmp = vector_add_scalar(var, 10e-7);
    B->std = sqrt_vector(var_tmp);

    // xn 
    B->xn = matrix_div_vector(B->xc, B->std);

    Vector* prev_running_mean = B->running_mean;
    Vector* prev_running_var = B->running_var;
 
    // running_mean
    scalar_vector(prev_running_mean, B->momentum);
    scalar_vector(mu, 1.0 - B->momentum);
    B->running_mean = add_vector(prev_running_mean, mu);

    // running_var
    scalar_vector(prev_running_var, B->momentum);
    scalar_vector(var, 1.0 - B->momentum);
    B->running_var = add_vector(prev_running_var, var);
   
    // out
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
    scalar_vector(dmu, 1.0 / B->batch_size);
    Matrix* dx = matrix_sub_vector(_dxc, dmu);

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

Dropout* create_dropout(double dropout_ratio) {
    Dropout* d = malloc(sizeof(Dropout));
    d->dropout_ratio = dropout_ratio;
    d->mask = NULL;
    return d;
}

void free_dropout(Dropout* D) {
    free_mask(D->mask);
    free(D);
}

Matrix* dropout_forward(Dropout* D, const Matrix* X) {
    if (D->mask != NULL) {
        free_mask(D->mask);
    }

    Matrix* RND = create_matrix(X->rows, X->cols);
    init_matrix_rand(RND);
    Matrix* M = create_matrix(X->rows, X->cols);
    D->mask = create_mask(X->rows, X->cols);

    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            if (RND->elements[i][j] > D->dropout_ratio) {
                D->mask->elements[i][j] = true;
                M->elements[i][j] = X->elements[i][j];
            } else {
                D->mask->elements[i][j] = false;
                M->elements[i][j] = 0;
            }
        }
    }

    free_matrix(RND);

    return M;
}

Matrix* dropout_backward(const Dropout* D, const Matrix* X) {
    Matrix* M = create_matrix(X->rows, X->cols);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            if (D->mask->elements[i][j]) {
                M->elements[i][j] = X->elements[i][j];
            } else {
                M->elements[i][j] = 0;
            }
        }
    }

    return M;
}

Convolution* create_convolution(Matrix4d* W, Vector* b, int stride, int pad) {
    Convolution* Conv = malloc(sizeof(Convolution));
    Conv->W = W;
    Conv->b = b;
    Conv->stride = stride;
    Conv->pad = pad;

    Conv->x = NULL;
    Conv->col = NULL;
    Conv->col_W = NULL;
    Conv->db = NULL;
    Conv->dW = NULL;

    return Conv;
}

void free_convolution(Convolution* C) {
    free_matrix_4d(C->W);
    free_vector(C->b);
    free_matrix_4d(C->x);
    free_matrix(C->col);
    free_matrix(C->col_W);
}

Matrix4d* convolution_forward(Convolution* Conv, Matrix4d* X) {
    const int FN = Conv->W->sizes[0];
    // const int C  = Conv->W->sizes[1];
    const int FH = Conv->W->sizes[2];
    const int FW = Conv->W->sizes[3];
    
    const int N  = X->sizes[0];
    // const int C2 = X->sizes[1];
    const int H  = X->sizes[2];
    const int W  = X->sizes[3];

    const int out_h = 1 + (int)((H + 2 * Conv->pad - FH) / Conv->stride);
    const int out_w = 1 + (int)((W + 2 * Conv->pad - FW) / Conv->stride);

    Matrix* col = im2col(X, FH, FW, Conv->stride, Conv->pad);
    Matrix* tmp2 = matrix_reshape_to_2d(Conv->W, FN, -1);
    Matrix* col_W = transpose(tmp2);

    Matrix* tmp = dot_matrix(col, col_W);
    Matrix* out = matrix_add_vector(tmp, Conv->b);

    Matrix4d* out_r = matrix_reshape_to_4d(out, N, out_h, out_w, -1);
    Matrix4d* out_rt = matrix_4d_transpose(out_r, 0, 3, 1, 2);

    free_matrix(tmp);
    free_matrix(tmp2);
    free_matrix(out);
    free_matrix_4d(out_r);

    if (Conv->x != NULL) {
        free_matrix_4d(Conv->x);
    }
    Conv->x = X;

    if (Conv->col != NULL) {
        free_matrix(Conv->col);
    }
    Conv->col = col;

    if (Conv->col_W != NULL) {
        free_matrix(Conv->col_W);
    }
    Conv->col_W = col_W;

    return out_rt;
}

Matrix4d* convolution_backward(Convolution* Conv, const Matrix4d* X) {
    const int FN = Conv->W->sizes[0];
    const int C  = Conv->W->sizes[1];
    const int FH = Conv->W->sizes[2];
    const int FW = Conv->W->sizes[3];
    
    Matrix4d* tmp = matrix_4d_transpose(X, 0, 2, 3, 1);  
    Matrix* dout = matrix_reshape_to_2d(tmp, -1, FN);

    // db
    Conv->db = matrix_col_sum(dout);

    // dW
    Matrix* col_T = transpose(Conv->col);
    Matrix* dW = dot_matrix(col_T, dout);
    Matrix* dW_T = transpose(dW);
    Conv->dW = matrix_reshape_to_4d(dW_T, FN, C, FH, FW);

    // dcol
    Matrix* col_W_T = transpose(Conv->col_W);
    Matrix* dcol = dot_matrix(dout, col_W_T);

    Matrix4d* dx = col2im(dcol, Conv->x->sizes, FH, FW, Conv->stride, Conv->pad);  

    return dx;
}

Pooling* create_pooling(int pool_h, int pool_w, int stride, int pad) {
    Pooling* P = malloc(sizeof(Pooling));
    P->pool_h  = pool_h;
    P->pool_w  = pool_w;
    P->stride  = stride;
    P->pad     = pad;
    P->x       = NULL;
    P->arg_max = NULL;
    return P;
}

void free_pooling(Pooling* P) {
    free_matrix_4d(P->x);
    free(P->arg_max);
    free(P);
}

Matrix4d* pooling_forward(Pooling* P, Matrix4d* X) {
    const int N  = X->sizes[0];
    const int C  = X->sizes[1];
    const int H  = X->sizes[2];
    const int W  = X->sizes[3];

    const int out_h = 1 + (H - P->pool_h) / P->stride;
    const int out_w = 1 + (W - P->pool_w) / P->stride;

    Matrix* tmp = im2col(X, P->pool_h, P->pool_w, P->stride, P->pad);
    Matrix* col = matrix_reshape(tmp, -1, P->pool_h * P->pool_w);

    if (P->arg_max != NULL) {
        free(P->arg_max);
    }
    P->arg_max = argmax_row(col); 

    Vector* tmp2 = matrix_row_max(col);

    Matrix4d* tmp3 = vector_reshape_to_4d(tmp2, N, out_h, out_w, C);
    Matrix4d* out = matrix_4d_transpose(tmp3, 0, 3, 1, 2);
   
    if (P->x != NULL) {
        free(P->x);
    }
    P->x = X;

    free_matrix(tmp);
    free_matrix(col);
    free_vector(tmp2);
    free_matrix_4d(tmp3);

    return out;
}

Matrix4d* pooling_backward(const Pooling* P, const Matrix4d* X) {
    Matrix4d* dout = matrix_4d_transpose(X, 0, 2, 3, 1);

    const int pool_size = P->pool_h * P->pool_w;
    const int dout_size = dout->sizes[0] * dout->sizes[1] * dout->sizes[2] * dout->sizes[3];
    Matrix* dmax = create_matrix(dout_size, pool_size); 

    Vector* dout_flat = matrix_4d_flatten(dout);
    for (int i = 0; i < dmax->rows; ++i) {
        dmax->elements[i][P->arg_max[i]] = dout_flat->elements[i];
    }

    Matrix* dcol = create_matrix(dout->sizes[0] * dout->sizes[1] * dout->sizes[2], dout->sizes[3] * pool_size);
    int r_pos = 0, c_pos = 0;
    for (int i = 0; i < dmax->rows; ++i) {
        for (int j = 0; j < dmax->cols; ++j) {
            dcol->elements[r_pos][c_pos++] = dmax->elements[i][j];
            if (c_pos == dcol->cols) {
                c_pos = 0;
                ++r_pos;
            }
        }
    }

    int sizes[4] = {P->x->sizes[0], P->x->sizes[1], P->x->sizes[2], P->x->sizes[3]};
    Matrix4d* dx = col2im(dcol, sizes, P->pool_h, P->pool_w, P->stride, P->pad);

    free_matrix_4d(dout); 
    free_matrix(dmax); 
    free_vector(dout_flat); 
    free_matrix(dcol); 

    return dx;
}
