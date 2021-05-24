#include "matrix.h" 
#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <float.h>

//
// factory
//
Vector* create_vector(int size) {
    Vector* v = calloc(1, sizeof(Vector));
    v->size = size;
    v->elements = calloc(size, sizeof(double));
    return v;
}

Vector* create_vector_initval(int size, double init_val) {
    Vector* v = calloc(1, sizeof(Vector));
    v->size = size;
    v->elements = calloc(size, sizeof(double));

    for (int i = 0; i < size; ++i) {
        v->elements[i] = init_val;
    }

    return v;
}

Matrix* create_matrix(int rows, int cols) {
    Matrix* M = malloc(sizeof(Matrix));
    M->rows = rows;
    M->cols = cols;
    M->elements = calloc(rows, sizeof(double*));
    for (int i = 0; i < rows; ++i) {
        M->elements[i] = calloc(cols, sizeof(double));
    }

    return M;
}

Matrix4d* create_matrix_4d(int s1, int s2, int s3, int s4) {
    Matrix4d* M = malloc(sizeof(Matrix4d));
    M->sizes[0] = s1;
    M->sizes[1] = s2;
    M->sizes[2] = s3;
    M->sizes[3] = s4;

    M->elements = calloc(s1, sizeof(double***));
    for (int i = 0; i < s1; ++i) {
        M->elements[i] = calloc(s2, sizeof(double**));
        for (int j = 0; j < s2; ++j) {
            M->elements[i][j] = calloc(s3, sizeof(double*));
            for (int k = 0; k < s3; ++k) {
                M->elements[i][j][k] = calloc(s4, sizeof(double));
            }
        }
    }

    return M;
}

//
// init
//

int init_vector_from_file(Vector* v, const char* file_path) {
    FILE* fp = fopen(file_path, "r");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open \"%s\".\n", file_path);
        return -1;
    }    

    double d;
    for (int i = 0; i < v->size; ++i) {
        const int r = fscanf(fp, "%lf", &d);
        if (r != 1) {
            fprintf(stderr, "fscanf failed.\n");
            fclose(fp);
            return -1;
        }
        v->elements[i] = d; 
    }

    fclose(fp);
    return 0;
} 

int init_matrix_from_file(Matrix* M, const char* file_path) {
    FILE* fp = fopen(file_path, "r");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open \"%s\".\n", file_path);
        return -1;
    }    

    double d;
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            const int r = fscanf(fp, "%lf", &d);
            if (r != 1) {
                fprintf(stderr, "fscanf failed.\n");
                fclose(fp);
                return -1;
            }
            M->elements[i][j] = d;
        }
    }

    fclose(fp);
    return 0;
}

void copy_matrix(Matrix* dst, const Matrix* src) {
    for (int i = 0; i < src->rows; ++i) {
        for (int j = 0; j < src->cols; ++j) {
            dst->elements[i][j] = src->elements[i][j];
        }
    }
}

void copy_vector(Vector* dst, const Vector* src) {
    for (int i = 0; i < src->size; ++i) {
        dst->elements[i] = src->elements[i];
    }
}

static double rand_normal() {
    double r1 = ((double)rand() + 0.5) / ((double)RAND_MAX + 1);
    double r2 = ((double)rand() + 0.5) / ((double)RAND_MAX + 1);
    return sqrt(-2.0 * log(r1)) * sin(2.0 * M_PI * r2);
}

void init_matrix_random(Matrix* M) {
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            M->elements[i][j] = rand_normal();
        }
    }
}

void init_matrix_4d_random(Matrix4d* M) {
    for (int i = 0; i < M->sizes[0]; ++i) {
        for (int j = 0; j < M->sizes[1]; ++j) {
            for (int k = 0; k < M->sizes[2]; ++k) {
                for (int l = 0; l < M->sizes[3]; ++l) {
                    M->elements[i][j][k][l] = rand_normal();
                }
            }
        }
    }
}

void init_matrix_rand(Matrix* M) {
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            M->elements[i][j] = (double)rand() / (double)RAND_MAX;
        }
    }
}

void init_vector_from_array(Vector* v, double* vals) {
    for (int i = 0; i < v->size; ++i) {
        v->elements[i] = vals[i];
    }
}

//
// free
//

void free_vector(Vector* v) {
    free(v->elements);
    free(v);
}

void free_matrix(Matrix* M) {
    for (int i = 0; i < M->rows; ++i) {
        free(M->elements[i]);
    }
    free(M->elements);
    free(M);
}

void free_matrix_4d(Matrix4d* M) {
    for (int i = 0; i < M->sizes[0]; ++i) {
        for (int j = 0; j < M->sizes[1]; ++j) {
            for (int k = 0; k < M->sizes[2]; ++k) {
                free(M->elements[i][j][k]);
            }
            free(M->elements[i][j]);
        }
        free(M->elements[i]);
    }
    free(M->elements);
    free(M);
}

//
// Operator
//

Vector* add_vector(const Vector* a, const Vector* b) {
    if (a->size != b->size) {
        fprintf(stderr, "Invalid size. %d and %d\n", a->size, b->size);
        return NULL;
    }

    Vector* r = create_vector(a->size);
    for (int i = 0; i < r->size; ++i) {
        r->elements[i] = a->elements[i] + b->elements[i];
    }

    return r;
}

Vector* dot_vector_matrix(const Vector* v, const Matrix* M) {
    if (v->size != M->rows) {
        fprintf(stderr, "Invalid size. %d and (%d, %d)\n", v->size, M->rows, M->cols);
        return NULL;
    }

    Vector* r = create_vector(M->cols);
    for (int i = 0; i < r->size; ++i) {
        double d = 0.0;
        for (int j = 0; j < M->rows; ++j) {
            d += (v->elements[j] * M->elements[j][i]);
        }
        r->elements[i] = d;
    }

    return r;
}

Matrix* dot_matrix(const Matrix* M, const Matrix* N) {
    if (M->cols != N->rows) {
        fprintf(stderr, "Invalid size. (%d, %d) and (%d, %d)\n", M->rows, M->cols, N->rows, N->cols);
        return NULL;
    }

    Matrix* A = create_matrix(M->rows, N->cols);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < N->cols; ++j) {
            double d = 0.0;
            for (int k = 0; k < M->cols; ++k) {
                d += M->elements[i][k] * N->elements[k][j];
            }
            A->elements[i][j] = d;
        }
    }

    return A;
}

Matrix* product_vector_matrix(const Vector* V, const Matrix* M) {
    if (V->size != M->cols) {
        fprintf(stderr, "Invalid size. %d and (%d, %d)\n", V->size, M->rows, M->cols);
        return NULL;
    }

    Matrix* A = create_matrix(M->rows, M->cols);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            A->elements[i][j] = M->elements[i][j] * V->elements[j];
        }
    }

    return A;
}

Matrix* product_matrix(const Matrix* M, const Matrix* N) {
    if (!(M->rows == N->rows && M->cols == N->cols)) {
        fprintf(stderr, "Invalid size. (%d, %d) and (%d, %d)\n", M->rows, M->cols, N->rows, N->cols);
        return NULL;
    }
    
    Matrix* A = create_matrix(M->rows, M->cols);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            A->elements[i][j] = M->elements[i][j] * N->elements[i][j];
        }
    }
   
    return A;
}

Vector* product_vector(const Vector* V, const Vector* U) {
    if (V->size != U->size) {
        fprintf(stderr, "Invalid size. %d and %d\n", V->size, U->size);
        return NULL;
    }

    Vector* R = create_vector(V->size);
    for (int i = 0; i < V->size; ++i) {
        R->elements[i] = V->elements[i] * U->elements[i];
    }

    return R;
}

Vector* matrix_col_mean(const Matrix* M) {
    Vector* V = create_vector(M->cols);

    for (int i = 0; i < M->cols; ++i) {
        double sum = 0.0;
        for (int j = 0; j < M->rows; ++j) {
            sum += M->elements[j][i];
        }
        V->elements[i] = sum / M->rows; 
    }

    return V;
}

Vector* matrix_col_sum(const Matrix* M) {
    Vector* v = create_vector(M->cols);

    for (int i = 0; i < M->cols; ++i) {
        double sum = 0.0;
        for (int j = 0; j < M->rows; ++j) {
            sum += M->elements[j][i];
        }
        v->elements[i] = sum;
    }

    return v;
}

Vector* matrix_row_max(const Matrix* M) {
    Vector* v = create_vector(M->rows);

    for (int i = 0; i < M->rows; ++i) {
        double max_val = -DBL_MAX;
        for (int j = 0; j < M->cols; ++j) {
            max_val = fmax(max_val, M->elements[i][j]);
        }
        v->elements[i] = max_val;
    }

    return v;
}

void scalar_matrix(Matrix* M, double k) {
    for (int i = 0; i < M->rows; ++i) {
       for (int j = 0; j < M->cols; ++j) {
            M->elements[i][j] *= k;
       }
    }
}

void scalar_matrix_4d(Matrix4d* M, double v) {
    for (int i = 0; i < M->sizes[0]; ++i) {
        for (int j = 0; j < M->sizes[1]; ++j) {
            for (int k = 0; k < M->sizes[2]; ++k) {
                for (int l = 0; l < M->sizes[3]; ++l) {
                    M->elements[i][j][k][l] *= v;
                }
            }
       }
    }
}

Matrix* _scalar_matrix(const Matrix* M, double k) {
    Matrix* R = create_matrix(M->rows, M->cols);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            R->elements[i][j] = M->elements[i][j] * k;
        }
    }

    return R;
}

void scalar_vector(Vector* V, double k) {
    for (int i = 0; i < V->size; ++i) {
        V->elements[i] *= k;
    }
}

Matrix* transpose(const Matrix* M) {
    Matrix* N = create_matrix(M->cols, M->rows);

    for (int i = 0; i < N->rows; ++i) {
        for (int j = 0; j < N->cols; ++j) {
            N->elements[i][j] = M->elements[j][i];
        }
    }

    return N;
}

Matrix4d* matrix_4d_transpose(const Matrix4d* M, int n1, int n2, int n3, int n4) {
    Matrix4d* R = create_matrix_4d(M->sizes[n1], M->sizes[n2], M->sizes[n3], M->sizes[n4]);

    int idx[4] = {0};
    int n[4] = {n1, n2, n3, n4};
    for (int i = 0; i < M->sizes[0]; ++i) {
        for (int j = 0; j < M->sizes[1]; ++j) {
            for (int k = 0; k < M->sizes[2]; ++k) {
                for (int l = 0; l < M->sizes[3]; ++l) {
                    const int org[] = {i, j, k, l};
                    for (int i = 0; i < 4; ++i) {
                        idx[i] = org[n[i]];
                    }

                    R->elements[idx[0]][idx[1]][idx[2]][idx[3]] = M->elements[i][j][k][l]; 
                }
            }
        }
    }

    return R;
}

Matrix4d* vector_reshape_to_4d(const Vector* v, int s1, int s2, int s3, int s4) {
    int sizes[] = {s1, s2, s3, s4};
    if (s4 < 0) {
        sizes[3] = v->size / (s1 * s2 * s3);
    }

    Matrix4d* R = create_matrix_4d(sizes[0], sizes[1], sizes[2], sizes[3]);
    int p1 = 0, p2 = 0, p3 = 0, p4 = 0;
    for (int i = 0; i < v->size; ++i) {
        R->elements[p1][p2][p3][p4] = v->elements[i];
        ++p4;
        if (p4 == sizes[3]) {
            p4 = 0;
            ++p3;
        }

        if (p3 == sizes[2]) {
            p3 = 0;
            ++p2;
        }

        if (p2 == sizes[1]) {
            p2 = 0;
            ++p1;
        } 
    }

    return R;
}

Matrix* matrix_reshape(const Matrix* M, int rows, int cols) {
    int r = rows;
    int c = cols;
    if (rows < 0) {
        r = (M->rows * M->cols) / cols;
    } else if (cols < 0) {
        c = (M->rows * M->cols) / rows;
    }

    int r_pos = 0, c_pos = 0;
    Matrix* R = create_matrix(r, c);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            R->elements[r_pos][c_pos++] = M->elements[i][j];
            if (c_pos == c) {
                c_pos = 0;
                ++r_pos;
            }
        }
    }

    return R;
}

Matrix* matrix_reshape_to_2d(const Matrix4d* M, int rows, int cols) {
    int r = rows;
    int c = cols;
    if (rows < 0) {
        r = M->sizes[0] * M->sizes[1] * M->sizes[2] * M->sizes[3] / cols;
    } else if (cols < 0) {
        c = M->sizes[0] * M->sizes[1] * M->sizes[2] * M->sizes[3] / rows;
    }

    Matrix* R = create_matrix(r, c);
    int r_pos = 0, c_pos = 0;
    for (int i = 0; i < M->sizes[0]; ++i) {
        for (int j = 0; j < M->sizes[1]; ++j) {
            for (int k = 0; k < M->sizes[2]; ++k) {
                for (int l = 0; l < M->sizes[3]; ++l) {
                    R->elements[r_pos][c_pos++] = M->elements[i][j][k][l];
                    if (c_pos == c) {
                        c_pos = 0;
                        ++r_pos;
                    }
                }
            }
        }
    }

    return R;
}

Matrix4d* matrix_reshape_to_4d(const Matrix* M, int s1, int s2, int s3, int s4) {
    int sizes[] = {s1, s2, s3, s4};
    if (s4 < 0) {
        sizes[3] = M->rows * M->cols / (s1 * s2 * s3);
    }

    Matrix4d* R = create_matrix_4d(sizes[0], sizes[1], sizes[2], sizes[3]);
    int p1 = 0, p2 = 0, p3 = 0, p4 = 0;
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            R->elements[p1][p2][p3][p4] = M->elements[i][j];
            ++p4;
            if (p4 == sizes[3]) {
                p4 = 0;
                ++p3;
            }

            if (p3 == sizes[2]) {
                p3 = 0;
                ++p2;
            }

            if (p2 == sizes[1]) {
                p2 = 0;
                ++p1;
            } 
        }
    }

    return R;
}

Vector* matrix_4d_flatten(const Matrix4d* M) {
    Vector* v = create_vector(M->sizes[0] * M->sizes[1] * M->sizes[2] * M->sizes[3]);

    int pos = 0;
    for (int i = 0; i < M->sizes[0]; ++i) {
        for (int j = 0; j < M->sizes[1]; ++j) {
            for (int k = 0; k < M->sizes[2]; ++k) {
                for (int l = 0; l < M->sizes[3]; ++l) {
                    v->elements[pos++] = M->elements[i][j][k][l];
                }
            }
        }
    }

    return v;
}

double matrix_sum(const Matrix* M) {
    double sum = 0.0;
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            sum += M->elements[i][j];
        }
    }

    return sum;
}

Vector* vector_div_vector(const Vector* v, const Vector* u) {
    if (v->size != u->size) {
        fprintf(stderr, "Invalid size. %d and %d\n", v->size, u->size);
        return NULL;
    }

    Vector* r = create_vector(v->size);
    for (int i = 0; i < v->size; ++i) {
        r->elements[i] = v->elements[i] / u->elements[i];
    }

    return r;
}

Matrix* matrix_add_vector(const Matrix* M, const Vector* v) {
    if (M->cols != v->size) {
        fprintf(stderr, "Invalid size. (%d, %d) and %d\n", M->rows, M->cols, v->size);
        return NULL;
    }

    Matrix* N = create_matrix(M->rows, M->cols);
    for (int i = 0; i < M->cols; ++i) {
        for (int j = 0; j < M->rows; ++j) {
            N->elements[j][i] = M->elements[j][i] + v->elements[i];
        }
    }

    return N;
}

Matrix* matrix_add_matrix(const Matrix* M, const Matrix* N) {
    if (M->rows != N->rows || M->cols != N->cols) {
        fprintf(stderr, "Invalid size. (%d, %d) and (%d, %d)\n", M->rows, M->cols, N->rows, N->cols);
        return NULL;
    }

    Matrix* R = create_matrix(M->rows, M->cols);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            R->elements[i][j] = M->elements[i][j] + N->elements[i][j];
        }
    }

    return R;
}

Matrix* matrix_sub_vector(const Matrix* M, const Vector* v) {
    if (M->cols != v->size) {
        fprintf(stderr, "Invalid size. (%d, %d) and %d\n", M->rows, M->cols, v->size);
        return NULL;
    }

    Matrix* N = create_matrix(M->rows, M->cols);
    for (int i = 0; i < M->cols; ++i) {
        for (int j = 0; j < M->rows; ++j) {
            N->elements[j][i] = M->elements[j][i] - v->elements[i];
        }
    }

    return N;
}

Matrix* matrix_div_vector(const Matrix* M, const Vector* v) {
    if (M->cols != v->size) {
        fprintf(stderr, "Invalid size. (%d, %d) and %d\n", M->rows, M->cols, v->size);
        return NULL;
    }

    Matrix* N = create_matrix(M->rows, M->cols);
    for (int i = 0; i < M->cols; ++i) {
        for (int j = 0; j < M->rows; ++j) {
            N->elements[j][i] = M->elements[j][i] / v->elements[i];
        }
    }

    return N;
}

Vector* vector_add_scalar(const Vector* V, double val) {
    Vector* r = create_vector(V->size);
    for (int i = 0; i < V->size; ++i) {
        r->elements[i] = V->elements[i] + val;
    }

    return r;
}

Matrix* pow_matrix(Matrix* M, double k) {
    Matrix* N = create_matrix(M->rows, M->cols);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            N->elements[i][j] = pow(M->elements[i][j], k);
        }
    }

    return N;
}

Vector* sqrt_vector(const Vector* V) {
    Vector* r = create_vector(V->size);
    for (int i = 0; i < V->size; ++i) {
        r->elements[i] = sqrt(V->elements[i]);
    }
    return r;
}

Matrix* im2col(const Matrix4d* M, int filter_h, int filter_w, int stride, int pad) {
    const int N = M->sizes[0];
    const int C = M->sizes[1]; 
    const int H = M->sizes[2];
    const int W = M->sizes[3];

    const int out_h = (H + 2 * pad - filter_h) / stride + 1; 
    const int out_w = (W + 2 * pad - filter_w) / stride + 1; 

    Matrix4d* A = matrix_4d_pad(M, pad);  

    Matrix* R = create_matrix(N * out_h * out_w, filter_h * filter_w * C);

    int rpos = 0, cpos = 0; 
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < out_h * stride; k += stride) {
            for (int l = 0; l < out_w * stride; l += stride) {
                for (int j = 0; j < C; ++j) {
                    for (int m = k; m < k + filter_w; ++m) {
                        for (int n = l; n < l + filter_h; ++n) {
                            R->elements[rpos][cpos] = A->elements[i][j][m][n]; 
                            ++cpos;
                            if (cpos == R->cols) {
                                cpos = 0;
                                ++rpos;
                            }
                        }
                    }
                }
            }
        }
    }

    free_matrix_4d(A);

    return R;
}

Matrix4d* col2im(const Matrix* M, int* sizes, int filter_h, int filter_w, int stride, int pad) {
    const int N = sizes[0];
    const int C = sizes[1]; 
    const int H = sizes[2];
    const int W = sizes[3];

    const int out_h = (H + 2 * pad - filter_h) / stride + 1; 
    const int out_w = (W + 2 * pad - filter_w) / stride + 1; 
    Matrix4d* B = create_matrix_4d(N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1);

    int n = 0, c = 0, h = 0, w = 0;
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; j += (filter_h * filter_w)) {
            double* buf = (double*)malloc(sizeof(double) * (filter_h * filter_w));
            for (int k = j; k < j + (filter_h * filter_w); ++k) {
                buf[k-j] = M->elements[i][k];
            }

            int idx = 0;
            for (int k = h; k < h + filter_h; ++k) {
                for (int l = w; l < w + filter_w; ++l) {
                    B->elements[n][c][k][l] += buf[idx++];
                }
            }

            ++c;
            if (c == C) {
                c = 0;
                w += stride;
            }

            if (w >= (out_w * stride)) {
                w = 0;
                h += stride;
            }

            if (h >= (out_h * stride)) {
                h = 0;
                ++n;
            }

            free(buf);
        }
    }
    
    Matrix4d* R = create_matrix_4d(N, C, H, W);
    for (int i = 0; i < B->sizes[0]; ++i) {
        for (int j = 0; j < B->sizes[1]; ++j) {
            for (int k = 0; k < B->sizes[2]; ++k) {
                for (int l = 0; l < B->sizes[3]; ++l) {
                    if ((k < pad) || ((R->sizes[2] + pad) <= k) || (l < pad) || ((R->sizes[3] + pad) <= l)) {
                        continue;
                    } 

                    R->elements[i][j][k - pad][l - pad] = B->elements[i][j][k][l];
                }
            }
        }
    }

    free_matrix_4d(B);
    
    return R;
}

Matrix4d* matrix_4d_pad(const Matrix4d* M, int pad) {
    Matrix4d* R = create_matrix_4d(M->sizes[0], M->sizes[1], M->sizes[2] + 2 * pad, M->sizes[3] + 2 * pad);

    for (int i = 0; i < R->sizes[0]; ++i) {
        for (int j = 0; j < R->sizes[1]; ++j) {
            for (int k = 0; k < R->sizes[2]; ++k) {
                for (int l = 0; l < R->sizes[3]; ++l) {
                    if ((k < pad) || ((M->sizes[2] + pad) <= k) || (l < pad) || ((M->sizes[3] + pad) <= l)) {
                        R->elements[i][j][k][l] = 0;  
                    } else {
                        R->elements[i][j][k][l] = M->elements[i][j][k - pad][l - pad];
                    }
                }
            }
        }
    }

    return R;
}

//
// create batch
//

Matrix* create_image_batch(double** images, const int* batch_index, int size) {
    Matrix* M = create_matrix(size, NUM_OF_PIXELS);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < NUM_OF_PIXELS; ++j) {
            M->elements[i][j] = images[batch_index[i]][j];
        }
    }

    return M;
}

Matrix4d* create_image_batch_4d(double**** images, const int* batch_index, int size) {
    Matrix4d* M = create_matrix_4d(size, 1, NUM_OF_ROWS, NUM_OF_COLS);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < NUM_OF_ROWS; ++j) {
            for (int k = 0; k < NUM_OF_COLS; ++k) {
                M->elements[i][0][j][k] = images[batch_index[i]][0][j][k];
            }
        }
    }

    return M;
}

Vector* create_label_batch(uint8_t* labels, const int* batch_index, int size) {
    Vector* v = create_vector(size);
    for (int i = 0; i < size; ++i) {
        v->elements[i] = labels[batch_index[i]];
    }

    return v;
}

//
// debug
//

void print_vector(const Vector* v) {
    printf("size=%d, [", v->size);
    for (int i = 0; i < v->size; ++i) {
        printf("%lf ", v->elements[i]);
    }
    printf("]\n");
}

void print_matrix(const Matrix* M) {
    printf("rows=%d,cols=%d,[", M->rows, M->cols);
    for (int i = 0; i < M->rows; ++i) {
        printf("[");
        for (int j = 0; j < M->cols; ++j) {
            printf("%lf ", M->elements[i][j]);
        }
        printf("]\n");
    }
    printf("]\n");
}

void print_matrix_4d(const Matrix4d* M) {
    printf("sizes = (%d,%d,%d,%d)\n", M->sizes[0], M->sizes[1], M->sizes[2], M->sizes[3]);
    for (int i = 0; i < M->sizes[0]; ++i) {
        printf("[\n");
        for (int j = 0; j < M->sizes[1]; ++j) {
            printf("    [\n");
            for (int k = 0; k < M->sizes[2]; ++k) {
                printf("        [");
                for (int l = 0; l < M->sizes[3]; ++l) {
                    printf("%lf ", M->elements[i][j][k][l]);
                }
                printf("]\n");
            }
            printf("    ]\n");
        }
        printf("]\n");
    }
    printf("]\n");
}

