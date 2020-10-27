#include "matrix.h" 
#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

//
// factory
//
Vector* create_vector(int size) {
    Vector* v = calloc(1, sizeof(Vector));
    v->size = size;
    v->elements = calloc(size, sizeof(double));
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

void scalar_matrix(Matrix* M, double k) {
    for (int i = 0; i < M->rows; ++i) {
       for (int j = 0; j < M->cols; ++j) {
            M->elements[i][j] *= k;
       }
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
    printf("[");
    for (int i = 0; i < v->size; ++i) {
        printf("%lf ", v->elements[i]);
    }
    printf("]\n");
}

void print_matrix(const Matrix* M) {
    printf("[");
    for (int i = 0; i < M->rows; ++i) {
        printf("[");
        for (int j = 0; j < M->cols; ++j) {
            printf("%lf ", M->elements[i][j]);
        }
        printf("]\n");
    }
    printf("]\n");
}

