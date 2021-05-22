#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>

typedef struct Vector Vector;
struct Vector {
    int size;
    double* elements;
};

typedef struct Matrix Matrix;
struct Matrix {
    int rows;
    int cols;
    double** elements;
};

typedef struct Matrix4d Matrix4d;
struct Matrix4d {
    int sizes[4];
    double **** elements;
};

//
// factory
//

Vector* create_vector(int size);
Vector* create_vector_initval(int size, double init_val);
Matrix* create_matrix(int rows, int cols);
Matrix4d* create_matrix_4d(int s1, int s2, int s3, int s4);

//
// init
//

int init_vector_from_file(Vector* v, const char* file_path);
void init_vector_from_array(Vector* v, double* vals);
int init_matrix_from_file(Matrix* M, const char* file_path);
void init_matrix_random(Matrix* M);
void init_matrix_rand(Matrix* M);
void copy_matrix(Matrix* dst, const Matrix* src);
void copy_vector(Vector* dst, const Vector* src);

//
// free
//

void free_vector(Vector* v);
void free_matrix(Matrix* M);
void free_matrix_4d(Matrix4d* M);

//
// Operator
//

Vector* add_vector(const Vector* a, const Vector* b);
Vector* dot_vector_matrix(const Vector* v, const Matrix* M);
Matrix* add_matrix(const Matrix* M, const Matrix* N);
Matrix* dot_matrix(const Matrix* M, const Matrix* N);
Matrix* product_vector_matrix(const Vector* V, const Matrix* M);
Matrix* product_matrix(const Matrix* M, const Matrix* N);
Vector* product_vector(const Vector* V, const Vector* U);
Matrix* transpose(const Matrix* M);
Matrix4d* matrix_4d_transpose(const Matrix4d* M, int n1, int n2, int n3, int n4);
Matrix4d* vector_reshape_to_4d(const Vector* v, int s1, int s2, int s3, int s4);
Matrix* matrix_reshape(const Matrix* M, int rows, int cols);
Matrix* matrix_reshape_to_2d(const Matrix4d* M, int rows, int cols);
Matrix4d* matrix_reshape_to_4d(const Matrix* M, int s1, int s2, int s3, int s4);

double matrix_sum(const Matrix* M);
Vector* matrix_col_mean(const Matrix* M);
Vector* matrix_col_sum(const Matrix* M);
Vector* matrix_row_max(const Matrix* M);
Vector* vector_div_vector(const Vector* V, const Vector* U);
Matrix* matrix_add_vector(const Matrix* M, const Vector* V);
Matrix* matrix_add_matrix(const Matrix* M, const Matrix* N);
Matrix* matrix_sub_vector(const Matrix* M, const Vector* V);
Matrix* matrix_div_vector(const Matrix* M, const Vector* V);
Vector* vector_add_scalar(const Vector* M, double v);
Matrix* pow_matrix(Matrix* M, double k);
Vector* sqrt_vector(const Vector* V);
Matrix* _scalar_matrix(const Matrix* M, double k);
void scalar_matrix(Matrix* M, double k);
void scalar_matrix_4d(Matrix4d* M, double v);
void scalar_vector(Vector* V, double k);

Matrix* im2col(const Matrix4d* M, int filter_h, int filter_w, int stride, int pad);
Matrix4d* col2im(const Matrix* M, int* sizes, int filter_h, int filter_w, int stride, int pad);
Matrix4d* matrix_4d_pad(const Matrix4d* M, int pad);

//
// create batch
//

Matrix* create_image_batch(double** images, const int* batch_index, int size);
Matrix4d* create_image_batch_4d(double** images, const int* batch_index, int size);
Vector* create_label_batch(uint8_t* labels, const int* batch_index, int size);

//
// debug
//

void print_vector(const Vector* v);
void print_matrix(const Matrix* M);
void print_matrix_4d(const Matrix4d* M);

#endif
