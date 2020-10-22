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

//
// factory
//

Vector* create_vector(int size);
Matrix* create_matrix(int rows, int cols);

//
// init
//

int init_vector_from_file(Vector* v, const char* file_path);
void init_vector_from_array(Vector* v, double* vals);
int init_matrix_from_file(Matrix* M, const char* file_path);
void init_matrix_random(Matrix* M);
void copy_matrix(Matrix* dst, const Matrix* src);
void copy_vector(Vector* dst, const Vector* src);

//
// free
//

void free_vector(Vector* v);
void free_matrix(Matrix* M);

//
// Operator
//

Vector* add_vector(const Vector* a, const Vector* b);
Vector* dot_vector_matrix(const Vector* v, const Matrix* M);
Matrix* dot_matrix(const Matrix* M, const Matrix* N);
Matrix* transpose(const Matrix* M);
void scalar_matrix(Matrix* M, double k);

//
// create batch
//

Matrix* create_image_batch(double** images, const int* batch_index, int size);
Vector* create_label_batch(uint8_t* labels, const int* batch_index, int size);

//
// debug
//

void print_vector(const Vector* v);
void print_matrix(const Matrix* M);

#endif
