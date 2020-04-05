#include "twolayernet.h"
#include <function.h>
#include <mnist.h>
#include <stdlib.h>
#include <math.h>

TwoLayerNet* create_two_layer_net(int input_size, int hidden_size, int output_size) {
    TwoLayerNet* net = malloc(sizeof(TwoLayerNet));

    net->W1 = create_matrix(input_size, hidden_size);
    net->b1 = create_vector(hidden_size);
    net->W2 = create_matrix(hidden_size, output_size);
    net->b2 = create_vector(output_size);

    init_matrix_random(net->W1);
    init_matrix_random(net->W2);

    return net;
}

static Matrix* predict(const TwoLayerNet* net, const Matrix* X) {
    Matrix* A1 = dot_matrix(X, net->W1);  
    Matrix* Z1 = create_matrix(A1->rows, A1->cols);
    for (int i = 0; i < A1->rows; ++i) {
        for (int j = 0; j < A1->cols; ++j) {
            A1->elements[i][j] += net->b1->elements[j];
            Z1->elements[i][j] = sigmoid(A1->elements[i][j]);
        }
    }

    Matrix* A2 = dot_matrix(Z1, net->W2);
    for (int i = 0; i < A2->rows; ++i) {
        for (int j = 0; j < A2->cols; ++j) {
            A2->elements[i][j] += net->b2->elements[j];
        }
    }

    Matrix* Y = matrix_softmax(A2);

    free_matrix(A1);
    free_matrix(A2);
    free_matrix(Z1);
    return Y;
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

static double loss(TwoLayerNet* net, const Matrix* X, const Vector* t) {
    Matrix* Y = predict(net, X);
    const double e = cross_entropy_error(Y, t);

    free_matrix(Y);
    return e;
}

static Matrix* numerical_gradient_matrix(TwoLayerNet* net, Matrix* M, const Matrix* X, const Vector* t) {
    static const double h = 1e-4;

    Matrix* dM = create_matrix(M->rows, M->cols);
    for (int i = 0; i < M->rows; ++i) {
        for (int j = 0; j < M->cols; ++j) {
            const double tmp = M->elements[i][j];
            M->elements[i][j] = tmp + h;
            const double f1 = loss(net, X, t);

            M->elements[i][j] = tmp - 2*h;
            const double f2 = loss(net, X, t);

            dM->elements[i][j] = (f1 - f2) / (2*h);
            M->elements[i][j] = tmp;
        }
    }

    return dM;
}

static Vector* numerical_gradient_vector(TwoLayerNet* net, Vector* v, const Matrix* X, const Vector* t) {
    static const double h = 1e-4;

    Vector* dv = create_vector(v->size);
    for (int i = 0; i < v->size; ++i) {
        const double tmp = v->elements[i];
        v->elements[i] = tmp + h;
        const double f1 = loss(net, X, t);

        v->elements[i] = tmp - 2*h;
        const double f2 = loss(net, X, t);

        dv->elements[i] = (f1 - f2) / (2*h);
        v->elements[i] = tmp;
    }

    return dv;
}

static void update_matrix(Matrix* A, const Matrix* dA) {
    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            A->elements[i][j] -= (dA->elements[i][j] * LEARNING_RATE);
        }
    }
}

static void update_vector(Vector* v, const Vector* dv) {
    for (int i = 0; i < v->size; ++i) {
        v->elements[i] -= (dv->elements[i] * LEARNING_RATE);
    }
}

void numerical_gradient(TwoLayerNet* net, const Matrix* X, const Vector* t) {
    Matrix* dW1 = numerical_gradient_matrix(net, net->W1, X, t);         
    Vector* db1 = numerical_gradient_vector(net, net->b1, X, t);         
    Matrix* dW2 = numerical_gradient_matrix(net, net->W2, X, t);         
    Vector* db2 = numerical_gradient_vector(net, net->b2, X, t);         

    update_matrix(net->W1, dW1);
    update_vector(net->b1, db1);
    update_matrix(net->W2, dW2);
    update_vector(net->b2, db2);

    free_matrix(dW1);
    free_vector(db1);
    free_matrix(dW2);
    free_vector(db2);
}

void gradient(TwoLayerNet* net, const Matrix* X, const Vector* t) {
    //
    // forward
    //
    Matrix* A1 = dot_matrix(X, net->W1);  
    Matrix* Z1 = create_matrix(A1->rows, A1->cols);
    for (int i = 0; i < A1->rows; ++i) {
        for (int j = 0; j < A1->cols; ++j) {
            A1->elements[i][j] += net->b1->elements[j];
            Z1->elements[i][j] = sigmoid(A1->elements[i][j]);
        }
    }

    Matrix* A2 = dot_matrix(Z1, net->W2);
    for (int i = 0; i < A2->rows; ++i) {
        for (int j = 0; j < A2->cols; ++j) {
            A2->elements[i][j] += net->b2->elements[j];
        }
    }

    Matrix* Y = matrix_softmax(A2);

    //
    // backward
    //
    Matrix* dY = create_matrix(Y->rows, Y->cols); 
    for (int i = 0; i < Y->rows; ++i) {
        for (int j = 0; j < Y->cols; ++j) {
            dY->elements[i][j] = Y->elements[i][j];
            if (j == t->elements[i]) {
                dY->elements[i][j] -= 1.0;
            }
            dY->elements[i][j] /= X->rows;
        }
    }

    // dW2
    Matrix* Z1_T = transpose(Z1);
    Matrix* dW2 = dot_matrix(Z1_T, dY);

    // db2
    Vector* db2 = create_vector(net->b2->size);
    for (int i = 0; i < dY->cols; ++i) {
        double sum = 0;
        for (int j = 0; j < dY->rows; ++j) {
            sum += dY->elements[j][i];
        }
        db2->elements[i] = sum;
    }

    Matrix* W2_T = transpose(net->W2);
    Matrix* dZ1 = dot_matrix(dY, W2_T);

    Matrix* dA1 = sigmoid_grad(A1); 
    for (int i = 0; i < dA1->rows; ++i) {
        for (int j = 0; j < dA1->cols; ++j) {
            dA1->elements[i][j] *= dZ1->elements[i][j];
        }
    }

    // dW1
    Matrix* X_T = transpose(X);
    Matrix* dW1 = dot_matrix(X_T, dA1);

    // db1
    Vector* db1 = create_vector(net->b1->size);
    for (int i = 0; i < dA1->cols; ++i) {
        double sum = 0;
        for (int j = 0; j < dA1->rows; ++j) {
            sum += dA1->elements[j][i];
        }
        db1->elements[i] = sum;
    }
   
    update_matrix(net->W1, dW1);
    update_vector(net->b1, db1);
    update_matrix(net->W2, dW2);
    update_vector(net->b2, db2);

    free_matrix(A1);
    free_matrix(Z1);
    free_matrix(A2);
    free_matrix(Y);
    free_matrix(dY);
    free_matrix(Z1_T);
    free_matrix(dW2);
    free_vector(db2);
    free_matrix(W2_T);
    free_matrix(dZ1);
    free_matrix(dA1);
    free_matrix(X_T);
    free_matrix(dW1);
    free_vector(db1);
}

static int _argmax(const double* v, int size) {
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

double accuracy_two_layer_net(const TwoLayerNet* net, double** images, uint8_t* labels, int size) {
    Matrix* X = create_matrix(size, NUM_OF_PIXELS);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < NUM_OF_PIXELS; ++j) {
           X->elements[i][j] = images[i][j];
        } 
    }

    Matrix* Y = predict(net, X);
    int cnt = 0;
    for (int i = 0; i < Y->rows; ++i) {
        const int max_index = _argmax(Y->elements[i], Y->cols);
        if (max_index == labels[i]) {
            ++cnt;
        }
    }

    free_matrix(X);
    free_matrix(Y);
    return (double)cnt / size;
}
