#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <util.h>
#include <mnist.h>
#include <matrix.h>
#include <function.h>
#include "twolayernet.h"

static const int ITERS_NUM  = 10000;
static const int BATCH_SIZE = 100;
static const double LEARNING_RATE = 0.1;

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

static void update_weight(TwoLayerNet* net) {
    update_matrix(net->W1, net->A1->dW);
    update_vector(net->b1, net->A1->db);
    update_matrix(net->W2, net->A2->dW);
    update_vector(net->b2, net->A2->db);
}

int main() {
    double** train_images = load_mnist_images("./../dataset/train-images-idx3-ubyte");
    if (train_images == NULL) {
        fprintf(stderr, "failed to load train images.\n");
        return -1;
    }

    uint8_t* train_labels = load_mnist_labels("./../dataset/train-labels-idx1-ubyte");
    if (train_labels == NULL) {
        fprintf(stderr, "failed to load train labels.\n");
        return -1;
    }

    double** test_images = load_mnist_images("./../dataset/t10k-images-idx3-ubyte");
    if (test_images == NULL) {
        fprintf(stderr, "failed to load test images.\n");
        return -1;
    }

    uint8_t* test_labels = load_mnist_labels("./../dataset/t10k-labels-idx1-ubyte");
    if (test_labels == NULL) {
        fprintf(stderr, "failed to load test labels.\n");
        return -1;
    }

    const int iter_per_epoch = NUM_OF_TRAIN_IMAGES / BATCH_SIZE;

    srand(time(NULL));
    TwoLayerNet* net = create_two_layer_net(784, 50, 10, BATCH_SIZE);
    for (int i = 0; i < ITERS_NUM; ++i) {
        int* batch_index = choice(NUM_OF_TRAIN_IMAGES, BATCH_SIZE);
        Matrix* x_batch  = create_image_batch(train_images, batch_index, BATCH_SIZE);
        Vector* t_batch  = create_label_batch(train_labels, batch_index, BATCH_SIZE);

        two_layer_net_gradient(net, x_batch, t_batch);
        update_weight(net);

        if (i % iter_per_epoch == 0) {
            const double train_acc = two_layer_net_accuracy(net, train_images, train_labels, NUM_OF_TRAIN_IMAGES);
            const double test_acc  = two_layer_net_accuracy(net, test_images,  test_labels,  NUM_OF_TEST_IMAGES);
            printf("train acc, test acc | %lf, %lf\n", train_acc, test_acc);
        }

        free(batch_index);
        free_matrix(x_batch);
        free_vector(t_batch); 
    }   

    return 0;   
}
