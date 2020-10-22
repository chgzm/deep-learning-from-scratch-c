#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <util.h>
#include <mnist.h>
#include <matrix.h>
#include <function.h>
#include "multi_layer_net.h"
#include "optimizer.h"

#define ITERS_NUM  10000
#define BATCH_SIZE 100

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
    MultiLayerNet* net = create_multi_layer_net(784, 1, 100, 10, BATCH_SIZE);
    for (int i = 0; i < ITERS_NUM; ++i) {
        int* batch_index = choice(NUM_OF_TRAIN_IMAGES, BATCH_SIZE);
        Matrix* x_batch  = create_image_batch(train_images, batch_index, BATCH_SIZE);
        Vector* t_batch  = create_label_batch(train_labels, batch_index, BATCH_SIZE);

        gradient(net, x_batch, t_batch);

        if (i % iter_per_epoch == 0) {
            const double train_acc = accuracy_multi_layer_net(net, train_images, train_labels, NUM_OF_TRAIN_IMAGES);
            const double test_acc  = accuracy_multi_layer_net(net, test_images,  test_labels,  NUM_OF_TEST_IMAGES);
            printf("train acc, test acc | %lf, %lf\n", train_acc, test_acc);
        }

        free(batch_index);
        free_matrix(x_batch);
        free_vector(t_batch); 
    }   

    return 0;   
}
