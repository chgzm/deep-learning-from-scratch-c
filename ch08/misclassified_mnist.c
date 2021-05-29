#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <util.h>
#include <debug.h>
#include <mnist.h>
#include <matrix.h>

#include "deep_convnet.h"

static const int BATCH_SIZE = 100;

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

int main() {
    double**** test_images = load_mnist_images_4d("./../dataset/t10k-images-idx3-ubyte");
    if (test_images == NULL) {
        fprintf(stderr, "failed to load test images.\n");
        return -1;
    }

    uint8_t* test_labels = load_mnist_labels("./../dataset/t10k-labels-idx1-ubyte");
    if (test_labels == NULL) {
        fprintf(stderr, "failed to load test labels.\n");
        return -1;
    }

    srand(time(NULL));

    int input_dim[3] = {1, 28, 28};
    ConvParam conv_param[6] = {
        {16, 3, 1, 1},
        {16, 3, 1, 1},
        {32, 3, 1, 1},
        {32, 3, 2, 1},
        {64, 3, 1, 1},
        {64, 3, 1, 1},
    }; 
    DeepConvNet* net = create_deep_convnet(input_dim, conv_param, 50, 10); 
    deep_convnet_load_params(net);

    int pos = 0;
    int miss_index[20] = {-1};
    int acc = 0;
    printf("calculating test accuracy ... \n");
    for (int i = 0; i < (NUM_OF_TEST_IMAGES / BATCH_SIZE); ++i) {
        int batch_index[BATCH_SIZE];
        for (int j = 0; j < BATCH_SIZE; ++j) {
            batch_index[j] = i * BATCH_SIZE + j; 
        }

        Matrix4d* x_batch = create_image_batch_4d(test_images, batch_index, BATCH_SIZE);
        Vector* t_batch = create_label_batch(test_labels, batch_index, BATCH_SIZE);

        Matrix* Y = deep_convnet_predict(net, x_batch, false);
        for (int j = 0; j < Y->rows; ++j) {
            const int max_index = _argmax(Y->elements[j], Y->cols);
            if (max_index == test_labels[i * BATCH_SIZE + j]) {
                ++acc;
            } else if (pos < 20) {
                miss_index[pos++] = i * BATCH_SIZE + j;  
            }
        }

        free_matrix_4d(x_batch);
        free_vector(t_batch);
        free_matrix(Y);
    }

    printf("test accuracy:%lf\n", (double)acc / NUM_OF_TEST_IMAGES);

    for (int i = 0; i < 20; ++i) {
        printf("miss_index[%d] = %d\n", i, miss_index[i]);
    }

    return 0;
}
