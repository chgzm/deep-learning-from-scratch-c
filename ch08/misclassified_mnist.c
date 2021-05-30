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
static const int MAX_VIEW = 20;

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

static int plot_missclassified(int* miss_index, double**** test_images) {
    FILE* gp = popen("gnuplot -persist", "w");
    if (gp == NULL) {
        fprintf(stderr, "Failed to open gnuplot pipe.\n");
        return -1;
    }

    const int cols = 5;
    const int rows = MAX_VIEW / cols;

    fprintf(gp, "set term qt size 1600, 1200\n");
    fprintf(gp, "set palette defined (0 'black', 1 'white')\n");
    fprintf(gp, "unset colorbox\n");
    fprintf(gp, "unset key\n");
    fprintf(gp, "unset tics\n");
    fprintf(gp, "set palette grey\n");
    fprintf(gp, "set multiplot layout %d, %d\n", rows, cols);
    fprintf(gp, "set yrange [] reverse\n");
    for (int i = 0; i < MAX_VIEW; ++i) {
        fprintf(gp, "set size %.2lf, %.2lf\n", 1.0 / rows, 1.0 / cols);
        fprintf(gp, "plot '-' matrix with image\n");
        double** img = test_images[miss_index[i]][0];
        for (int j = 0; j < NUM_OF_ROWS; ++j) {
            for (int k = 0; k < NUM_OF_ROWS; ++k) {
                fprintf(gp, "%lf ", img[j][k]);
            }
            fprintf(gp, "\n");
        }
        fprintf(gp, "e\n");
        fprintf(gp, "e\n");
        fflush(gp);
    }

    fflush(gp);
    pclose(gp);

    return 0;
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

    int miss_cnt = 0;
    int miss_index[MAX_VIEW];
    uint8_t miss_inference[MAX_VIEW];
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
            } else if (miss_cnt < MAX_VIEW) {
                miss_index[miss_cnt] = i * BATCH_SIZE + j;  
                miss_inference[miss_cnt] = max_index;
                ++miss_cnt;
            } 
        }

        free_matrix_4d(x_batch);
        free_vector(t_batch);
        free_matrix(Y);
    }

    printf("test accuracy:%lf\n", (double)acc / NUM_OF_TEST_IMAGES);

    printf("======= misclassified result =======\n");
    printf("{view index: (label, inference), ...}\n");
    for (int i = 0; i < miss_cnt; ++i) {
        if (i == 0) {
            printf("{");
        } else {
            printf(", ");
        }

        printf("%d: (%d, %d)", i + 1, test_labels[miss_index[i]], miss_inference[i]);

        if (i == (miss_cnt - 1)) {
            printf("}\n");
        }
    }

    plot_missclassified(miss_index, test_images);

    return 0;
}
