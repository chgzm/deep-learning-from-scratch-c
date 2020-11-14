#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <util.h>
#include <debug.h>
#include <optimizer.h>
#include <mnist.h>
#include <matrix.h>
#include <trainer.h>
#include <multi_layer_net_extend.h>

#define ITERS_NUM  2000
#define TRAIN_SIZE 300
#define MINI_BATCH_SIZE 100
#define EPOCHS 301
#define LEARNING_RATE 0.01
#define DROPOUT_RATIO 0.2

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

    MultiLayerNetExtend* net = create_multi_layer_net_extend(784, 6, 100, 10, MINI_BATCH_SIZE, He, 0, true, DROPOUT_RATIO);

    TrainerExtend* trainer = create_trainer_extend(
        net, train_images, train_labels, test_images, test_labels, EPOCHS, MINI_BATCH_SIZE, SGD,
        TRAIN_SIZE, LEARNING_RATE
    );

    trainer_extend_train(trainer);

    FILE* fp = fopen("overfit_dropout.txt", "w");
    if (!fp) {
        fprintf(stderr, "failed to open file.\n");
        return -1;
    }

    for (int i = 0; i < EPOCHS; ++i) {
        fprintf(fp, "%d %lf %lf\n", i, trainer->train_acc_list[i], trainer->test_acc_list[i]);
    }

    fclose(fp);

    return 0;   
}
