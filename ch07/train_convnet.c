#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <util.h>
#include <debug.h>
#include <mnist.h>
#include <matrix.h>
#include <optimizer.h>
#include <simple_convnet.h>
#include <trainer.h>

static const int TRAIN_SIZE = 5000;
static const int TEST_SIZE = 1000;
static const int EPOCHS = 20;
static const int MINI_BATCH_SIZE = 100;
static const double LEARNING_RATE = 0.001;

int main() {
    double**** train_images = load_mnist_images_4d("./../dataset/train-images-idx3-ubyte");
    if (train_images == NULL) {
        fprintf(stderr, "failed to load train images.\n");
        return -1;
    }

    uint8_t* train_labels = load_mnist_labels("./../dataset/train-labels-idx1-ubyte");
    if (train_labels == NULL) {
        fprintf(stderr, "failed to load train labels.\n");
        return -1;
    }

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

    SimpleConvNet* net = create_simple_convnet(1, 28, 28, 30, 5, 0, 1, 100, 10, 0.01);
    SimpleConvNetTrainer* trainer = create_simple_convnet_trainer(
        net,
        train_images,
        train_labels,
        test_images,
        test_labels,
        EPOCHS,
        MINI_BATCH_SIZE,
        Adam,
        TRAIN_SIZE,
        TEST_SIZE,
        LEARNING_RATE,
        true
    );

    simple_convnet_trainer_train(trainer);

    free_simple_convnet_trainer(trainer);
    free(train_images);
    free(train_labels);
    free(test_images);
    free(test_labels);

    return 0;   
}
