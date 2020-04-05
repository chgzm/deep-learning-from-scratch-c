#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>

#define LABEL_MAGIC   0x00000801
#define IMAGE_MAGIC   0x00000803
#define NUM_OF_ROWS   28
#define NUM_OF_COLS   28
#define NUM_OF_PIXELS 784
#define NUM_OF_TRAIN_IMAGES 60000
#define NUM_OF_TEST_IMAGES  10000

double** load_mnist_images(const char* file_path);
uint8_t* load_mnist_labels(const char* file_path);

#endif
