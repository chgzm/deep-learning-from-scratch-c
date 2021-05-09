#include "gtest/gtest.h"

extern "C" {
#include <mnist.h>
}

TEST(load_mnist_images, success) {
    double** images = load_mnist_images("../../dataset/t10k-images-idx3-ubyte");
    EXPECT_NE(nullptr, images);
    
    free(images);
}

TEST(load_mnist_images, error) {
    double** images = load_mnist_images("foo.dat");
    EXPECT_EQ(nullptr, images);
}

TEST(load_mnist_labels, success) {
    uint8_t* labels = load_mnist_labels("../../dataset/t10k-labels-idx1-ubyte");
    EXPECT_NE(nullptr, labels);
    
    free(labels);
}

TEST(load_mnist_labels, error) {
    uint8_t* labels = load_mnist_labels("foo.dat");
    EXPECT_EQ(nullptr, labels);
}
