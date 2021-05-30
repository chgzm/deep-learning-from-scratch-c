#include <stdio.h>
#include <stdlib.h>

#include <util.h>
#include <mnist.h>
#include <matrix.h>
#include <function.h>

int main() {
    double** images = load_mnist_images("./../dataset/t10k-images-idx3-ubyte");
    const uint8_t* labels = load_mnist_labels("./../dataset/t10k-labels-idx1-ubyte");

    Matrix* W1 = create_matrix_from_file("./data/W1.csv", 784, 50);
    if (W1 == NULL) {
        fprintf(stderr, "Failed to create W1\n");
        return -1;
    }

    Matrix* W2 = create_matrix_from_file("./data/W2.csv", 50, 100);
    if (W2 == NULL) {
        fprintf(stderr, "Failed to create W2\n");
        return -1;
    }

    Matrix* W3 = create_matrix_from_file("./data/W3.csv", 100, 10);
    if (W3 == NULL) {
        fprintf(stderr, "Failed to create W3\n");
        return -1;
    }

    Vector* b1 = create_vector_from_file("./data/b1.csv", 50);
    if (b1 == NULL) {
        fprintf(stderr, "Failed to create b1\n");
        return -1;
    }

    Vector* b2 = create_vector_from_file("./data/b2.csv", 100);
    if (b2 == NULL) {
        fprintf(stderr, "Failed to create b2\n");
        return -1;
    }

    Vector* b3 = create_vector_from_file("./data/b3.csv", 10);
    if (b3 == NULL) {
        fprintf(stderr, "Failed to create b3\n");
        return -1;
    }

    int accuracy_cnt = 0;
    Vector* x = create_vector(NUM_OF_PIXELS);
    for (int i = 0; i < NUM_OF_TEST_IMAGES; ++i) {
        x->elements = images[i];

        Vector* t1 = dot_vector_matrix(x, W1);
        Vector* a1 = add_vector(t1, b1);
        Vector* z1 = vector_sigmoid(a1);
        Vector* t2 = dot_vector_matrix(z1, W2);
        Vector* a2 = add_vector(t2, b2);
        Vector* z2 = vector_sigmoid(a2);
        Vector* t3 = dot_vector_matrix(z2, W3);
        Vector* a3 = add_vector(t3, b3);
        Vector* y  = vector_softmax(a3);

        if (vector_argmax(y) == labels[i]) {
            ++accuracy_cnt;
        }

        free_vector(t1);
        free_vector(a1);
        free_vector(z1);
        free_vector(t2);
        free_vector(a2);
        free_vector(z2);
        free_vector(t3);
        free_vector(a3);
        free_vector(y);
    }

    printf("Accuracy:%lf\n", (double)(accuracy_cnt) / NUM_OF_TEST_IMAGES);

    free_vector(x);
    free_matrix(W1);
    free_matrix(W2);
    free_matrix(W3);
    free_vector(b1);
    free_vector(b2);
    free_vector(b3);

    return 0;
}
