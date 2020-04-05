#include <stdio.h>
#include <stdlib.h>
#include <util.h>
#include <mnist.h>
#include <matrix.h>
#include <function.h>

int main() {
    double** images = load_mnist_images("./../dataset/t10k-images-idx3-ubyte");
    uint8_t* labels = load_mnist_labels("./../dataset/t10k-labels-idx1-ubyte");

    Matrix* W1 = create_matrix(784, 50);
    if (init_matrix_from_file(W1, "./../dataset/W1.csv") != 0) {
        fprintf(stderr, "Failed to load W1.csv\n");
        return -1;
    }

    Matrix* W2 = create_matrix(50, 100);
    if (init_matrix_from_file(W2, "./../dataset/W2.csv") != 0) {
        fprintf(stderr, "Failed to load W2.csv\n");
        return -1;
    }

    Matrix* W3 = create_matrix(100, 10);
    if (init_matrix_from_file(W3, "./../dataset/W3.csv") != 0) {
        fprintf(stderr, "Failed to load W3.csv\n");
        return -1;
    }

    Vector* b1 = create_vector(50);
    if (init_vector_from_file(b1, "./../dataset/b1.csv") != 0) {
        fprintf(stderr, "Failed to load b1.csv\n");
        return -1;
    }

    Vector* b2 = create_vector(100);
    if (init_vector_from_file(b2, "./../dataset/b2.csv") != 0) {
        fprintf(stderr, "Failed to load b2.csv\n");
        return -1;
    }

    Vector* b3 = create_vector(10);
    if (init_vector_from_file(b3, "./../dataset/b3.csv") != 0) {
        fprintf(stderr, "Failed to load b3.csv\n");
        return -1;
    }

    int accuracy_cnt = 0;
    for (int i = 0; i < NUM_OF_TEST_IMAGES; ++i) {
        Vector* x = create_vector(NUM_OF_PIXELS);
        init_vector_from_array(x, images[i]);

        Vector* t1 = dot_vector_matrix(x, W1);
        Vector* a1 = add_vector(t1, b1);
        Vector* z1 = vector_sigmoid(a1);
        Vector* t2 = dot_vector_matrix(z1, W2);
        Vector* a2 = add_vector(t2, b2);
        Vector* z2 = vector_sigmoid(a2);
        Vector* t3 = dot_vector_matrix(z2, W3);
        Vector* a3 = add_vector(t3, b3);
        Vector* y  = vector_softmax(a3);

        if (argmax(y) == labels[i]) {
            ++accuracy_cnt;
        }

        free_vector(x);
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

    return 0;
}
