#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <util.h>
#include <debug.h>
#include <mnist.h>
#include <matrix.h>
#include <function.h>
#include <multi_layer_net.h>
#include <multi_layer_net_extend.h>
#include <optimizer.h>

#define ITERS_NUM  2000
#define TRAIN_SIZE 300
#define BATCH_SIZE 100
#define MAX_EPOCHS 201

// const double weight_decay_lambda = 0;
const double weight_decay_lambda = 0.1;

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

    srand(time(NULL));

    FILE* fp = fopen("overfit_weight_decay.txt", "w");
    if (!fp) {
        fprintf(stderr, "Failed to open file.\n");
        return -1;
    }

    MultiLayerNet* net = create_multi_layer_net(784, 6, 100, 10, BATCH_SIZE, He, 0, weight_decay_lambda);
    const double lr = 0.01;

    int epoch_cnt = 0;
    for (int i = 0; i < 1000000000; ++i) {
        int* batch_index = choice(TRAIN_SIZE, BATCH_SIZE);

        Matrix* x_batch  = create_image_batch(train_images, batch_index, BATCH_SIZE);
        Vector* t_batch  = create_label_batch(train_labels, batch_index, BATCH_SIZE);

        multi_layer_net_gradient(net, x_batch, t_batch);

        for (int j = 0; j < net->hidden_layer_num + 1; ++j) {
            SGD_update_vector(net->b[j], net->A[j]->db, lr);
            Matrix* tmp = _scalar_matrix(net->W[j], net->weight_decay_lambda);
            Matrix* _dW = matrix_add_matrix(net->A[j]->dW, tmp);
            SGD_update_matrix(net->W[j], _dW, lr);

            free_matrix(tmp);
            free_matrix(_dW);
        }

        if (i % 3 == 0) {
            const double train_acc = multi_layer_net_accuracy(net, train_images, train_labels, TRAIN_SIZE);
            const double test_acc  = multi_layer_net_accuracy(net, test_images, test_labels, NUM_OF_TEST_IMAGES);
            printf("epoch:%d train acc, test acc | %lf, %lf\n", epoch_cnt, train_acc, test_acc);
            fprintf(fp, "%d %lf %lf\n", epoch_cnt, train_acc, test_acc);
            ++epoch_cnt;
        }

        free(batch_index);
        free_matrix(x_batch);
        free_vector(t_batch); 

        if (epoch_cnt >= MAX_EPOCHS) {
            break;
        }   
    }

    fclose(fp);

    return 0;   
}
