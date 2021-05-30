#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <util.h>
#include <mnist.h>
#include <matrix.h>
#include <function.h>
#include <multi_layer_net.h>
#include <optimizer.h>

static const int ITERS_NUM  = 2000;
static const int BATCH_SIZE = 128;

static void process(int weight_init, double** train_images, uint8_t* train_labels, double** test_images, uint8_t* test_labels, double weight) {
    MultiLayerNet* net = create_multi_layer_net(784, 4, 100, 10, BATCH_SIZE, weight_init, weight, 0);
    const double lr = 0.01;

    char* filename = NULL;
    switch (weight_init) {
    case STD:    { filename = strdup("weight_init_compare_STD.txt");    break; }
    case Xavier: { filename = strdup("weight_init_compare_Xavier.txt"); break; }
    case He:     { filename = strdup("weight_init_compare_He.txt");     break; }
    default:     { break; }
    }

    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open \"%s\".\n", filename);
        return;
    }

    for (int i = 0; i < ITERS_NUM; ++i) {
        int* batch_index = choice(NUM_OF_TRAIN_IMAGES, BATCH_SIZE);
        Matrix* x_batch  = create_image_batch(train_images, batch_index, BATCH_SIZE);
        Vector* t_batch  = create_label_batch(train_labels, batch_index, BATCH_SIZE);

        multi_layer_net_gradient(net, x_batch, t_batch);

        for (int j = 0; j < net->hidden_layer_num + 1; ++j) {
            SGD_update_vector(net->b[j], net->A[j]->db, lr);
            SGD_update_matrix(net->W[j], net->A[j]->dW, lr);
        }

        fprintf(fp, "%lf\n", multi_layer_net_loss(net, x_batch, t_batch));

        if (i % 100 == 0) {
            const double train_acc = multi_layer_net_accuracy(net, train_images, train_labels, NUM_OF_TRAIN_IMAGES);
            const double test_acc  = multi_layer_net_accuracy(net, test_images,  test_labels,  NUM_OF_TEST_IMAGES);
            printf("train acc, test acc | %lf, %lf\n", train_acc, test_acc);
        }

        free(batch_index);
        free_matrix(x_batch);
        free_vector(t_batch); 
    }   

    fclose(fp);
}

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
    process(STD,    train_images, train_labels, test_images, test_labels, 0.01);
    process(Xavier, train_images, train_labels, test_images, test_labels, 0);
    process(He,     train_images, train_labels, test_images, test_labels, 0);

    plot_gpfile("plot_weight_init_compare.gp");

    return 0;   
}
