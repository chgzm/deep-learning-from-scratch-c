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

static const int TRAIN_SIZE = 1000;
static const int BATCH_SIZE = 100;
static const int MAX_EPOCHS = 20;

static void process(double** train_images, uint8_t* train_labels) {
    double* weight_scale_list = logspace(0, -4, 16);

    for (int i = 0; i < 16; ++i) {
        char file_name[32];
        snprintf(file_name, 32, "batch_norm_test_%.4lf.txt", weight_scale_list[i]);
        FILE* fp = fopen(file_name, "w");
        if (!fp) {
            fprintf(stderr, "Failed to open file=%s\n", file_name);
            return;
        }

        MultiLayerNet* net = create_multi_layer_net(784, 5, 100, 10, BATCH_SIZE, STD, weight_scale_list[i], 0);
        MultiLayerNetExtend* net_bn = create_multi_layer_net_extend(784, 5, 100, 10, BATCH_SIZE, STD, weight_scale_list[i], false, 0.0);
        const double lr = 0.01;

        int epoch_cnt = 0;
        for (int i = 0; i < 1000000000; ++i) {
            int* batch_index = choice(TRAIN_SIZE, BATCH_SIZE);

            Matrix* x_batch  = create_image_batch(train_images, batch_index, BATCH_SIZE);
            Vector* t_batch  = create_label_batch(train_labels, batch_index, BATCH_SIZE);

            multi_layer_net_gradient(net, x_batch, t_batch);
            multi_layer_net_extend_gradient(net_bn, x_batch, t_batch);

            for (int j = 0; j < net->hidden_layer_num + 1; ++j) {
                SGD_update_vector(net->b[j], net->A[j]->db, lr);
                SGD_update_matrix(net->W[j], net->A[j]->dW, lr);

                SGD_update_vector(net_bn->b[j], net_bn->A[j]->db, lr);
                SGD_update_matrix(net_bn->W[j], net_bn->A[j]->dW, lr);

                if (j != net->hidden_layer_num) {
                    SGD_update_vector(net_bn->gamma[j], net_bn->B[j]->dg, lr);
                    SGD_update_vector(net_bn->beta[j],  net_bn->B[j]->db, lr);
                }
            }

            if (i % 10 == 0) {
                const double train_acc = multi_layer_net_accuracy(net, train_images, train_labels, TRAIN_SIZE);
                const double train_acc_bn = multi_layer_net_extend_accuracy(net_bn, train_images, train_labels, TRAIN_SIZE);
                printf("epoch:%d train acc, test acc | %lf, %lf\n", epoch_cnt, train_acc, train_acc_bn);
                fprintf(fp, "%d %lf %lf\n", epoch_cnt, train_acc, train_acc_bn);
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
    }
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

    srand(time(NULL));
    process(train_images, train_labels);

    plot_gpfile("plot_batch_norm_test.gp");

    return 0;   
}
