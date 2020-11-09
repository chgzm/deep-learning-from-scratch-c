#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <util.h>
#include <mnist.h>
#include <matrix.h>
#include <function.h>
#include <multi_layer_net.h>
#include <optimizer.h>

#define ITERS_NUM  2000
#define BATCH_SIZE 128

static void SGD_process(MultiLayerNet* net, double** train_images, uint8_t* train_labels, double** test_images, uint8_t* test_labels) { 
    const double lr = 0.01;

    FILE* fp = fopen("mnist_SGD.txt", "w");
    if (!fp) {
        fprintf(stderr, "Failed to open file.\n");
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

static void Momentum_process(MultiLayerNet* net, double** train_images, uint8_t* train_labels, double** test_images, uint8_t* test_labels) { 
    const double lr = 0.01;
    const double momentum = 0.9;

    FILE* fp = fopen("mnist_Momentum.txt", "w");
    if (!fp) {
        fprintf(stderr, "Failed to open file.\n");
        return;
    }

    Matrix** m = malloc(sizeof(Matrix*) * net->hidden_layer_num + 1);
    Vector** v = malloc(sizeof(Vector*) * net->hidden_layer_num + 1);

    m[0] = create_matrix(net->input_size, net->hidden_size);
    v[0] = create_vector(net->hidden_size);
 
    for (int i = 1; i < net->hidden_layer_num + 1; ++i) {
        m[i] = create_matrix(net->hidden_size, net->hidden_size);
        v[i] = create_vector(net->hidden_size);
    }
 
    for (int i = 0; i < ITERS_NUM; ++i) {
        int* batch_index = choice(NUM_OF_TRAIN_IMAGES, BATCH_SIZE);
        Matrix* x_batch  = create_image_batch(train_images, batch_index, BATCH_SIZE);
        Vector* t_batch  = create_label_batch(train_labels, batch_index, BATCH_SIZE);

        multi_layer_net_gradient(net, x_batch, t_batch);

        fprintf(fp, "%lf\n", multi_layer_net_loss(net, x_batch, t_batch));

        for (int j = 0; j < net->hidden_layer_num + 1; ++j) {
            Momentum_update_vector(net->b[j], net->A[j]->db, lr, momentum, v[j]);
            Momentum_update_matrix(net->W[j], net->A[j]->dW, lr, momentum, m[j]);
        }

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

static void AdaGrad_process(MultiLayerNet* net, double** train_images, uint8_t* train_labels, double** test_images, uint8_t* test_labels) {
    const double lr = 0.01;

    FILE* fp = fopen("mnist_AdaGrad.txt", "w");
    if (!fp) {
        fprintf(stderr, "Failed to open file.\n");
        return;
    }

    Matrix** m = malloc(sizeof(Matrix*) * net->hidden_layer_num + 1);
    Vector** v = malloc(sizeof(Vector*) * net->hidden_layer_num + 1);

    m[0] = create_matrix(net->input_size, net->hidden_size);
    v[0] = create_vector(net->hidden_size);
 
    for (int i = 1; i < net->hidden_layer_num + 1; ++i) {
        m[i] = create_matrix(net->hidden_size, net->hidden_size);
        v[i] = create_vector(net->hidden_size);
    }

    for (int i = 0; i < ITERS_NUM; ++i) {
        int* batch_index = choice(NUM_OF_TRAIN_IMAGES, BATCH_SIZE);
        Matrix* x_batch  = create_image_batch(train_images, batch_index, BATCH_SIZE);
        Vector* t_batch  = create_label_batch(train_labels, batch_index, BATCH_SIZE);

        multi_layer_net_gradient(net, x_batch, t_batch);

        fprintf(fp, "%lf\n", multi_layer_net_loss(net, x_batch, t_batch));

        for (int j = 0; j < net->hidden_layer_num + 1; ++j) {
            AdaGrad_update_matrix(net->W[j], net->A[j]->dW, lr, m[j]);
            AdaGrad_update_vector(net->b[j], net->A[j]->db, lr, v[j]);
        }

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

static void Adam_process(MultiLayerNet* net, double** train_images, uint8_t* train_labels, double** test_images, uint8_t* test_labels) {
    const double lr = 0.001;
    const double beta1 = 0.9;
    const double beta2 = 0.999;

    FILE* fp = fopen("mnist_Adam.txt", "w");
    if (!fp) {
        fprintf(stderr, "Failed to open file.\n");
        return;
    }

    Matrix** m = malloc(sizeof(Matrix*) * net->hidden_layer_num + 1);
    Matrix** n = malloc(sizeof(Matrix*) * net->hidden_layer_num + 1);
    Vector** v = malloc(sizeof(Vector*) * net->hidden_layer_num + 1);
    Vector** u = malloc(sizeof(Vector*) * net->hidden_layer_num + 1);

    m[0] = create_matrix(net->input_size, net->hidden_size);
    n[0] = create_matrix(net->input_size, net->hidden_size);
    v[0] = create_vector(net->hidden_size);
    u[0] = create_vector(net->hidden_size);

    for (int i = 1; i < net->hidden_layer_num + 1; ++i) {
        m[i] = create_matrix(net->hidden_size, net->hidden_size);
        n[i] = create_matrix(net->hidden_size, net->hidden_size);
        v[i] = create_vector(net->hidden_size);
        u[i] = create_vector(net->hidden_size);
    }

    for (int i = 0; i < ITERS_NUM; ++i) {
        int* batch_index = choice(NUM_OF_TRAIN_IMAGES, BATCH_SIZE);
        Matrix* x_batch  = create_image_batch(train_images, batch_index, BATCH_SIZE);
        Vector* t_batch  = create_label_batch(train_labels, batch_index, BATCH_SIZE);

        multi_layer_net_gradient(net, x_batch, t_batch);
        fprintf(fp, "%lf\n", multi_layer_net_loss(net, x_batch, t_batch));

        for (int j = 0; j < net->hidden_layer_num + 1; ++j) {
            Adam_update_matrix(net->W[j], net->A[j]->dW, lr, beta1, beta2, m[j], n[j], i);
            Adam_update_vector(net->b[j], net->A[j]->db, lr, beta1, beta2, v[j], u[j], i);
        }

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

static void process(int optimizer, double** train_images, uint8_t* train_labels, double** test_images, uint8_t* test_labels) {
    MultiLayerNet* net = create_multi_layer_net(784, 4, 100, 10, BATCH_SIZE, He, 0, 0);
    switch (optimizer) {
    case SGD: { 
        SGD_process(net, train_images, train_labels, test_images, test_labels);
        break;
    }
    case Momentum: {
        Momentum_process(net, train_images, train_labels, test_images, test_labels);
        break;
    }
    case AdaGrad: {
        AdaGrad_process(net, train_images, train_labels, test_images, test_labels);
        break;
    }
    case Adam: {
        Adam_process(net, train_images, train_labels, test_images, test_labels);
        break;
    }
    default: {
        break;
    }
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
    process(SGD,      train_images, train_labels, test_images, test_labels);
    process(Momentum, train_images, train_labels, test_images, test_labels);
    process(AdaGrad,  train_images, train_labels, test_images, test_labels);
    process(Adam,     train_images, train_labels, test_images, test_labels);

    return 0;   
}
