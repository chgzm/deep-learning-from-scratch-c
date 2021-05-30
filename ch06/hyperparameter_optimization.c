#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <util.h>
#include <debug.h>
#include <optimizer.h>
#include <mnist.h>
#include <matrix.h>
#include <trainer.h>
#include <multi_layer_net_extend.h>

static const int OPTIMIZATION_TRIAL = 100;
static const int TRAIN_SIZE = 400;
static const int VALIDATION_NUM = 100;
static const int EPOCHS = 50;
static const int MINI_BATCH_SIZE = 100;

void __train(
    double** x_train, uint8_t* t_train, double** x_val, uint8_t* t_val, double lr, double weight_decay,
    int trial, double val_result[OPTIMIZATION_TRIAL][EPOCHS], double train_result[OPTIMIZATION_TRIAL][EPOCHS]
) {
    MultiLayerNet* net = create_multi_layer_net(784, 6, 100, 10, MINI_BATCH_SIZE, He, 0, weight_decay);
    Trainer* trainer = create_trainer(net, x_train, t_train, x_val, t_val, EPOCHS, MINI_BATCH_SIZE, SGD, TRAIN_SIZE, VALIDATION_NUM, lr, false);

    trainer_train(trainer);

    for (int i = 0; i < EPOCHS; ++i) {
        val_result[trial][i] = trainer->test_acc_list[i];
        train_result[trial][i] = trainer->train_acc_list[i]; 
    }

    free_trainer(trainer);
}

typedef struct Data Data;
struct Data {
    int index;
    double val_acc;
    double lr;
    double weight_decay;
};

static int comp(const void* p, const void* q) {
    if (((Data*)p)->val_acc < ((Data*)q)->val_acc) {
        return 1;
    } else if (((Data*)p)->val_acc > ((Data*)q)->val_acc) {
        return -1;
    } else {
        return 0;
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

    double** x_train = malloc(sizeof(double*) * (TRAIN_SIZE));
    uint8_t* t_train = malloc(sizeof(uint8_t) * (TRAIN_SIZE));
    double** x_val = malloc(sizeof(double*) * VALIDATION_NUM);
    uint8_t* t_val = malloc(sizeof(uint8_t) * VALIDATION_NUM);

    for (int i = 0; i < (TRAIN_SIZE + VALIDATION_NUM); ++i) {
        if (i < (TRAIN_SIZE)) {
            x_train[i] = train_images[i];
            t_train[i] = train_labels[i];
        } else {
            x_val[i - TRAIN_SIZE] = train_images[i];
            t_val[i - TRAIN_SIZE] = train_labels[i];
        }
    }

    Data dat[OPTIMIZATION_TRIAL];
    double val_result[OPTIMIZATION_TRIAL][EPOCHS];
    double train_result[OPTIMIZATION_TRIAL][EPOCHS];

    srand(time(NULL));
    for (int i = 0; i < OPTIMIZATION_TRIAL; ++i) {
        const double weight_decay = pow(10, uniform(-8, -4));
        const double lr = pow(10, uniform(-6, -2));

        __train(x_train, t_train, x_val, t_val, lr, weight_decay, i, val_result, train_result);
        printf("val acc:%.2lf | lr:%.10lf, weight decay:%.10lf\n", val_result[i][EPOCHS-1], lr, weight_decay);

        dat[i].index = i;
        dat[i].val_acc = val_result[i][EPOCHS-1];
        dat[i].lr = lr;
        dat[i].weight_decay = weight_decay;
    }

    qsort(dat, OPTIMIZATION_TRIAL, sizeof(Data), comp);

    printf("=========== Hyper-Parameter Optimization Result ===========\n");
    for (int i = 0; i < 20; ++i) {
        printf("Best-%02d(val acc:%.2lf) | lr:%.10lf, weight decay:%.10lf\n", i + 1, dat[i].val_acc, dat[i].lr, dat[i].weight_decay);
    }

    return 0;   
}
