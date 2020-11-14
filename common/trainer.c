#include "trainer.h"
#include "util.h"
#include "optimizer.h"
#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>

Trainer* create_trainer(
    MultiLayerNet* net,
    double** train_images,
    uint8_t* train_labels,
    double** test_images,
    uint8_t* test_labels,
    int epochs,
    int mini_batch_size,
    int optimizer_type,
    int train_size,
    int test_size,
    double learning_rate,
    bool verbose
) {
    Trainer* trainer = malloc(sizeof(Trainer));

    trainer->net             = net;
    trainer->train_images    = train_images;
    trainer->train_labels    = train_labels;
    trainer->test_images     = test_images;
    trainer->test_labels     = test_labels;
    trainer->epochs          = epochs;
    trainer->mini_batch_size = mini_batch_size;
    trainer->optimizer_type  = optimizer_type;
    trainer->train_size      = train_size;
    trainer->test_size       = test_size;
    trainer->learning_rate   = learning_rate;
    trainer->verbose         = verbose;

    trainer->iter_per_epoch = train_size / mini_batch_size;
    trainer->max_iter = trainer->epochs * trainer->iter_per_epoch;

    trainer->train_acc_list = malloc(sizeof(double) * epochs);
    trainer->test_acc_list  = malloc(sizeof(double) * epochs);

    trainer->current_iter = 0;
    trainer->current_epoch = 0;

    return trainer;
}


static void trainer_train_step(Trainer* trainer) {
    int* batch_index = choice(trainer->train_size, trainer->mini_batch_size);

    Matrix* x_batch  = create_image_batch(trainer->train_images, batch_index, trainer->mini_batch_size);
    Vector* t_batch  = create_label_batch(trainer->train_labels, batch_index, trainer->mini_batch_size);

    multi_layer_net_gradient(trainer->net, x_batch, t_batch);

    for (int i = 0; i < trainer->net->hidden_layer_num + 1; ++i) {
        SGD_update_vector(trainer->net->b[i], trainer->net->A[i]->db, trainer->learning_rate);
        SGD_update_matrix(trainer->net->W[i], trainer->net->A[i]->dW, trainer->learning_rate);
    }

    if (trainer->current_iter % trainer->iter_per_epoch == 0) {
        const double train_acc = multi_layer_net_accuracy(trainer->net, trainer->train_images, trainer->train_labels, trainer->train_size);
        const double test_acc  = multi_layer_net_accuracy(trainer->net, trainer->test_images,  trainer->test_labels, trainer->test_size);

        if (trainer->verbose) {
            printf("epoch:%d train acc, test acc | %lf, %lf\n", trainer->current_epoch, train_acc, test_acc);
        }

        trainer->train_acc_list[trainer->current_epoch] = train_acc;
        trainer->test_acc_list[trainer->current_epoch]  = test_acc;

        ++(trainer->current_epoch);
    }

    ++(trainer->current_iter);

    free(batch_index);
    free_matrix(x_batch);
    free_vector(t_batch); 
}

void trainer_train(Trainer* trainer) {
    for (int i = 0; i < trainer->max_iter; ++i) {
        trainer_train_step(trainer);
    }

    const double test_acc = multi_layer_net_accuracy(trainer->net, trainer->test_images, trainer->test_labels, trainer->train_size);
    if (trainer->verbose) {
        printf("=============== Final Test Accuracy ===============\n");
        printf("test acc:%lf\n", test_acc);
    }
}

//
// TrainerExtend
//

TrainerExtend* create_trainer_extend(
    MultiLayerNetExtend* net,
    double** train_images,
    uint8_t* train_labels,
    double** test_images,
    uint8_t* test_labels,
    int epochs,
    int mini_batch_size,
    int optimizer_type,
    int train_size,
    int test_size,
    double learning_rate,
    bool verbose
) {
    TrainerExtend* trainer = malloc(sizeof(TrainerExtend));

    trainer->net             = net;
    trainer->train_images    = train_images;
    trainer->train_labels    = train_labels;
    trainer->test_images     = test_images;
    trainer->test_labels     = test_labels;
    trainer->epochs          = epochs;
    trainer->mini_batch_size = mini_batch_size;
    trainer->optimizer_type  = optimizer_type;
    trainer->train_size      = train_size;
    trainer->test_size       = test_size;
    trainer->learning_rate   = learning_rate;
    trainer->verbose         = verbose;

    trainer->iter_per_epoch = train_size / mini_batch_size;
    trainer->max_iter = trainer->epochs * trainer->iter_per_epoch;

    trainer->train_acc_list = malloc(sizeof(double) * epochs);
    trainer->test_acc_list  = malloc(sizeof(double) * epochs);

    trainer->current_iter = 0;
    trainer->current_epoch = 0;

    return trainer;
}


static void trainer_extend_train_step(TrainerExtend* trainer) {
    int* batch_index = choice(trainer->train_size, trainer->mini_batch_size);

    Matrix* x_batch  = create_image_batch(trainer->train_images, batch_index, trainer->mini_batch_size);
    Vector* t_batch  = create_label_batch(trainer->train_labels, batch_index, trainer->mini_batch_size);

    multi_layer_net_extend_gradient(trainer->net, x_batch, t_batch);

    for (int i = 0; i < trainer->net->hidden_layer_num + 1; ++i) {
        SGD_update_vector(trainer->net->b[i], trainer->net->A[i]->db, trainer->learning_rate);
        SGD_update_matrix(trainer->net->W[i], trainer->net->A[i]->dW, trainer->learning_rate);

        if (i != trainer->net->hidden_layer_num) {
            SGD_update_vector(trainer->net->gamma[i], trainer->net->B[i]->dg, trainer->learning_rate);
            SGD_update_vector(trainer->net->beta[i],  trainer->net->B[i]->db, trainer->learning_rate);
        }
    }


    if (trainer->current_iter % trainer->iter_per_epoch == 0) {
        const double train_acc = multi_layer_net_extend_accuracy(trainer->net, trainer->train_images, trainer->train_labels, trainer->train_size);
        const double test_acc  = multi_layer_net_extend_accuracy(trainer->net, trainer->test_images,  trainer->test_labels, trainer->test_size);

        if (trainer->verbose) {
            printf("epoch:%d train acc, test acc | %lf, %lf\n", trainer->current_epoch, train_acc, test_acc);
        }

        trainer->train_acc_list[trainer->current_epoch] = train_acc;
        trainer->test_acc_list[trainer->current_epoch]  = test_acc;

        ++(trainer->current_epoch);
    }

    ++(trainer->current_iter);

    free(batch_index);
    free_matrix(x_batch);
    free_vector(t_batch); 
}

void trainer_extend_train(TrainerExtend* trainer) {
    for (int i = 0; i < trainer->max_iter; ++i) {
        trainer_extend_train_step(trainer);
    }

    const double test_acc = multi_layer_net_extend_accuracy(trainer->net, trainer->test_images, trainer->test_labels, trainer->train_size);
    if (trainer->verbose) {
        printf("=============== Final Test Accuracy ===============\n");
        printf("test acc:%lf\n", test_acc);
    }
}

