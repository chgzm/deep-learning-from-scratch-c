#include "trainer.h"
#include "util.h"
#include "optimizer.h"
#include "mnist.h"
#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>

//
// for Adam
//

static Matrix4d* m1;
static Matrix4d* m2;

static Matrix* n1;
static Matrix* n2;
static Matrix* o1;
static Matrix* o2;
                 
static Vector* u1;
static Vector* u2;
static Vector* v1;
static Vector* v2;
static Vector* w1;
static Vector* w2;

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

void free_trainer(Trainer* trainer) {
    free_multi_layer_net(trainer->net);

    free(trainer->train_acc_list);
    free(trainer->test_acc_list);

    free(trainer);
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

    const double test_acc = multi_layer_net_accuracy(trainer->net, trainer->test_images, trainer->test_labels, trainer->test_size);
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

    const double test_acc = multi_layer_net_extend_accuracy(trainer->net, trainer->test_images, trainer->test_labels, trainer->test_size);
    if (trainer->verbose) {
        printf("=============== Final Test Accuracy ===============\n");
        printf("test acc:%lf\n", test_acc);
    }
}

//
// SimpleConvNet
//

SimpleConvNetTrainer* create_simple_convnet_trainer(
    SimpleConvNet* net,
    double**** train_images,
    uint8_t* train_labels,
    double**** test_images,
    uint8_t* test_labels,
    int epochs,
    int mini_batch_size,
    int optimizer_type,
    int train_size,
    int test_size,
    double learning_rate,
    bool verbose
) {
    SimpleConvNetTrainer* trainer = malloc(sizeof(SimpleConvNetTrainer));

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

void free_simple_convnet_trainer(SimpleConvNetTrainer* trainer) {
    free_simple_convnet(trainer->net);

    free(trainer->train_acc_list);
    free(trainer->test_acc_list);

    free(trainer);
}

static void simple_convnet_trainer_train_step(SimpleConvNetTrainer* trainer, int iter_num) {
    int* batch_index = choice(trainer->train_size, trainer->mini_batch_size);
    Matrix4d* x_batch = create_image_batch_4d(trainer->train_images, batch_index, trainer->mini_batch_size);
    Vector* t_batch = create_label_batch(trainer->train_labels, batch_index, trainer->mini_batch_size);

    simple_convnet_gradient(trainer->net, x_batch, t_batch);

    switch (trainer->optimizer_type) {
    case SGD: {
        SGD_update_vector(trainer->net->C->b, trainer->net->C->db, trainer->learning_rate);
        SGD_update_matrix_4d(trainer->net->C->W, trainer->net->C->dW, trainer->learning_rate);

        SGD_update_vector(trainer->net->A[0]->b, trainer->net->A[0]->db, trainer->learning_rate);
        SGD_update_matrix(trainer->net->A[0]->W, trainer->net->A[0]->dW, trainer->learning_rate);

        SGD_update_vector(trainer->net->A[1]->b, trainer->net->A[1]->db, trainer->learning_rate);
        SGD_update_matrix(trainer->net->A[1]->W, trainer->net->A[1]->dW, trainer->learning_rate);

        break;
    }
    case Adam: {
        static const double beta1 = 0.9;
        static const double beta2 = 0.999;

        Adam_update_vector(trainer->net->C->b,  trainer->net->C->db,    trainer->learning_rate, beta1, beta2, u1, u2, iter_num);
        Adam_update_matrix_4d(trainer->net->C->W, trainer->net->C->dW,    trainer->learning_rate, beta1, beta2, m1, m2, iter_num);

        Adam_update_vector(trainer->net->A[0]->b,  trainer->net->A[0]->db, trainer->learning_rate, beta1, beta2, v1, v2, iter_num);
        Adam_update_matrix(trainer->net->A[0]->W,  trainer->net->A[0]->dW, trainer->learning_rate, beta1, beta2, n1, n2, iter_num);

        Adam_update_vector(trainer->net->A[1]->b,  trainer->net->A[1]->db, trainer->learning_rate, beta1, beta2, w1, w2, iter_num);
        Adam_update_matrix(trainer->net->A[1]->W,  trainer->net->A[1]->dW, trainer->learning_rate, beta1, beta2, o1, o2, iter_num);

        break;
    }
    default: {
        fprintf(stderr, "Invalid optimizer type.\n");
        break;
    }
    }

    if (trainer->verbose) {
        const double loss = simple_convnet_loss(trainer->net, x_batch, t_batch);
        printf("train loss:%lf\n", loss); 
    }

    if (trainer->current_iter % trainer->iter_per_epoch == 0) {
        const double train_acc = simple_convnet_accuracy(trainer->net, trainer->train_images, trainer->train_labels, trainer->train_size, 1);
        const double test_acc  = simple_convnet_accuracy(trainer->net, trainer->test_images,  trainer->test_labels, trainer->test_size, 1);

        if (trainer->verbose) {
            printf("epoch:%d train acc, test acc | %lf, %lf\n", trainer->current_epoch, train_acc, test_acc);
        }

        trainer->train_acc_list[trainer->current_epoch] = train_acc;
        trainer->test_acc_list[trainer->current_epoch]  = test_acc;

        ++(trainer->current_epoch);
    }

    ++(trainer->current_iter);

    free(batch_index);
    free_matrix_4d(x_batch);
    free_vector(t_batch); 
}

void simple_convnet_trainer_train(SimpleConvNetTrainer* trainer) {
    if (trainer->optimizer_type == Adam) {
        m1 = create_matrix_4d(
            trainer->net->C->W->sizes[0], 
            trainer->net->C->W->sizes[1], 
            trainer->net->C->W->sizes[2], 
            trainer->net->C->W->sizes[3]
        );
        m2 = create_matrix_4d(
            trainer->net->C->W->sizes[0], 
            trainer->net->C->W->sizes[1], 
            trainer->net->C->W->sizes[2], 
            trainer->net->C->W->sizes[3]
        );

        n1 = create_matrix(trainer->net->A[0]->W->rows, trainer->net->A[0]->W->cols); 
        n2 = create_matrix(trainer->net->A[0]->W->rows, trainer->net->A[0]->W->cols); 
        o1 = create_matrix(trainer->net->A[1]->W->rows, trainer->net->A[1]->W->cols); 
        o2 = create_matrix(trainer->net->A[1]->W->rows, trainer->net->A[1]->W->cols); 

        u1 = create_vector(trainer->net->C->b->size);  
        u2 = create_vector(trainer->net->C->b->size);  
        v1 = create_vector(trainer->net->A[0]->b->size);  
        v2 = create_vector(trainer->net->A[0]->b->size);  
        w1 = create_vector(trainer->net->A[1]->b->size);  
        w2 = create_vector(trainer->net->A[1]->b->size);  
    }

    for (int i = 0; i < trainer->max_iter; ++i) {
        simple_convnet_trainer_train_step(trainer, i);
    }

    const double test_acc = simple_convnet_accuracy(trainer->net, trainer->test_images, trainer->test_labels, trainer->test_size, 1);
    if (trainer->verbose) {
        printf("=============== Final Test Accuracy ===============\n");
        printf("test acc:%lf\n", test_acc);
    }
}
