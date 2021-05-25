#ifndef TRAINER_H
#define TRAINER_H

#include "multi_layer_net.h"
#include "multi_layer_net_extend.h"
#include "simple_convnet.h"

typedef struct Trainer Trainer;
struct Trainer {
    MultiLayerNet* net;
    double** train_images;
    uint8_t* train_labels;
    double** test_images;
    uint8_t* test_labels;
    int epochs;
    int mini_batch_size;
    int optimizer_type;
    int train_size;
    int test_size;
    int iter_per_epoch;
    int max_iter;
    int current_iter;
    int current_epoch;
    double learning_rate;
    double* train_acc_list;
    double* test_acc_list;
    bool verbose;
};

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
);

void free_trainer(Trainer* trainer);

void trainer_train(Trainer* trainer);

//
// TrainerExtend
//

typedef struct TrainerExtend TrainerExtend;
struct TrainerExtend {
    MultiLayerNetExtend* net;
    double** train_images;
    uint8_t* train_labels;
    double** test_images;
    uint8_t* test_labels;
    int epochs;
    int mini_batch_size;
    int optimizer_type;
    int train_size;
    int test_size;
    int iter_per_epoch;
    int max_iter;
    int current_iter;
    int current_epoch;
    double learning_rate;
    double* train_acc_list;
    double* test_acc_list;
    bool verbose;
};

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
);

void free_trainer_extend(TrainerExtend* trainer);

void trainer_extend_train(TrainerExtend* trainer);

//
// SimpleConvnetTrainer
//

typedef struct SimpleConvNetTrainer SimpleConvNetTrainer;
struct SimpleConvNetTrainer {
    SimpleConvNet* net;
    double**** train_images;
    uint8_t* train_labels;
    double**** test_images;
    uint8_t* test_labels;
    int epochs;
    int mini_batch_size;
    int optimizer_type;
    int train_size;
    int test_size;
    int iter_per_epoch;
    int max_iter;
    int current_iter;
    int current_epoch;
    double learning_rate;
    double* train_acc_list;
    double* test_acc_list;
    bool verbose;
};

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
);

void free_simple_convnet_trainer(SimpleConvNetTrainer* trainer);

void simple_convnet_trainer_train(SimpleConvNetTrainer* trainer);

#endif
