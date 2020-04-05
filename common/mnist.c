#include "mnist.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>

double** load_mnist_images(const char* file_path) {
    uint8_t* addr = read_file(file_path);
    if (addr == NULL) {
        fprintf(stderr, "Failed to read file=\"%s\"\n", file_path);
        return NULL;
    }

    int pos = 0;
    const int magic = read_int32(addr, &pos);
    if (magic != IMAGE_MAGIC) {
        fprintf(stderr, "Invalid magic=0x%08x\n", magic);
        return NULL;
    }

    const int num_of_images = read_int32(addr, &pos);
    if (num_of_images != 10000 && num_of_images != 60000) {
        fprintf(stderr, "Invalid number of images=%d\n", num_of_images);
        return NULL;
    }

    const int num_of_rows = read_int32(addr, &pos);
    if (num_of_rows != NUM_OF_ROWS) {
        fprintf(stderr, "Invalid number of rows=%d\n", num_of_rows);
        return NULL;
    }

    const int num_of_cols = read_int32(addr, &pos);
    if (num_of_cols != NUM_OF_COLS) {
        fprintf(stderr, "Invalid number of cols=%d\n", num_of_cols);
        return NULL;
    }

    double** imgs = malloc(sizeof(double*) * num_of_images);  
    for (int i = 0; i < num_of_images; ++i) {
        imgs[i] = malloc(sizeof(double) * NUM_OF_PIXELS);
        for (int j = 0; j < NUM_OF_PIXELS; ++j) {
            const uint8_t d = read_uint8(addr, &pos);      
            imgs[i][j] = d / 255.0;
        }
    }

    free(addr);
    return imgs;
}

uint8_t* load_mnist_labels(const char* file_path) {
    uint8_t* addr = read_file(file_path);
    if (addr == NULL) {
        fprintf(stderr, "Failed to read file=\"%s\"\n", file_path);
        return NULL;
    }

    int pos = 0;
    const int magic = read_int32(addr, &pos);
    if (magic != LABEL_MAGIC) {
        fprintf(stderr, "Invalid magic=0x%08x\n", magic);
        return NULL;
    }

    const int num_of_labels = read_int32(addr, &pos);
    uint8_t* labels = calloc(num_of_labels, sizeof(uint8_t));
    for (int i = 0; i < num_of_labels; ++i) {
        const uint8_t v = read_uint8(addr, &pos);
        labels[i] = v;
    }
         
    free(addr);  
    return labels;
}
