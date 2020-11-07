#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <byteswap.h>

uint8_t* read_file(const char* file_path) {
    FILE* fp = fopen(file_path, "r");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open \"%s\".\n", file_path);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    const int fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint8_t* addr = calloc(fsize, sizeof(char) + 1);
    const int r = fread(addr, sizeof(uint8_t), fsize, fp);
    fclose(fp);

    if (r != fsize) {
        fprintf(stderr, "fread failed.\n");
        return NULL;
    }

    return addr;
}

uint8_t read_uint8(const uint8_t* addr, int* pos) {
    const uint8_t v = addr[*pos];
    ++(*pos);
    return v;
}

int32_t read_int32(const uint8_t* addr, int* pos) {
    const int32_t v = __bswap_32(*(int32_t*)(&(addr[*pos])));
    *pos += 4;
    return v;
}

int* choice(int size, int num) {
    if (size < num) {
        fprintf(stderr, "Invalid size. %d and %d.\n", size, num);
        return NULL;
    }

    int vals[size];
    for (int i = 0; i < size; ++i) {
        vals[i] = i;
    }
    
    // shuffle
    for (int i = 0; i < size; ++i) {
        int j = rand() % size;
        int t = vals[i];
        vals[i] = vals[j];
        vals[j] = t;
    }

    int* ret = malloc(sizeof(int) * num); 
    memcpy(ret, vals, sizeof(int) * num);

    return ret;
}

double* logspace(double start, double stop, int num) {
    double* ret = malloc(sizeof(double) * num);

    double cur = start;
    double delta = (stop - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        ret[i] = pow(10, cur);
        cur += delta;
    }

    return ret;
}
