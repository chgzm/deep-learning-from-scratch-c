#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

uint8_t* read_file(const char* file_path);

uint8_t read_uint8(const uint8_t* addr, int* pos);
int32_t read_int32(const uint8_t* addr, int* pos);

int* choice(int size, int num);

#endif
