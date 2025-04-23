#pragma once

#include <tmatrix/common.h>

void bit256_2_uint64_arr(bit256 bitmap, uint64_t *arr);
void print_uint64(uint64_t *arr);
void print_bit256(bit256 bitmap);
void convert_b64_to_b256(const uint64_t *src, bit256 &dst);
int b64_multiply(const uint64_t *A, const uint64_t *B, uint64_t *C);