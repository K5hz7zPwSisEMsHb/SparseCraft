#pragma once

#include <math.h>
#include <stdio.h>
#include <bitset>
#include <string>
#include <cstdint>

#define BIN_NUM  4  //# 0～4，4～8，8～12，other
#define TILE_N   16
#define ALL_BITS 256

typedef std::bitset<ALL_BITS> bit256;

typedef unsigned char TileIndex;
typedef int MatIndex;
typedef double MatValue;

#define TileIndex_CSR_Get(arr, idx) (((arr)[(idx) >> 1] >> (((idx) & 1) << 2)) & 0x0f)
#define TileIndex_CSR_Set(arr, idx, val) ((arr)[(idx) >> 1] = (\
    ((arr)[(idx) >> 1] & ~(0x0f << (((idx) & 1) << 2))) | ((val) << (((idx) & 1) << 2))\
))
#define TileIndex_CSR_Or(arr, idx, val) ((arr)[(idx) >> 1] |= ((val) << (((idx) & 1) << 2)))

const std::string __mask_col_bits = "0000000000000001000000000000000100000000000000010000000000000001000000000000000100000000000000010000000000000001000000000000000100000000000000010000000000000001000000000000000100000000000000010000000000000001000000000000000100000000000000010000000000000001";
const bit256 mask_row = 0xffffull;
const bit256 mask_col(__mask_col_bits);

#define cudaCheckError(op) {cudaError_t err = op; if (err != cudaSuccess) {echo(error, "CUDA Error: %s", cudaGetErrorString(err)); exit(1);}}

enum _TileFormat
{
    COO,
    CSR,
    ELL,
    HYB,
    DRW,
    DCL,
    DNS,
    DIA
};

typedef uint8_t TileFormat;

bool file_exists(const char* filename);
void cudaInit(int device);
