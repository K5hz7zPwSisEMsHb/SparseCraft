#pragma once

#include <math.h>
#include <stdio.h>
#include <bitset>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define BIN_NUM  7
#define BIN_NUM_N  4
#define TILE_N   16
#define ALL_BITS 256

typedef unsigned char TileIndex;
typedef int MatIndex;
typedef double MatValue;

#define TileIndex_CSR_Get(arr, idx) (((arr)[(idx) >> 1] >> (((idx) & 1) << 2)) & 0x0f)
#define TileIndex_CSR_Set(arr, idx, val) ((arr)[(idx) >> 1] = (\
    ((arr)[(idx) >> 1] & ~(0x0f << (((idx) & 1) << 2))) | ((val) << (((idx) & 1) << 2))\
))
#define TileIndex_CSR_Or(arr, idx, val) ((arr)[(idx) >> 1] |= ((val) << (((idx) & 1) << 2)))

#define cudaCheckError(op) {cudaError_t err = op; if (err != cudaSuccess) {echo(error, "Line %s:%d, CUDA Error: %s", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);}}

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

struct dense_tile
{
    uint64_t bitmap[4];
    MatValue val[TILE_N * TILE_N];

    dense_tile() {
        memset(bitmap, 0, 4 * sizeof(uint64_t));
        memset(val, 0, TILE_N * TILE_N * sizeof(MatValue));
    }
};

typedef uint8_t TileFormat;

bool file_exists(const char* filename);
void cudaInit(int device);
void cudaDebug(std::string&msg);