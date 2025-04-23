#pragma once

#include <tmatrix/DataStructure/TileMatrix.h>
// #define div_round_up(a, b) ((a % b == 0)? a / b : a / b + 1)


// #define WARP 32
#define PWARP 4
#define IMB_PWMIN 32
#define B_PWMIN 16
#define IMB_MIN 512
#define B_MIN 256
#define IMB_PW_SH_SIZE 2048
#define B_PW_SH_SIZE 1024
#define IMB_SH_SIZE 1024
#define B_SH_SIZE 512

#define HASH_SCAL 1

void bitmap256_2_fmt(const uint64_t* bitmap, char *bits, TileFormat fmt);

typedef struct {
    cudaStream_t stream[BIN_NUM];
    int bin_size[BIN_NUM];
    int bin_offset[BIN_NUM];
    int *d_bin_size;
    int *d_bin_offset;
    int *d_row_nz;
    int *d_row_perm;
    int max_intprod;
    int max_nz;
    int *d_max;
    char inited;
} sfBIN;

void init_bin(sfBIN *bin, int M);
void release_bin(sfBIN bin);
double set_max_bin(sfBIN *bin, int *d_arpt, int *d_acol, int *dbrpt, int M);
double set_row_nnz(sfBIN *bin, int *d_arpt, int *d_acol, Tile*d_atile, int *d_brpt, int *d_bcol, Tile*d_btile, int *d_crpt, int M, int *nnz);
double set_min_bin(sfBIN *bin, int M);
double set_min_bin_for_numeric(sfBIN *bin, int M);
double calculate_value_col_bin(int *d_arpt, int *d_acol, Tile*d_tileA, int *d_brpt, int *d_bcol, Tile*d_tileB, int *d_crpt, int *d_ccol, Tile*d_tileC, sfBIN *bin, int M, int N);

__global__ void tile_spgemm_step1_numeric_cuda_spa_kernel(int *d_blkrowptrA, int *d_blkcolidxA, Tile*d_bitmapA, int blkmA,
    int *d_blkrowptrB, int *d_blkcolidxB, Tile*d_bitmapB, int blknB,
    int *d_blkrowptrC, int *d_blkcolidxC, Tile*d_bitmapC, uint64_t*bitmap_spa_buffer);

__global__ void tile_spgemm_step1_cuda_spa_kernel(
    int *d_blkrowptrA, int *d_blkcolidxA, Tile*d_tileA, int blkmA, 
    int *d_blkrowptrB, int *d_blkcolidxB, Tile*d_tileB, int blknB,
    int *d_blkrowptrC);

__forceinline__ __device__ int sum_32_shfl(int sum)
{
#pragma unroll
    for (int mask = 32 / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    return sum;
}

__forceinline__ __device__
int scan_32_shfl(      int x,
                 const int lane_id)
{
    int y = __shfl_up_sync(0xffffffff, x, 1);
    x = lane_id >= 1 ? x + y : x;
    y = __shfl_up_sync(0xffffffff, x, 2);
    x = lane_id >= 2 ? x + y : x;
    y = __shfl_up_sync(0xffffffff, x, 4);
    x = lane_id >= 4 ? x + y : x;
    y = __shfl_up_sync(0xffffffff, x, 8);
    x = lane_id >= 8 ? x + y : x;
    y = __shfl_up_sync(0xffffffff, x, 16);
    x = lane_id >= 16 ? x + y : x;

    return x;
}