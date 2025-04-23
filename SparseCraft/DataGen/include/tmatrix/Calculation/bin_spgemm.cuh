#pragma once
#include <tmatrix/DataStructure/TileMatrix.h>

__global__ void bin_spgemm_smem(
    const int *d_row_perm, int bin_offset,
    const MatIndex *A_row_ptr, const MatIndex *A_col_idx, const Tile *A_tiles, const char *A_data,
    const MatIndex *B_row_ptr, const MatIndex *B_col_idx, const Tile *B_tiles, const char *B_data,
    const MatIndex *C_row_ptr, const MatIndex *C_col_idx,       Tile *C_tiles,       char *C_data);

__global__ void bin_spgemm_gmem(
    const int *d_row_perm, int bin_offset, int*g_check, int*g_index, double*g_value, int hash_block_size,
    const int *A_row_ptr, const int *A_col_idx, const Tile *A_tiles, const char *A_data,
    const int *B_row_ptr, const int *B_col_idx, const Tile *B_tiles, const char *B_data,
    const int *C_row_ptr, const int *C_col_idx,       Tile *C_tiles,       char *C_data);

double bin_spgemm_call(sfBIN&, BaseMatrix *A, BaseMatrix *B, BaseMatrix *C, int*d_check, int*d_index, double*d_value, int table_size);