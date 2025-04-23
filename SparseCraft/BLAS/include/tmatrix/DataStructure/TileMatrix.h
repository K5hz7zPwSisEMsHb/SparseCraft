#pragma once

#include <tmatrix/common.h>

struct Tile
{
    uint64_t bits_off;
    uint64_t vals_off;
    uint64_t bitmap[4];
    TileFormat fmt;
    uint8_t bitslen, valslen;

    Tile() : bits_off(0), bitslen(0), valslen(0), fmt(COO) {
        bitmap[0] = bitmap[1] = bitmap[2] = bitmap[3] = 0;
    }
};

struct BaseMatrix
{
    MatIndex meta_m, meta_n, meta_nnz;
    MatIndex _m, _n, _nnz;
    uint64_t _data_len;

    MatIndex  *tile_row_ptr;
    MatIndex  *tile_col_idx;
    Tile      *tiles;
    char      *data;
};

struct BaseMatrixCSC
{
    MatIndex meta_m, meta_n, meta_nnz;
    MatIndex _m, _n, _nnz;
    uint64_t _data_len;

    MatIndex  *tile_col_ptr;
    MatIndex  *tile_row_idx;
    Tile      *tiles;
    char      *data;
};