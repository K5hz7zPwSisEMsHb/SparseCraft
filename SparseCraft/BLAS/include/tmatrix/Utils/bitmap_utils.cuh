#pragma once

#include <tmatrix/common.h>

void print_uint64(const uint64_t *arr);
bool bitmap_check(const uint64_t *bitmap, TileIndex row, TileIndex col);
bool bitmap_check(const uint64_t *bitmap, TileIndex idx);
uint16_t bitmap_count(const uint64_t *bitmap);
TileIndex b64_count_row(const uint64_t*bitmap, TileIndex row);
TileIndex b64_count_col(const uint64_t*bitmap, TileIndex col);
uint16_t bitmap_get_row(const uint64_t *bitmap, TileIndex row);
uint16_t bitmap_get_col(const uint64_t *bitmap, TileIndex col);

TileIndex host_bitmap2codelen(const uint64_t*bitmap, MatIndex*nnz, TileFormat format);
// Device version for GPU execution
// TileIndex device_bitmap2codelen(const uint64_t* bitmap, int* codelen, int n);
