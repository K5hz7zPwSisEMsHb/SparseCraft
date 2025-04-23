#pragma once
#include <tmatrix/common.h>
#define div_round_up(a, b) (((a) + ((b) - 1)) / (b))

__device__ __forceinline__ bool check_in_bitmap(const uint64_t *map, TileIndex idx)
{
    TileIndex row = idx >> 4, col = idx & 15;
    TileIndex row8 = row >> 3, col8 = col >> 3;
    row &= 7, col &= 7;
    return map[row8 << 1 | col8] & (1ULL << (row << 3 | col));
}

__device__ __forceinline__ void mma_m8n8k4(MatValue *acc, MatValue &frag_a, MatValue &frag_b)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"
        " { %0, %1 }, "
        " { %2 }, "
        " { %3 }, "
        " { %0, %1 };"
        : "+d"(acc[0]), "+d"(acc[1]) : "d"(frag_a), "d"(frag_b));
}

// __device__ __forceinline__ int check_bitmap_idx(const uint64_t *map, TileIndex idx)
// {
//     TileIndex row = idx >> 4, col = idx & 15;
//     TileIndex row8 = row >> 3, col8 = col >> 3;
//     TileIndex map_idx = row8 << 1 | col8;
//     row &= 7, col &= 7;
//     uint64_t mask = 1ULL << (row << 3 | col);
    
//     if (map[map_idx] & mask)
//     {
//         int sum = row8? __popcll(map[0]) + __popcll(map[1]) : 0;
//         for (int i = row8? 2: 0; i < map_idx; ++i) sum += __popcll(map[i] & ((1ULL << (row * 8)) - 1));
//         return sum + (__popcll(map[map_idx] & (mask - 1)));
//     }
//     return -1;
// }

// __device__ __forceinline__ int check_bitmap_idx(const uint64_t *map, TileIndex idx)
// {
//     TileIndex row = idx >> 4, col = idx & 15;
//     TileIndex row8 = row >> 3, col8 = col >> 3;
//     TileIndex map_idx = (row8 << 1) | col8;
//     row &= 7, col &= 7;
//     uint64_t mask = 1ULL << (row << 3 | col);

//     if (map[map_idx] & mask) {
//         int sum = row8 ? __popcll(map[0]) + __popcll(map[1]) : 0;
//         sum += col8? __popcll(map[map_idx - 1] & (1ULL << ((row + 1) * 8) - 1)): 0; // 位置在右边，把左边的加起来
//         sum += __popcll(map[map_idx] & (mask - 1)); // 位置在块内，把前面的加起来

//         return sum;
//     }
//     return -1;
// }

__device__ __forceinline__ uint16_t get_row_map(const uint64_t *map, TileIndex row)
{
    TileIndex row8 = row >> 3;
    row &= 7;
    TileIndex row_map_l = (map[row8 << 1] >> (row << 3)) & 0xff;
    TileIndex row_map_h = (map[row8 << 1 | 1] >> (row << 3)) & 0xff;
    uint16_t row_map = row_map_l | (row_map_h << 8);
    return row_map;
}

__device__ __forceinline__ int check_bitmap_idx_slow(const uint64_t *map, TileIndex idx)
{
    TileIndex row = idx >> 4, col = idx & 15;
    TileIndex row8 = row >> 3;
    uint16_t row_map = get_row_map(map, row);
    if (row_map & (1 << col))
    {
        int sum = row8 ? __popcll(map[0]) + __popcll(map[1]) : 0; // 位置在下边，把上面的加起来
        for (int i = row8 << 3; i < row; ++i) sum += __popc(get_row_map(map, i));
        return sum + (__popc(row_map & ((1 << col) - 1)));
    }
    return -1;
}

__device__ __forceinline__ int check_bitmap_row_idx(const uint16_t row_map, TileIndex col)
{
    uint16_t mask = 1 << col;
    return row_map & mask? (__popc(row_map & (mask - 1))) : -1;
}

template <typename T>
__device__ __host__ __forceinline__ MatIndex binary_find_index(const T *col_ptr, MatIndex len, T col)
{
    MatIndex left = 0, right = len - 1;
    while (left <= right)
    {
        MatIndex mid = (left + right) >> 1;
        if (col_ptr[mid] == col)
        {
            return mid;
        }
        else if (col_ptr[mid] < col)
            left = mid + 1;
        else
            right = mid - 1;
    }
    return -1;
}

__device__ __host__ __forceinline__ MatIndex binary_find_index_4bit(const TileIndex *col_ptr, MatIndex col, TileIndex start_off, TileIndex stop_off)
{
    MatIndex left = start_off, right = stop_off - 1;
    while (left <= right)
    {
        MatIndex mid = (left + right) >> 1;
        TileIndex mid_val = TileIndex_CSR_Get(col_ptr, mid);
        if (mid_val == col) return mid;
        else if (mid_val < col) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

__device__ __forceinline__ MatIndex binary_find_index_warp_level(const MatIndex *col_ptr, MatIndex len, MatIndex col, int lane_id)
{
    int idx = -1, mask;
    if (len <= 32)
    {
        idx = lane_id < len && col_ptr[lane_id] == col? lane_id : -1;
        mask = __ballot_sync(0xffffffff, idx != -1);
        if (mask)
            idx = __shfl_sync(0xffffffff, idx, __ffs(mask) - 1);
        return idx;
    } else {
        MatIndex thread_block = div_round_up(len, 32);
        MatIndex left = lane_id * thread_block, right = min((lane_id + 1) * thread_block, len - 1);
        
        while (left <= right)
        {
            MatIndex mid = (left + right) >> 1;
            if (col_ptr[mid] == col)
            {
                idx = mid;
                break;
            }
            else if (col_ptr[mid] < col)
                left = mid + 1;
            else
                right = mid - 1;
        }
        mask = __ballot_sync(0xffffffff, idx != -1);
        if (mask)
            idx = __shfl_sync(0xffffffff, idx, __ffs(mask) - 1);
    }
    return idx;
}