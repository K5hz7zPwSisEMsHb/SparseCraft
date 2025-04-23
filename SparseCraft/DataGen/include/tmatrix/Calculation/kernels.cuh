#pragma once

#include <tmatrix/common.h>
#include <tmatrix/DataStructure/TileMatrix.h>

template<typename IT, typename VT>
__device__ __forceinline__ VT spm_dv(const Tile* A_tiles, const char*A_data, const IT*A_colind, const VT *x, VT *y, int&lane_id, IT&start_aj, IT&end_aj, VT*warp_x, VT*warp_y)
{
    int vwarp_id = lane_id >> 1, hwarp_id = lane_id >> 4;
    int vlane_id = lane_id & 1, hlane_id = lane_id & 15;
    VT sumsum = 0;
    for (IT Aj = start_aj; Aj < end_aj; ++Aj)
    {
        TileIndex *bits = (TileIndex*)(A_data + A_tiles[Aj].bits_off);
        const VT *A = (VT*)(A_data + A_tiles[Aj].bits_off + A_tiles[Aj].bitslen * 16), *dx = x + (A_colind[Aj] << 4);

        switch (A_tiles[Aj].fmt)
        {
            case COO:
                {
                    int nnz = bits[0] + 1;
                    for (int i = lane_id; i < nnz; i += 32)
                    {
                        TileIndex idx = bits[i + 1], row = idx >> 4, col = idx & 0xf;
                        atomicAdd(warp_y + row, A[i] * dx[col]);
                    }
                }
                break;
            case CSR:
                {
                    if (lane_id < TILE_N)
                    {
                        warp_x[lane_id] = dx[lane_id];
                    }

                    TileIndex *colidx = bits + TILE_N + 1, start = bits[vwarp_id], stop = bits[vwarp_id + 1];
                    VT sum = 0;

                    for (int j = start + vlane_id; j < stop; j += 2)
                    {
                        sum += warp_x[TileIndex_CSR_Get(colidx, j)] * A[j];
                    }
                    sum += __shfl_down_sync(0xffffffff, sum, 1);
                    sumsum += __shfl_down_sync(0xffffffff, sum, lane_id);
                }
                break;
            case ELL:
                {
                    TileIndex max_row_len = TileIndex_CSR_Get(bits, 0) + 1;
                    VT sum = 0;
                    if (lane_id < TILE_N) warp_x[lane_id] = dx[lane_id];

                    for (int j = vwarp_id * max_row_len + vlane_id; j < (vwarp_id + 1) * max_row_len; j += 2)
                    {
                        TileIndex col = TileIndex_CSR_Get(bits, 1 + j);
                        sum += A[j] * warp_x[col];
                    }
                    sum += __shfl_down_sync(0xffffffff, sum, 1);
                    sumsum += __shfl_down_sync(0xffffffff, sum, lane_id);
                }
                break;
            case HYB:
                {
                    TileIndex min_row_len = bits[0], coo_cnt = bits[1];
                    IT coo_start = (min_row_len * 16 + 1) / 2 + 2;

                    VT sum = 0;
                    if (lane_id < TILE_N) warp_x[lane_id] = dx[lane_id];
                    for (int j = vwarp_id * min_row_len + vlane_id; j < (vwarp_id + 1) * min_row_len; j += 2)
                    {
                        TileIndex col = TileIndex_CSR_Get(bits, 4 + j);
                        sum += A[j] * warp_x[col];
                    }
                    sum += __shfl_down_sync(0xffffffff, sum, 1);
                    sumsum += __shfl_down_sync(0xffffffff, sum, lane_id);

                    for (int j = lane_id; j < coo_cnt; j += 32)
                    {
                        TileIndex idx = bits[coo_start + j], row = idx >> 4, col = idx & 0xf;
                        atomicAdd(warp_y + row, A[j + min_row_len * 16] * warp_x[col]);
                    }
                }
                break;
            case DRW:
                {
                    TileIndex cnt = TileIndex_CSR_Get(bits, 0) + 1;
                    VT rx = dx[hlane_id];
                    for (int j = hwarp_id; j < cnt; j += 2)
                    {
                        VT sum = A[j * 16 + hlane_id] * rx;

                        sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 1);
                        sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 2);
                        sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 4);
                        sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 8);

                        if (hlane_id == 0) {
                            TileIndex row = TileIndex_CSR_Get(bits, 1 + j);
                            warp_y[row] += sum;
                        }
                    }
                }
                break;
            case DCL:
                {
                    TileIndex cnt = TileIndex_CSR_Get(bits, 0) + 1;
                    VT sum = 0;
                    for (int j = hwarp_id; j < cnt; j += 2)
                    {
                        TileIndex col = TileIndex_CSR_Get(bits, 1 + j);
                        sum += A[j * 16 + hlane_id] * dx[col];
                    }
                    sum += __shfl_down_sync(0xffffffff, sum, 16);
                    sumsum += sum;
                }
                break;
            case DNS:
                {
                    VT sum = 0, rx = lane_id < 16? dx[hlane_id]: 0;

                    sum += A[hwarp_id + hlane_id * 16 + 0] * __shfl_sync(0xffffffff, rx, hwarp_id + 0);
                    sum += A[hwarp_id + hlane_id * 16 + 2] * __shfl_sync(0xffffffff, rx, hwarp_id + 2);
                    sum += A[hwarp_id + hlane_id * 16 + 4] * __shfl_sync(0xffffffff, rx, hwarp_id + 4);
                    sum += A[hwarp_id + hlane_id * 16 + 6] * __shfl_sync(0xffffffff, rx, hwarp_id + 6);
                    sum += A[hwarp_id + hlane_id * 16 + 8] * __shfl_sync(0xffffffff, rx, hwarp_id + 8);
                    sum += A[hwarp_id + hlane_id * 16 + 10] * __shfl_sync(0xffffffff, rx, hwarp_id + 10);
                    sum += A[hwarp_id + hlane_id * 16 + 12] * __shfl_sync(0xffffffff, rx, hwarp_id + 12);
                    sum += A[hwarp_id + hlane_id * 16 + 14] * __shfl_sync(0xffffffff, rx, hwarp_id + 14);
                    sumsum += sum + __shfl_down_sync(0xffffffff, sum, 16);
                }
                break;
            default:
                break;
        }
    }
 
    if (lane_id < TILE_N) 
    {
        sumsum += warp_y[lane_id];
    }
    return sumsum;
}
