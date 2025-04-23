#pragma once
#include "./common.cuh"
#include <tmatrix/DataStructure/TileMatrix.h>


__device__ __forceinline__ void hyb_x_coo_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int nzB = Bbits[0] + 1, nzC = Cbits[0] + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        int j = 0;
        for (; j < nzB; ++j)
            if (Acol == Bbits[1 + j] >> 4)
                break;
        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rC[Bcol] += Aval * Bvals[j];
            ++j;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        int idx = Abits[i + Aelllen], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        int j = 0;
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;

        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            int idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | Bcol);
            if (idx != -1) atomicAdd(Cvals + idx, Aval * Bvals[j]);
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < 16; ++i)
        {
            if (rC[i] != 0)
            {
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, vwarp_id << 4 | i);
                if (idx != -1) Cvals[idx] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void hyb_x_coo_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int nzB = Bbits[0] + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        int j = 0;
        for (; j < nzB; ++j)
            if (Acol == Bbits[1 + j] >> 4)
                break;
        while (Acol == Bbits[1 + j] >> 4 && j < nzB)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rC[Bcol] += Aval * Bvals[j];
            ++j;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        int idx = Abits[i + Aelllen], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        int j = 0;
        uint16_t row_map = get_row_map(Cbitmap, Arow);
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;

        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Cbits[Arow] + idx, Aval * Bvals[j]);
            ++j;
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = Cbits[vwarp_id]; i < Cbits[vwarp_id + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rC[col];
        }
    }
}

__device__ __forceinline__ void hyb_x_coo_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int nzB = Bbits[0] + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        int j = 0;
        for (; j < nzB; ++j)
            if (Acol == Bbits[1 + j] >> 4)
                break;
        while (Acol == Bbits[1 + j] >> 4 && j < nzB)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rC[Bcol] += Aval * Bvals[j];
            ++j;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        int idx = Abits[i + Aelllen], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        
        uint16_t row_map = get_row_map(Cbitmap, Arow);
        int j = 0;
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;

        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Arow * max_row_len_C + idx, Aval * Bvals[j]);
            ++j;
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = vwarp_id * max_row_len_C; i < (vwarp_id + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void hyb_x_coo_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    int nzB = Bbits[0] + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        int j = 0;
        for (; j < nzB; ++j)
            if (Acol == Bbits[1 + j] >> 4)
                break;
        while (Acol == Bbits[1 + j] >> 4 && j < nzB)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rC[Bcol] += Aval * Bvals[j];
            ++j;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        int idx = Abits[i + Aelllen], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        int j = 0;
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;

        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            int idx_ell = binary_find_index_4bit(Cbits + 2, Bcol, Arow * min_row_len_C, (Arow + 1) * min_row_len_C);
            int idx_coo = binary_find_index<TileIndex>(Cbits + Celllen, coo_cnt_C, Arow << 4 | Bcol);
            if (idx_ell != -1) atomicAdd(Cvals + idx_ell, Aval * Bvals[j]);
            if (idx_coo != -1) atomicAdd(Cvals + 16 * min_row_len_C + idx_coo, Aval * Bvals[j]);
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < min_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, vwarp_id * min_row_len_C + i);
            Cvals[vwarp_id * min_row_len_C + i] += rC[Ccol];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (vwarp_id == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && vwarp_id == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rC[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void hyb_x_coo_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int nzB = Bbits[0] + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        int j = 0;
        for (; j < nzB; ++j)
            if (Acol == Bbits[1 + j] >> 4)
                break;
        while (Acol == Bbits[1 + j] >> 4 && j < nzB)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rC[Bcol] += Aval * Bvals[j];
            ++j;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        int idx = Abits[i + Aelllen], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        int k = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        int j = 0;
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;

        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            atomicAdd(Cvals + k + Bcol, Aval * Bvals[j]);
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, vwarp_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void hyb_x_coo_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int nzB = Bbits[0] + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    uint16_t col_map = 0;
    for (int i = 0; i < cntC; ++i) col_map |= 1 << TileIndex_CSR_Get(Cbits, 1 + i);

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        int j = 0;
        for (; j < nzB; ++j)
            if (Acol == Bbits[1 + j] >> 4)
                break;
        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rC[Bcol] += Aval * Bvals[j];
            ++j;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        int idx = Abits[i + Aelllen], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        int j = 0;
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;

        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            uint16_t col_mask = 1 << Bcol;
            if (col_mask & col_map)
            {
                int idx = __popc(col_map & (col_mask - 1));
                atomicAdd(Cvals + Arow + 16 * idx, Aval * Bvals[j]);
            }
            ++j;
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rC[Ccol] += __shfl_down_sync(0xffffffff, rC[Ccol], 1);
        if (vlane_id == 0) Cvals[vwarp_id + i * 16] += rC[Ccol];
    }
}

__device__ __forceinline__ void hyb_x_coo_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int cntB = Bbits[0] + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        int j = 0;
        for (; j < cntB; ++j)
            if (Acol == Bbits[1 + j] >> 4)
                break;
        while (Acol == Bbits[1 + j] >> 4 && j < cntB)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rC[Bcol] += Aval * Bvals[j];
            ++j;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        int j = 0;
        for (; j < cntB; ++j)
            if (Acol == Bbits[1 + j] >> 4)
                break;
        while (Acol == Bbits[1 + j] >> 4 && j < cntB)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            atomicAdd(Cvals + Arow * 16 + Bcol, Aval * Bvals[j]);
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < 16; ++i)
            if (rC[i])
                Cvals[vwarp_id * 16 + i] += rC[i];
    }
}

__device__ __forceinline__ void hyb_x_csr_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int nzC = Cbits[0] + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        int idx = Abits[i + Aelllen], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            int idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | Bcol);
            if (idx != -1) atomicAdd(Cvals + idx, Aval * Bvals[j]);
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < 16; ++i)
        {
            if (rC[i] != 0)
            {
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, vwarp_id << 4 | i);
                if (idx != -1) Cvals[idx] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void hyb_x_csr_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        int idx = Abits[i + Aelllen], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        uint16_t row_map = get_row_map(Cbitmap, Arow);

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Cbits[Arow] + idx, Aval * Bvals[j]);
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = Cbits[vwarp_id]; i < Cbits[vwarp_id + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rC[col];
        }
    }
}

__device__ __forceinline__ void hyb_x_csr_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        int idx = Abits[i + Aelllen], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        uint16_t row_map = get_row_map(Cbitmap, Arow);
        
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Arow * max_row_len_C + idx, Aval * Bvals[j]);
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = vwarp_id * max_row_len_C; i < (vwarp_id + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void hyb_x_csr_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        int idx = Abits[i + Aelllen], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        uint16_t row_map = get_row_map(Cbitmap, vwarp_id);
        
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            int idx_ell = binary_find_index_4bit(Cbits + 2, Bcol, Arow * min_row_len_C, (Arow + 1) * min_row_len_C);
            int idx_coo = binary_find_index<TileIndex>(Cbits + Celllen, coo_cnt_C, Arow << 4 | Bcol);
            if (idx_ell != -1) atomicAdd(Cvals + idx_ell, Aval * Bvals[j]);
            if (idx_coo != -1) atomicAdd(Cvals + 16 * min_row_len_C + idx_coo, Aval * Bvals[j]);
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < min_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, vwarp_id * min_row_len_C + i);
            Cvals[vwarp_id * min_row_len_C + i] += rC[Ccol];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (vwarp_id == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && vwarp_id == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rC[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void hyb_x_csr_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        int idx = Abits[i + Aelllen], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        int k = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            atomicAdd(Cvals + k + Bcol, Aval * Bvals[j]);
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, vwarp_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void hyb_x_csr_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    uint16_t col_map = 0;
    for (int i = 0; i < cntC; ++i) col_map |= 1 << TileIndex_CSR_Get(Cbits, 1 + i);

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[i + Aelllen], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            uint16_t col_mask = 1 << Bcol;
            if (col_mask & col_map)
            {
                int idx = __popc(col_map & (col_mask - 1));
                atomicAdd(Cvals + Arow + idx * 16, Aval * Bvals[j]);
            }
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rC[Ccol] += __shfl_down_sync(0xffffffff, rC[Ccol], 1);
        if (vlane_id == 0) Cvals[vwarp_id + i * 16] += rC[Ccol];
    }
}


__device__ __forceinline__ void hyb_x_csr_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            atomicAdd(Cvals + Arow * 16 + Bcol, Aval * Bvals[j]);
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < 16; ++i)
            if (rC[i])
                Cvals[vwarp_id * 16 + i] += rC[i];
    }
}

__device__ __forceinline__ void hyb_x_ell_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int nzC = Cbits[0] + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            int idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | Bcol);
            if (idx != -1) atomicAdd(Cvals + idx, Aval * Bvals[j]);
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < 16; ++i)
        {
            MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, vwarp_id << 4 | i);
            if (idx != -1) Cvals[idx] += rC[i];
        }
    }
}

__device__ __forceinline__ void hyb_x_ell_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        uint16_t row_map = get_row_map(Cbitmap, Arow);

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Cbits[Arow] + idx, Aval * Bvals[j]);
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = Cbits[vwarp_id]; i < Cbits[vwarp_id + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rC[col];
        }
    }
}

__device__ __forceinline__ void hyb_x_ell_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        uint16_t row_map = get_row_map(Cbitmap, Arow);

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Arow * max_row_len_C + idx, Aval * Bvals[j]);
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = vwarp_id * max_row_len_C; i < (vwarp_id + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void hyb_x_ell_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        
        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        int idx = Abits[i + Aelllen], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        uint16_t row_map = get_row_map(Cbitmap, vwarp_id);
        
        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            int idx_ell = binary_find_index_4bit(Cbits + 2, Bcol, Arow * min_row_len_C, (Arow + 1) * min_row_len_C);
            int idx_coo = binary_find_index<TileIndex>(Cbits + Celllen, coo_cnt_C, Arow << 4 | Bcol);
            if (idx_ell != -1) atomicAdd(Cvals + idx_ell, Aval * Bvals[j]);
            if (idx_coo != -1) atomicAdd(Cvals + 16 * min_row_len_C + idx_coo, Aval * Bvals[j]);
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < min_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, vwarp_id * min_row_len_C + i);
            Cvals[vwarp_id * min_row_len_C + i] += rC[Ccol];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (vwarp_id == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && vwarp_id == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rC[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void hyb_x_ell_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        
        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        int idx = Abits[i + Aelllen], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        int k = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            atomicAdd(Cvals + k + Bcol, Aval * Bvals[j]);
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, vwarp_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void hyb_x_ell_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    uint16_t col_map = 0;
    for (int i = 0; i < cntC; ++i) col_map |= 1 << TileIndex_CSR_Get(Cbits, 1 + i);

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        
        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[i + Aelllen], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            uint16_t col_mask = 1 << Bcol;
            if (col_mask & col_map)
            {
                int idx = __popc(col_map & (col_mask - 1));
                atomicAdd(Cvals + Arow + idx * 16, Aval * Bvals[j]);
            }
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rC[Ccol] += __shfl_down_sync(0xffffffff, rC[Ccol], 1);
        if (vlane_id == 0) Cvals[vwarp_id + i * 16] += rC[Ccol];
    }
}

__device__ __forceinline__ void hyb_x_ell_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            atomicAdd(Cvals + Arow * 16 + Bcol, Aval * Bvals[j]);
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < 16; ++i)
            if (rC[i])
                Cvals[vwarp_id * 16 + i] += rC[i];
    }
}

__device__ __forceinline__ void hyb_x_hyb_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int nzC = Cbits[0] + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            rC[Bcol] += Aval * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rC[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        if (Aval == 0)
            continue;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            int idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | Bcol);
            if (idx != -1) atomicAdd(Cvals + idx, Aval * Bvals[j]);
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            int idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | Bcol);
            if (idx != -1) atomicAdd(Cvals + idx, Aval * Bvals[j + min_row_len_B * 16]);
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < 16; ++i)
            if (rC[i]) {
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, vwarp_id << 4 | i);
                if (idx != -1) Cvals[idx] += rC[i];
            }
    }
}

__device__ __forceinline__ void hyb_x_hyb_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            rC[Bcol] += Aval * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rC[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        if (Aval == 0)
            continue;

        uint16_t row_map = get_row_map(Cbitmap, Arow);

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Cbits[Arow] + idx, Aval * Bvals[j]);
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Cbits[Arow] + idx, Aval * Bvals[j + min_row_len_B * 16]);
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = Cbits[vwarp_id]; i < Cbits[vwarp_id + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rC[col];
        }
    }
}

__device__ __forceinline__ void hyb_x_hyb_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            rC[Bcol] += Aval * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rC[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        if (Aval == 0)
            continue;

        uint16_t row_map = get_row_map(Cbitmap, Arow);

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, j);
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Arow * max_row_len_C + idx, Aval * Bvals[j]);
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Arow * max_row_len_C + idx, Aval * Bvals[j + min_row_len_B * 16]);
            ++j;
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = vwarp_id * max_row_len_C; i < (vwarp_id + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void hyb_x_hyb_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, j);
            rC[Bcol] += Aval * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rC[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        if (Aval == 0)
            continue;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, j);
            int idx_ell = binary_find_index_4bit(Cbits + 2, Bcol, Arow * min_row_len_C, (Arow + 1) * min_row_len_C);
            int idx_coo = binary_find_index<TileIndex>(Cbits + Celllen, coo_cnt_C, Arow << 4 | Bcol);
            if (idx_ell != -1) atomicAdd(Cvals + idx_ell, Aval * Bvals[j]);
            if (idx_coo != -1) atomicAdd(Cvals + 16 * min_row_len_C + idx_coo, Aval * Bvals[j]);
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (j < coo_cnt_B && Acol == Bbits[Belllen + j] >> 4)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            int idx_ell = binary_find_index_4bit(Cbits + 2, Bcol, Arow * min_row_len_C, (Arow + 1) * min_row_len_C);
            int idx_coo = binary_find_index<TileIndex>(Cbits + Celllen, coo_cnt_C, Arow << 4 | Bcol);
            if (idx_ell != -1) atomicAdd(Cvals + idx_ell, Aval * Bvals[j + min_row_len_B * 16]);
            if (idx_coo != -1) atomicAdd(Cvals + 16 * min_row_len_C + idx_coo, Aval * Bvals[j + min_row_len_B * 16]);
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < min_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, vwarp_id * min_row_len_C + i);
            Cvals[vwarp_id * min_row_len_C + i] += rC[Ccol];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (vwarp_id == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && vwarp_id == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rC[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void hyb_x_hyb_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        if (Aval == 0) continue;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            rC[Bcol] += Aval * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rC[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        if (Aval == 0) continue;

        int k = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            atomicAdd(Cvals + k + Bcol, Aval * Bvals[j]);
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            atomicAdd(Cvals + k + Bcol, Aval * Bvals[j + min_row_len_B * 16]);
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, vwarp_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void hyb_x_hyb_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    uint16_t col_map = 0;
    for (int i = 0; i < cntC; ++i) col_map |= 1 << TileIndex_CSR_Get(Cbits, 1 + i);

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        
        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            rC[Bcol] += Aval * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (j < coo_cnt_B && Acol == Bbits[Belllen + j] >> 4)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rC[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        if (Aval == 0) continue;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            uint16_t col_mask = 1 << Bcol;
            if (col_mask & col_map)
            {
                int idx = __popc(col_map & (col_mask - 1));
                atomicAdd(Cvals + Arow + idx * 16, Aval * Bvals[j]);
            }
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (j < coo_cnt_B && Acol == Bbits[Belllen + j] >> 4)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            uint16_t col_mask = 1 << Bcol;
            if (col_mask & col_map)
            {
                int idx = __popc(col_map & (col_mask - 1));
                atomicAdd(Cvals + Arow + idx * 16, Aval * Bvals[j + min_row_len_B * 16]);
            }
            ++j;
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rC[Ccol] += __shfl_down_sync(0xffffffff, rC[Ccol], 1);
        if (vlane_id == 0) Cvals[i * 16 + vwarp_id] += rC[Ccol];
    }
}

__device__ __forceinline__ void hyb_x_hyb_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            rC[Bcol] += Aval * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rC[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        if (Aval == 0)
            continue;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            atomicAdd(Cvals + Arow * 16 + Bcol, Aval * Bvals[j]);
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            atomicAdd(Cvals + Arow * 16 + Bcol, Aval * Bvals[j + min_row_len_B * 16]);
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < 16; ++i)
            if (rC[i])
                Cvals[vwarp_id * 16 + i] += rC[i];
    }
}

__device__ __forceinline__ void hyb_x_drw_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, nzC = Cbits[0] + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];
        if (Aval == 0) continue;

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = 0; j < 16; ++j) rC[j] += Aval * Bvals[k + j];
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        int j = 0;
        for (; j < nzC; ++j) if (Arow == Cbits[1 + j] >> 4) break;

        while (j < nzC && Arow == Cbits[1 + j] >> 4)
        {
            TileIndex Ccol = Cbits[1 + j] & 15;
            atomicAdd(Cvals + j, Aval * Bvals[k + Ccol]);
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < 16; ++i)
        {
            if (rC[i] != 0)
            {
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, vwarp_id << 4 | i);
                if (idx != -1) Cvals[idx] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void hyb_x_drw_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = Cbits[vwarp_id]; j < Cbits[vwarp_id + 1]; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 17, j);
            rC[Ccol] += Aval * Bvals[k + Ccol];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = Cbits[Arow]; j < Cbits[Arow + 1]; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 17, j);
            atomicAdd(Cvals + j, Aval * Bvals[k + Ccol]);
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = Cbits[vwarp_id]; i < Cbits[vwarp_id + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rC[col];
        }
    }
}

__device__ __forceinline__ void hyb_x_drw_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = vwarp_id * max_row_len_C; j < (vwarp_id + 1) * max_row_len_C; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + j);
            if (j > vwarp_id * max_row_len_C && Ccol == 0) break;
            rC[Ccol] += Aval * Bvals[k + Ccol];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = Arow * max_row_len_C; j < (Arow + 1) * max_row_len_C; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + j);
            atomicAdd(Cvals + j, Aval * Bvals[k + Ccol]);
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = vwarp_id * max_row_len_C; i < (vwarp_id + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void hyb_x_drw_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = vwarp_id * min_row_len_C; j < (vwarp_id + 1) * min_row_len_C; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, j);
            rC[Ccol] += Aval * Bvals[k + Ccol];
        }

        int j = 0;
        for (; j < coo_cnt_C; ++j)
            if (vwarp_id == Cbits[Celllen + j] >> 4)
                break;
        while (j < coo_cnt_C && vwarp_id == Cbits[Celllen + j] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + j] & 15;
            rC[Ccol] += Aval * Bvals[Ccol + k];
            ++j;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = Arow * min_row_len_C; j < (Arow + 1) * min_row_len_C; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, j);
            atomicAdd(Cvals + j, Aval * Bvals[k + Ccol]);
        }

        int j = 0;
        for (; j < coo_cnt_C; ++j)
            if (Arow == Cbits[Celllen + j] >> 4)
                break;
        while (j < coo_cnt_C && Arow == Cbits[Celllen + j] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + j] & 15;
            atomicAdd(Cvals + j + min_row_len_C * 16, Aval * Bvals[k + Ccol]);
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < min_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, vwarp_id * min_row_len_C + i);
            Cvals[vwarp_id * min_row_len_C + i] += rC[Ccol];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (vwarp_id == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && vwarp_id == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rC[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void hyb_x_drw_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = 0; j < 16; ++j) rC[j] += Aval * Bvals[k + j];
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        int kk = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (kk == -1) continue;
        kk -= 1;
        kk <<= 4;

        for (int j = 0; j < 16; ++j) atomicAdd(Cvals + kk + j, Aval * Bvals[k + j]);
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, vwarp_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            idx <<= 4;
            for (int i = 0; i < 16; ++i) Cvals[idx + i] += rC[i];
        }
    }
}

__device__ __forceinline__ void hyb_x_drw_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = 0; j < cntC; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits, 1 + j);
            rC[col] += Aval * Bvals[k + col];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = 0; j < cntC; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits, 1 + j);
            atomicAdd(Cvals + Arow + j * 16, Aval * Bvals[k + col]);
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex col = TileIndex_CSR_Get(Cbits, 1 + i);
        rC[col] += __shfl_down_sync(0xffffffff, rC[col], 1);
        if (vlane_id == 0) Cvals[i * 16 + vwarp_id] += rC[col];
    }
}

__device__ __forceinline__ void hyb_x_drw_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = 0; j < 16; ++j) rC[j] += Aval * Bvals[k + j];
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = 0; j < 16; ++j) atomicAdd(Cvals + Arow * 16 + j, Aval * Bvals[k + j]);
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
        if (vlane_id == 0) Cvals[vwarp_id * 16 + i] += rC[i];
    }
}

__device__ __forceinline__ void hyb_x_dcl_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, nzC = Cbits[0] + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[col] += Aval * Bvals[Acol + j * 16];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Bbits, 1 + j);
            int idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | col);
            if (idx != -1) atomicAdd(Cvals + idx, Aval * Bvals[Acol + j * 16]);
        }
    }

    for (int i = 0; i < cntB; ++i)
    {
        TileIndex col = TileIndex_CSR_Get(Bbits, 1 + i);
        rC[col] += __shfl_down_sync(0xffffffff, rC[col], 1);
        if (vlane_id == 0)
        {
            MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, vwarp_id << 4 | col);
            if (idx != -1) Cvals[idx] += rC[col];
        }
    }
}

__device__ __forceinline__ void hyb_x_dcl_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[col] += Aval * Bvals[Acol + j * 16];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        uint16_t row_map = get_row_map(Cbitmap, Arow);

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Bbits, 1 + j);
            int idx = check_bitmap_row_idx(row_map, col);
            if (idx != -1) atomicAdd(Cvals + Cbits[Arow] + idx, Aval * Bvals[Acol + j * 16]);
        }
    }

    uint16_t row_map = get_row_map(Cbitmap, vwarp_id);
    for (int i = 0; i < cntB; ++i)
    {
        TileIndex col = TileIndex_CSR_Get(Bbits, 1 + i);
        rC[col] += __shfl_down_sync(0xffffffff, rC[col], 1);
        int idx = check_bitmap_row_idx(row_map, col);
        if (vlane_id == 0 && idx != -1) Cvals[Cbits[vwarp_id] + idx] += rC[col];
    }
}

__device__ __forceinline__ void hyb_x_dcl_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[col] += Aval * Bvals[Acol + j * 16];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        uint16_t row_map = get_row_map(Cbitmap, Arow);

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Bbits, 1 + j);
            int idx = check_bitmap_row_idx(row_map, col);
            if (idx != -1) atomicAdd(Cvals + Arow * max_row_len_C + idx, Aval * Bvals[Acol + j * 16]);
        }
    }

    uint16_t row_map = get_row_map(Cbitmap, vwarp_id);
    for (int i = 0; i < cntB; ++i)
    {
        TileIndex col = TileIndex_CSR_Get(Bbits, 1 + i);
        rC[col] += __shfl_down_sync(0xffffffff, rC[col], 1);
        int idx = check_bitmap_row_idx(row_map, col);
        if (idx != -1 && vlane_id == 0) Cvals[vwarp_id * max_row_len_C + idx] += rC[col];
    }
}

__device__ __forceinline__ void hyb_x_dcl_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[col] += Aval * Bvals[Acol + j * 16];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Bbits, 1 + j);
            int idx_ell = binary_find_index_4bit(Cbits + 2, col, Arow * min_row_len_C, (Arow + 1) * min_row_len_C);
            int idx_coo = binary_find_index<TileIndex>(Cbits + Celllen, coo_cnt_C, Arow << 4 | col);
            if (idx_ell != -1) atomicAdd(Cvals + idx_ell, Aval * Bvals[Acol + j * 16]);
            if (idx_coo != -1) atomicAdd(Cvals + idx_coo + min_row_len_C * 16, Aval * Bvals[Acol + j * 16]);
        }
    }

    for (int i = 0; i < cntB; ++i)
    {
        TileIndex col = TileIndex_CSR_Get(Bbits, 1 + i);
        rC[col] += __shfl_down_sync(0xffffffff, rC[col], 1);
        if (vlane_id == 0)
        {
            int idx_ell = binary_find_index_4bit(Cbits + 2, col, vwarp_id * min_row_len_C, (vwarp_id + 1) * min_row_len_C);
            int idx_coo = binary_find_index<TileIndex>(Cbits + Celllen, coo_cnt_C, vwarp_id << 4 | col);
            if (idx_ell != -1) Cvals[idx_ell] += rC[col];
            if (idx_coo != -1) Cvals[idx_coo + min_row_len_C * 16] += rC[col];
        }
    }
}

__device__ __forceinline__ void hyb_x_dcl_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[col] += Aval * Bvals[Acol + j * 16];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        int k = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Bbits, 1 + j);
            atomicAdd(Cvals + k + col, Aval * Bvals[Acol + j * 16]);
        }
    }

    for (int i = 0; i < cntB; ++i)
    {
        TileIndex col = TileIndex_CSR_Get(Bbits, 1 + i);
        rC[col] += __shfl_down_sync(0xffffffff, rC[col], 1);
    }

    if (vlane_id == 0)
    {
        int k = binary_find_index_4bit(Cbits, vwarp_id, 1, cntC + 1);
        if (k != -1) {
            k -= 1;
            k <<= 4;
            for (int i = 0; i < 16; ++i) Cvals[k + i] += rC[i];
        }
    }
}

__device__ __forceinline__ void hyb_x_dcl_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    uint16_t col_map = 0;
    for (int i = 0; i < cntC; ++i)
    {
        TileIndex col = TileIndex_CSR_Get(Cbits, 1 + i);
        col_map |= 1 << col;
    }

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[col] += Aval * Bvals[Acol + j * 16];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Bbits, 1 + j);
            uint16_t col_mask = 1 << col;
            if (col_mask & col_map)
            {
                int idx = __popc(col_map & (col_mask - 1));
                atomicAdd(Cvals + Arow + 16 * idx, Aval * Bvals[Acol + j * 16]);
            }
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rC[Ccol] += __shfl_down_sync(0xffffffff, rC[Ccol], 1);
        if (vlane_id == 0) Cvals[vwarp_id + i * 16] += rC[Ccol];
    }
}

__device__ __forceinline__ void hyb_x_dcl_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        for (int j = 0; j < cntB; ++j)
        {
            rC[TileIndex_CSR_Get(Bbits, 1 + j)] += Aval * Bvals[j * 16 + Acol];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        for (int j = 0; j < cntB; ++j)
        {
            atomicAdd(Cvals + Arow * 16 + TileIndex_CSR_Get(Bbits, 1 + j), Aval * Bvals[j * 16 + Acol]);
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < 16; ++i)
            if (rC[i])
                Cvals[vwarp_id * 16 + i] += rC[i];
    }
}

__device__ __forceinline__ void hyb_x_dns_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int nzC = Cbits[0] + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        int j = 0;
        for (;j < nzC; ++j) if (vwarp_id == Cbits[1 + j] >> 4) break;
        while (j < nzC && vwarp_id == Cbits[1 + j] >> 4)
        {
            TileIndex col = Cbits[1 + j] & 15;
            rC[col] += Aval * Bvals[Acol * 16 + col];
            j++;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        int j = 0;
        for (;j < nzC; ++j) if (Arow == Cbits[1 + j] >> 4) break;
        while (j < nzC && Arow == Cbits[1 + j] >> 4)
        {
            TileIndex col = Cbits[1 + j] & 15;
            int idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | col);
            if (idx != -1) atomicAdd(Cvals + idx, Aval * Bvals[Acol * 16 + col]);
            j++;
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }
    
    if (vlane_id == 0) {
        int j = 0;
        for (;j < nzC; ++j) if (vwarp_id == Cbits[1 + j] >> 4) break;
        while (j < nzC && vwarp_id == Cbits[1 + j] >> 4)
        {
            TileIndex col = Cbits[1 + j] & 15;
            Cvals[j] += rC[col];
            j++;
        }
    }
}

__device__ __forceinline__ void hyb_x_dns_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        for (int j = Cbits[vwarp_id]; j < Cbits[vwarp_id + 1]; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits + 17, j);
            rC[col] += Aval * Bvals[Acol * 16 + col];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        for (int j = Cbits[Arow]; j < Cbits[Arow + 1]; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits + 17, j);
            atomicAdd(Cvals + j, Aval * Bvals[Acol * 16 + col]);
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0) 
    for (int i = Cbits[vwarp_id]; i < Cbits[vwarp_id + 1]; ++i)
    {
        TileIndex col = TileIndex_CSR_Get(Cbits + 17, i);
        Cvals[i] += rC[col];
    }
}

__device__ __forceinline__ void hyb_x_dns_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        for (int j = vwarp_id * max_row_len_C; j < (vwarp_id + 1) * max_row_len_C; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits, 1 + j);
            if (j > vwarp_id * max_row_len_C && col == 0) break;
            rC[col] += Aval * Bvals[Acol * 16 + col];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        for (int j = Arow * max_row_len_C; j < (Arow + 1) * max_row_len_C; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits, 1 + j);
            if (j > Arow * max_row_len_C && col == 0) break;
            atomicAdd(Cvals + j, Aval * Bvals[Acol * 16 + col]);
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0) 
    for (int i = vwarp_id * max_row_len_C; i < (vwarp_id + 1) * max_row_len_C; ++i)
    {
        TileIndex col = TileIndex_CSR_Get(Cbits, 1 + i);
        Cvals[i] += rC[col];
    }
}

__device__ __forceinline__ void hyb_x_dns_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        for (int j = vwarp_id * min_row_len_C; j < (vwarp_id + 1) * min_row_len_C; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits + 2, j);
            rC[col] += Aval * Bvals[Acol * 16 + col];
        }

        int j = 0;
        for (; j < coo_cnt_C; ++j) if (vwarp_id == Cbits[Celllen + j] >> 4) break;
        while (j < coo_cnt_C && vwarp_id == Cbits[Celllen + j] >> 4)
        {
            TileIndex col = Cbits[Celllen + j] & 15;
            rC[col] += Aval * Bvals[Acol * 16 + col];
            j++;
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        for (int j = Arow * min_row_len_C; j < (Arow + 1) * min_row_len_C; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits + 2, j);
            atomicAdd(Cvals + j, Aval * Bvals[Acol * 16 + col]);
        }

        int j = 0;
        for (; j < coo_cnt_C; ++j) if (Arow == Cbits[Celllen + j] >> 4) break;
        while (j < coo_cnt_C && Arow == Cbits[Celllen + j] >> 4)
        {
            TileIndex col = Cbits[Celllen + j] & 15;
            atomicAdd(Cvals + min_row_len_C * 16 + j, Aval * Bvals[Acol * 16 + col]);
            j++;
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0) {
        for (int i = vwarp_id * min_row_len_C; i < (vwarp_id + 1) * min_row_len_C; ++i)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits + 2, i);
            Cvals[i] += rC[col];
        }

        int j = 0;
        for (; j < coo_cnt_C; ++j) if (vwarp_id == Cbits[Celllen + j] >> 4) break;
        while (j < coo_cnt_C && vwarp_id == Cbits[Celllen + j] >> 4)
        {
            TileIndex col = Cbits[Celllen + j] & 15;
            if (vlane_id == 0) Cvals[min_row_len_C * 16 + j] += rC[col];
            j++;
        }
    }
}

__device__ __forceinline__ void hyb_x_dns_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    uint16_t row_map = 0;
    uint16_t row_mask = 1 << vwarp_id;
    for (int i = 0; i < cntC; ++i)
    {
        TileIndex row = TileIndex_CSR_Get(Cbits, 1 + i);
        row_map |= 1 << row;
    }
    bool flag = row_map & row_mask;
    int active_warp_mask = __ballot_sync(0xffffffff, flag);

    if (flag) {
        int idx = __popc(row_map & (row_mask - 1));
        for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
        {
            TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
            MatValue Aval = Avals[i];
            for (int j = 0; j < 16; ++j) rC[j] += Aval * Bvals[Acol * 16 + j];
        }

        for (int i = 0; i < 16; ++i)
        {
            rC[i] += __shfl_down_sync(active_warp_mask, rC[i], 1);
            if (vlane_id == 0) Cvals[idx * 16 + i] += rC[i];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];
        uint16_t Arow_mask = 1 << Arow;

        if (row_map & Arow_mask) {
            int idx = __popc(row_map & (Arow_mask - 1));
            for (int j = 0; j < 16; ++j) {
                atomicAdd(Cvals + idx * 16 + j, Aval * Bvals[Acol * 16 + j]);
            }
        }
    }
}

__device__ __forceinline__ void hyb_x_dns_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

        for (int j = 0; j < cntC; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits, 1 + j);
            rC[col] += Aval * Bvals[Acol * 16 + col];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

        for (int j = 0; j < cntC; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits, 1 + j);
            atomicAdd(Cvals + Arow + j * 16, Aval * Bvals[Acol * 16 + col]);
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex col = TileIndex_CSR_Get(Cbits, 1 + i);
        rC[col] += __shfl_down_sync(0xffffffff, rC[col], 1);
        if (vlane_id == 0) Cvals[vwarp_id + i * 16] += rC[col];
    }
}

__device__ __forceinline__ void hyb_x_dns_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int min_row_len_A = Abits[0], coo_cnt_A = Abits[1], Aelllen = (min_row_len_A * 16 + 1) / 2 + 2;

    for (int i = vwarp_id * min_row_len_A + vlane_id; i < (vwarp_id + 1) * min_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 2, i);
        MatValue Aval = Avals[i];

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rC[j] += Aval * Bvals[Acol * 16 + j];
        }
    }

    for (int i = lane_id; i < coo_cnt_A; i += 32)
    {
        TileIndex idx = Abits[Aelllen + i], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i + min_row_len_A * 16];

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            atomicAdd(Cvals + Arow * 16 + j, Aval * Bvals[Acol * 16 + j]);
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < 16; ++i)
            if (rC[i])
                Cvals[vwarp_id * 16 + i] += rC[i];
    }
}