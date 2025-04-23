#pragma once
#include "./common.cuh"
#include <tmatrix/DataStructure/TileMatrix.h>

__device__ __forceinline__ void csr_x_coo_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int Arow = vwarp_id, nzB = Bbits[0] + 1;
    int nzC = Cbits[0] + 1;
    MatValue rC[16] = {0};
    for (int i = Abits[vwarp_id] + vlane_id; i < Abits[vwarp_id + 1]; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + i);
        MatValue Aval = Avals[i];
        
        int j = 0;
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;

        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rC[Bcol] += Aval * Bvals[j];
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
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | i);
                if (idx != -1) Cvals[idx] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void csr_x_coo_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int Arow = vwarp_id, nzB = Bbits[0] + 1;
    MatValue rC[16] = {0};
    for (int i = Abits[Arow] + vlane_id; i < Abits[Arow + 1]; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + i);
        MatValue Aval = Avals[i];
        
        int j = 0;
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;
        
        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rC[Bcol] += Aval * Bvals[j];
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

__device__ __forceinline__ void csr_x_coo_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int Arow = vwarp_id, nzB = Bbits[0] + 1;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = Abits[Arow] + vlane_id; i < Abits[Arow + 1]; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + i);
        MatValue Aval = Avals[i];

        int j = 0;
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;

        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rC[Bcol] += Aval * Bvals[j];
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

__device__ __forceinline__ void csr_x_coo_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzB = Bbits[0] + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int i = Abits[vwarp_id] + vlane_id; i < Abits[vwarp_id + 1]; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + i);
        MatValue Aval = Avals[i];

        int j = 0;
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;

        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rC[Bcol] += Aval * Bvals[j];
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

__device__ __forceinline__ void csr_x_coo_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzB = Bbits[0] + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = Abits[vwarp_id] + vlane_id; i < Abits[vwarp_id + 1]; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + i);
        MatValue Aval = Avals[i];

        int j = 0;
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;

        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rC[Bcol] += Aval * Bvals[j];
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

__device__ __forceinline__ void csr_x_coo_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int Arow = vwarp_id, nzB = Bbits[0] + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = Abits[vwarp_id] + vlane_id; i < Abits[vwarp_id + 1]; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + i);
        MatValue Aval = Avals[i];

        int j = 0;
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;

        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rC[Bcol] += Aval * Bvals[j];
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
        for (int i = 0; i < cntC; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[Arow + 16 * i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void csr_x_coo_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzB = Bbits[0] + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];
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

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        #pragma unroll
        for (int i = 0; i < 16; ++i) Cvals[vwarp_id * 16 + i] += rC[i];
    }
}

__device__ __forceinline__ void csr_x_csr_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int nzC = Cbits[0] + 1;

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bi = Bbits[Acol]; bi < Bbits[Acol + 1]; ++bi)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + bi);
            rC[Bcol] += Aval * Bvals[bi];
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

__device__ __forceinline__ void csr_x_csr_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bi = Bbits[Acol]; bi < Bbits[Acol + 1]; ++bi)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + bi);
            rC[Bcol] += Aval * Bvals[bi];
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

__device__ __forceinline__ void csr_x_csr_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bi = Bbits[Acol]; bi < Bbits[Acol + 1]; ++bi)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + bi);
            rC[Bcol] += Aval * Bvals[bi];
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

__device__ __forceinline__ void csr_x_csr_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bi = Bbits[Acol]; bi < Bbits[Acol + 1]; ++bi)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + bi);
            rC[Bcol] += Aval * Bvals[bi];
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

__device__ __forceinline__ void csr_x_csr_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bi = Bbits[Acol]; bi < Bbits[Acol + 1]; ++bi)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + bi);
            rC[Bcol] += Aval * Bvals[bi];
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

__device__ __forceinline__ void csr_x_csr_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bi = Bbits[Acol]; bi < Bbits[Acol + 1]; ++bi)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + bi);
            rC[Bcol] += Aval * Bvals[bi];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < cntC; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[vwarp_id + 16 * i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void csr_x_csr_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bi = Bbits[Acol]; bi < Bbits[Acol + 1]; ++bi)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + bi);
            rC[Bcol] += Aval * Bvals[bi];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        #pragma unroll
        for (int i = 0; i < 16; ++i) Cvals[vwarp_id * 16 + i] += rC[i];
    }
}

__device__ __forceinline__ void csr_x_ell_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1, Arow = vwarp_id, nzC = Cbits[0] + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = max_row_len_B * Acol; bj < max_row_len_B * (Acol + 1); ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + bj);
            rC[Bcol] += Aval * Bvals[bj];
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
                // MatIndex idx = check_bitmap_idx(Cbitmap, vwarp_id << 4 | i);
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | i);
                if (idx != -1) Cvals[idx] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void csr_x_ell_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = max_row_len_B * Acol; bj < max_row_len_B * (Acol + 1); ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + bj);
            rC[Bcol] += Aval * Bvals[bj];
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

__device__ __forceinline__ void csr_x_ell_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = max_row_len_B * Acol; bj < max_row_len_B * (Acol + 1); ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + bj);
            rC[Bcol] += Aval * Bvals[bj];
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

__device__ __forceinline__ void csr_x_ell_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = max_row_len_B * Acol; bj < max_row_len_B * (Acol + 1); ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + bj);
            rC[Bcol] += Aval * Bvals[bj];
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

__device__ __forceinline__ void csr_x_ell_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = max_row_len_B * Acol; bj < max_row_len_B * (Acol + 1); ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + bj);
            rC[Bcol] += Aval * Bvals[bj];
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

__device__ __forceinline__ void csr_x_ell_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = max_row_len_B * Acol; bj < max_row_len_B * (Acol + 1); ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + bj);
            rC[Bcol] += Aval * Bvals[bj];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < cntC; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[vwarp_id + 16 * i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void csr_x_ell_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = max_row_len_B * Acol; bj < max_row_len_B * (Acol + 1); ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + bj);
            rC[Bcol] += Aval * Bvals[bj];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        #pragma unroll
        for (int i = 0; i < 16; ++i) Cvals[vwarp_id * 16 + i] += rC[i];
    }
}

__device__ __forceinline__ void csr_x_hyb_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2, Arow = vwarp_id, nzC = Cbits[0] + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = min_row_len_B * Acol; bj < min_row_len_B * (Acol + 1); ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + bj);
            rC[Bcol] += Aval * Bvals[bj];
        }

        int bj = 0;
        for (; bj < coo_cnt_B; ++bj)
            if (Acol == Bbits[bj + Belllen] >> 4)
                break;
        while (Acol == Bbits[bj + Belllen] >> 4 && bj < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[bj + Belllen] & 15;
            rC[Bcol] += Aval * Bvals[bj + min_row_len_B * 16];
            ++bj;
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
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | i);
                if (idx != -1) Cvals[idx] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void csr_x_hyb_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = min_row_len_B * Acol; bj < min_row_len_B * (Acol + 1); ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + bj);
            rC[Bcol] += Aval * Bvals[bj];
        }

        int bj = 0;
        for (; bj < coo_cnt_B; ++bj)
            if (Acol == Bbits[bj + Belllen] >> 4)
                break;
        while (Acol == Bbits[bj + Belllen] >> 4 && bj < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[bj + Belllen] & 15;
            rC[Bcol] += Aval * Bvals[bj + min_row_len_B * 16];
            ++bj;
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

__device__ __forceinline__ void csr_x_hyb_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = min_row_len_B * Acol; bj < min_row_len_B * (Acol + 1); ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + bj);
            rC[Bcol] += Aval * Bvals[bj];
        }

        int bj = 0;
        for (; bj < coo_cnt_B; ++bj)
            if (Acol == Bbits[bj + Belllen] >> 4)
                break;
        while (Acol == Bbits[bj + Belllen] >> 4 && bj < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[bj + Belllen] & 15;
            rC[Bcol] += Aval * Bvals[bj + min_row_len_B * 16];
            ++bj;
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

__device__ __forceinline__ void csr_x_hyb_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = min_row_len_B * Acol; bj < min_row_len_B * (Acol + 1); ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + bj);
            rC[Bcol] += Aval * Bvals[bj];
        }

        int bj = 0;
        for (; bj < coo_cnt_B; ++bj)
            if (Acol == Bbits[bj + Belllen] >> 4)
                break;
        while (Acol == Bbits[bj + Belllen] >> 4 && bj < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[bj + Belllen] & 15;
            rC[Bcol] += Aval * Bvals[bj + min_row_len_B * 16];
            ++bj;
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

__device__ __forceinline__ void csr_x_hyb_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = min_row_len_B * Acol; bj < min_row_len_B * (Acol + 1); ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + bj);
            rC[Bcol] += Aval * Bvals[bj];
        }

        int bj = 0;
        for (; bj < coo_cnt_B; ++bj)
            if (Acol == Bbits[bj + Belllen] >> 4)
                break;
        while (Acol == Bbits[bj + Belllen] >> 4 && bj < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[bj + Belllen] & 15;
            rC[Bcol] += Aval * Bvals[bj + min_row_len_B * 16];
            ++bj;
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

__device__ __forceinline__ void csr_x_hyb_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = min_row_len_B * Acol; bj < min_row_len_B * (Acol + 1); ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + bj);
            rC[Bcol] += Aval * Bvals[bj];
        }

        int bj = 0;
        for (; bj < coo_cnt_B; ++bj)
            if (Acol == Bbits[bj + Belllen] >> 4)
                break;
        while (Acol == Bbits[bj + Belllen] >> 4 && bj < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[bj + Belllen] & 15;
            rC[Bcol] += Aval * Bvals[bj + min_row_len_B * 16];
            ++bj;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < cntC; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[vwarp_id + 16 * i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void csr_x_hyb_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = min_row_len_B * Acol; bj < min_row_len_B * (Acol + 1); ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + bj);
            rC[Bcol] += Aval * Bvals[bj];
        }

        int bj = 0;
        for (; bj < coo_cnt_B; ++bj)
            if (Acol == Bbits[bj + Belllen] >> 4)
                break;
        while (Acol == Bbits[bj + Belllen] >> 4 && bj < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[bj + Belllen] & 15;
            rC[Bcol] += Aval * Bvals[bj + min_row_len_B * 16];
            ++bj;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        #pragma unroll
        for (int i = 0; i < 16; ++i) Cvals[vwarp_id * 16 + i] += rC[i];
    }
}

__device__ __forceinline__ void csr_x_drw_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, nzC = Cbits[0] + 1, Arow = vwarp_id;
    MatValue rC[16] = {0};

    for (int ai = Abits[Arow] + vlane_id; ai < Abits[Arow + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits + 17, ai);
        MatValue Aval = Avals[ai];
        int bj = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);

        if (bj != -1)
        {
            bj -= 1;
            for (int k = 0; k < 16; ++k) {
                rC[k] += Aval * Bvals[bj * 16 + k];
            }
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
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | i);
                if (idx != -1) Cvals[idx] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void csr_x_drw_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];
        int bj = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);

        if (bj != -1)
        {
            bj -= 1;
            for (int k = 0; k < 16; ++k)
                rC[k] += Aval * Bvals[bj * 16 + k];
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

__device__ __forceinline__ void csr_x_drw_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];
        int bj = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);

        if (bj != -1)
        {
            bj -= 1;
            for (int k = 0; k < 16; ++k)
                rC[k] += Aval * Bvals[bj * 16 + k];
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

__device__ __forceinline__ void csr_x_drw_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];
        int bj = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);

        if (bj != -1)
        {
            bj -= 1;
            for (int k = 0; k < 16; ++k)
                rC[k] += Aval * Bvals[bj * 16 + k];
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

__device__ __forceinline__ void csr_x_drw_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];
        int bj = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);

        if (bj != -1)
        {
            bj -= 1;
            for (int k = 0; k < 16; ++k)
                rC[k] += Aval * Bvals[bj * 16 + k];
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

__device__ __forceinline__ void csr_x_drw_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];
        int bj = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);

        if (bj != -1)
        {
            bj -= 1;
            for (int k = 0; k < 16; ++k)
                rC[k] += Aval * Bvals[bj * 16 + k];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < cntC; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[vwarp_id + 16 * i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void csr_x_drw_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        int bj = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (bj != -1)
        {
            bj -= 1;
            for (int k = 0; k < 16; ++k) rC[k] += Aval * Bvals[bj * 16 + k];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        #pragma unroll
        for (int i = 0; i < 16; ++i) Cvals[vwarp_id * 16 + i] += rC[i];
    }
}

__device__ __forceinline__ void csr_x_dcl_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, nzC = Cbits[0] + 1, Arow = vwarp_id;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = 0; bj < cntB; ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + bj);
            rC[Bcol] += Aval * Bvals[bj * 16 + Acol];
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
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | i);
                if (idx != -1) Cvals[idx] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void csr_x_dcl_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = 0; bj < cntB; ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + bj);
            rC[Bcol] += Aval * Bvals[bj * 16 + Acol];
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

__device__ __forceinline__ void csr_x_dcl_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = 0; bj < cntB; ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + bj);
            rC[Bcol] += Aval * Bvals[bj * 16 + Acol];
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

__device__ __forceinline__ void csr_x_dcl_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = 0; bj < cntB; ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + bj);
            rC[Bcol] += Aval * Bvals[bj * 16 + Acol];
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

__device__ __forceinline__ void csr_x_dcl_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = 0; bj < cntB; ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + bj);
            rC[Bcol] += Aval * Bvals[bj * 16 + Acol];
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

__device__ __forceinline__ void csr_x_dcl_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = 0; bj < cntB; ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + bj);
            rC[Bcol] += Aval * Bvals[bj * 16 + Acol];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < cntC; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[vwarp_id + 16 * i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void csr_x_dcl_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = 0; bj < cntB; ++bj)
        {
            rC[TileIndex_CSR_Get(Bbits, 1 + bj)] += Aval * Bvals[bj * 16 + Acol];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        #pragma unroll
        for (int i = 0; i < 16; ++i) Cvals[vwarp_id * 16 + i] += rC[i];
    }
}

__device__ __forceinline__ void csr_x_dns_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int nzC = Cbits[0] + 1, Arow = vwarp_id;

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = 0; bj < 16; ++bj)
        {
            rC[bj] += Aval * Bvals[Acol * 16 + bj];
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
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | i);
                if (idx != -1) Cvals[idx] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void csr_x_dns_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = 0; bj < 16; ++bj)
        {
            rC[bj] += Aval * Bvals[Acol * 16 + bj];
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

__device__ __forceinline__ void csr_x_dns_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = 0; bj < 16; ++bj)
        {
            rC[bj] += Aval * Bvals[Acol * 16 + bj];
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

__device__ __forceinline__ void csr_x_dns_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = 0; bj < 16; ++bj)
        {
            rC[bj] += Aval * Bvals[Acol * 16 + bj];
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

__device__ __forceinline__ void csr_x_dns_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = 0; bj < 16; ++bj)
        {
            rC[bj] += Aval * Bvals[Acol * 16 + bj];
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

__device__ __forceinline__ void csr_x_dns_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

        for (int bj = 0; bj < 16; ++bj)
        {
            rC[bj] += Aval * Bvals[Acol * 16 + bj];
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < cntC; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[vwarp_id + 16 * i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void csr_x_dns_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};

    for (int ai = Abits[vwarp_id] + vlane_id; ai < Abits[vwarp_id + 1]; ai += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 34 + ai);
        MatValue Aval = Avals[ai];

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rC[j] += Aval * Bvals[Acol * 16 + j];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        #pragma unroll
        for (int i = 0; i < 16; ++i) Cvals[vwarp_id * 16 + i] += rC[i];
    }
}