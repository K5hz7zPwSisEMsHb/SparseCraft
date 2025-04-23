#pragma once
#include <tmatrix/Calculation/kernel_matrix/common.cuh>


__device__ __forceinline__ void ell_x_coo_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int Arow = vwarp_id;
    int nzB = Bbits[0] + 1;
    int nzC = Cbits[0] + 1;
    MatValue rC[16] = {0};
    int i_start = vwarp_id * max_row_len_A, i_end = (vwarp_id + 1) * max_row_len_A;

    for (int i = i_start + vlane_id; i < i_end; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (i > i_start && Acol == 0) continue;

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
                // MatIndex idx = check_bitmap_idx(Cbitmap, vwarp_id << 4 | i);
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | i);
                Cvals[idx] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void ell_x_coo_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int nzB = Bbits[0] + 1;
    MatValue rC[16] = {0};
    int i_start = vwarp_id * max_row_len_A, i_end = (vwarp_id + 1) * max_row_len_A;

    for (int i = i_start + vlane_id; i < i_end; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (i > i_start && Acol == 0) continue;

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
        for (int i = Cbits[vwarp_id]; i < Cbits[vwarp_id + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rC[col];
        }
    }
}

__device__ __forceinline__ void ell_x_coo_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    int nzB = Bbits[0] + 1;
    MatValue rC[16] = {0};
    int i_start = vwarp_id * max_row_len_A, i_end = (vwarp_id + 1) * max_row_len_A;

    for (int i = i_start + vlane_id; i < i_end; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (i > i_start && Acol == 0) continue;

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

__device__ __forceinline__ void ell_x_coo_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    int nzB = Bbits[0] + 1;
    MatValue rC[16] = {0};
    int i_start = vwarp_id * max_row_len_A, i_end = (vwarp_id + 1) * max_row_len_A;

    for (int i = i_start + vlane_id; i < i_end; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (i > i_start && Acol == 0) continue;

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

__device__ __forceinline__ void ell_x_coo_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int nzB = Bbits[0] + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};
    int i_start = vwarp_id * max_row_len_A, i_end = (vwarp_id + 1) * max_row_len_A;

    // if (debug_flag && lane_id == 0) {
    //     printf("cntC: %d\n", cntC);
    //     for (int i = 0; i < cntC; ++i) printf("Cbits[%d]: %d\n", i, TileIndex_CSR_Get(Cbits, 1 + i));
    // }

    for (int i = i_start + vlane_id; i < i_end; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (i > i_start && Acol == 0) break;

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
        // if (debug_flag && vwarp_id == 15) printf("idx: %d\n", idx);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void ell_x_coo_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int nzB = Bbits[0] + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};
    int i_start = vwarp_id * max_row_len_A, i_end = (vwarp_id + 1) * max_row_len_A;

    for (int i = i_start + vlane_id; i < i_end; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (i > i_start && Acol == 0) continue;

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
            Cvals[vwarp_id + 16 * i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void ell_x_coo_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, nzB = Bbits[0] + 1;

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        int bj = 0;
        for (; bj < nzB; ++bj)
            if (Acol == Bbits[1 + bj] >> 4)
                break;
        while (Acol == Bbits[1 + bj] >> 4 && bj < nzB)
        {
            TileIndex Bcol = Bbits[1 + bj] & 15;
            rC[Bcol] += Aval * Bvals[bj];
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
        for (int i = 0; i < 16; ++i)
        {
            Cvals[vwarp_id * 16 + i] += rC[i];
        }
    }
}

__device__ __forceinline__ void ell_x_csr_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, Arow = vwarp_id, nzC = Cbits[0] + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
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
                Cvals[idx] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void ell_x_csr_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
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

__device__ __forceinline__ void ell_x_csr_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
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

__device__ __forceinline__ void ell_x_csr_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
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

__device__ __forceinline__ void ell_x_csr_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
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

__device__ __forceinline__ void ell_x_csr_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
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

__device__ __forceinline__ void ell_x_csr_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rC[Bcol] += Aval * Bvals[j];
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
        for (int i = 0; i < 16; ++i)
        {
            Cvals[vwarp_id * 16 + i] += rC[i];
        }
    }
}

__device__ __forceinline__ void ell_x_ell_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int Arow = vwarp_id, nzC = Cbits[0] + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (Bvals[j]) rC[Bcol] += Aval * Bvals[j];
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
                Cvals[idx] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void ell_x_ell_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (Bvals[j]) rC[Bcol] += Aval * Bvals[j];
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

__device__ __forceinline__ void ell_x_ell_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (Bvals[j]) rC[Bcol] += Aval * Bvals[j];
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

__device__ __forceinline__ void ell_x_ell_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (Bvals[j]) rC[Bcol] += Aval * Bvals[j];
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

__device__ __forceinline__ void ell_x_ell_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (Bvals[j]) rC[Bcol] += Aval * Bvals[j];
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

__device__ __forceinline__ void ell_x_ell_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (Bvals[j]) rC[Bcol] += Aval * Bvals[j];
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

__device__ __forceinline__ void ell_x_ell_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (Bvals[j]) rC[Bcol] += Aval * Bvals[j];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        // memcpy(Cvals + vwarp_id * 16, rC, 16 * sizeof(MatValue));
        for (int i = 0; i < 16; ++i) Cvals[vwarp_id * 16 + i] += rC[i];
    }
}

__device__ __forceinline__ void ell_x_hyb_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int Arow = vwarp_id, nzC = Cbits[0] + 1;
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0) continue;
        
        for (int bj = Acol * min_row_len_B; bj < (Acol + 1) * min_row_len_B; ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, bj);
            rC[Bcol] += Aval * Bvals[bj];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j) if (Acol == Bbits[Belllen + j] >> 4) break;
        while (j < coo_cnt_B && Acol == Bbits[Belllen + j] >> 4)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rC[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
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

__device__ __forceinline__ void ell_x_hyb_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0) continue;
        
        for (int bj = Acol * min_row_len_B; bj < (Acol + 1) * min_row_len_B; ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, bj);
            rC[Bcol] += Aval * Bvals[bj];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j) if (Acol == Bbits[Belllen + j] >> 4) break;
        while (j < coo_cnt_B && Acol == Bbits[Belllen + j] >> 4)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rC[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
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

__device__ __forceinline__ void ell_x_hyb_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0) continue;
        
        for (int bj = Acol * min_row_len_B; bj < (Acol + 1) * min_row_len_B; ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, bj);
            rC[Bcol] += Aval * Bvals[bj];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j) if (Acol == Bbits[Belllen + j] >> 4) break;
        while (j < coo_cnt_B && Acol == Bbits[Belllen + j] >> 4)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rC[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
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

__device__ __forceinline__ void ell_x_hyb_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0) continue;
        
        for (int bj = Acol * min_row_len_B; bj < (Acol + 1) * min_row_len_B; ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, bj);
            rC[Bcol] += Aval * Bvals[bj];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j) if (Acol == Bbits[Belllen + j] >> 4) break;
        while (j < coo_cnt_B && Acol == Bbits[Belllen + j] >> 4)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rC[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
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

__device__ __forceinline__ void ell_x_hyb_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0) continue;
        
        for (int bj = Acol * min_row_len_B; bj < (Acol + 1) * min_row_len_B; ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, bj);
            rC[Bcol] += Aval * Bvals[bj];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j) if (Acol == Bbits[Belllen + j] >> 4) break;
        while (j < coo_cnt_B && Acol == Bbits[Belllen + j] >> 4)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rC[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
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
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void ell_x_hyb_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0) continue;
        
        for (int bj = Acol * min_row_len_B; bj < (Acol + 1) * min_row_len_B; ++bj)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, bj);
            rC[Bcol] += Aval * Bvals[bj];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j) if (Acol == Bbits[Belllen + j] >> 4) break;
        while (j < coo_cnt_B && Acol == Bbits[Belllen + j] >> 4)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rC[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
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
            Cvals[vwarp_id + 16 * i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void ell_x_hyb_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            if (Bvals[j])
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

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 1);
    }

    if (vlane_id == 0)
    {
        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            Cvals[vwarp_id * 16 + i] += rC[i];
        }
    }
}

__device__ __forceinline__ void ell_x_drw_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int Arow = vwarp_id, nzC = Cbits[0] + 1;
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        int bj = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (bj != -1)
        {
            bj -= 1;
            for (int j = 0; j < 16; ++j)
            {
                rC[j] += Aval * Bvals[bj * 16 + j];
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
        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            if (rC[i] != 0)
            {
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | i);
                Cvals[idx] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void ell_x_drw_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        int bj = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (bj != -1)
        {
            bj -= 1;
            for (int j = 0; j < 16; ++j)
            {
                rC[j] += Aval * Bvals[bj * 16 + j];
            }
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

__device__ __forceinline__ void ell_x_drw_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        int bj = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (bj != -1)
        {
            bj -= 1;
            for (int j = 0; j < 16; ++j)
            {
                rC[j] += Aval * Bvals[bj * 16 + j];
            }
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

__device__ __forceinline__ void ell_x_drw_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        int bj = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (bj != -1)
        {
            bj -= 1;
            for (int j = 0; j < 16; ++j)
            {
                rC[j] += Aval * Bvals[bj * 16 + j];
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

__device__ __forceinline__ void ell_x_drw_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        int bj = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (bj != -1)
        {
            bj -= 1;
            for (int j = 0; j < 16; ++j)
            {
                rC[j] += Aval * Bvals[bj * 16 + j];
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
        int idx = binary_find_index_4bit(Cbits, vwarp_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void ell_x_drw_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        int bj = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (bj != -1)
        {
            bj -= 1;
            for (int j = 0; j < 16; ++j)
            {
                rC[j] += Aval * Bvals[bj * 16 + j];
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
        for (int i = 0; i < cntC; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[vwarp_id + 16 * i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void ell_x_drw_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Brow = TileIndex_CSR_Get(Bbits, 1 + j);
            if (Brow == Acol)
            {
#pragma unroll
                for (int k = 0; k < 16; ++k)
                    rC[k] += Aval * Bvals[j * 16 + k];
                break;
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
        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            Cvals[vwarp_id * 16 + i] += rC[i];
        }
    }
}

__device__ __forceinline__ void ell_x_dcl_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int Arow = vwarp_id, nzC = Cbits[0] + 1;
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j * 16 + Acol];
        }
    }

#pragma unroll
    for (int i = 0; i < cntB; ++i)
    {
        TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + i);
        rC[Bcol] += __shfl_down_sync(0xffffffff, rC[Bcol], 1);
    }

    if (vlane_id == 0)
    {
        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            if (rC[i] != 0)
            {
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | i);
                Cvals[idx] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void ell_x_dcl_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j * 16 + Acol];
        }
    }

#pragma unroll
    for (int i = 0; i < cntB; ++i)
    {
        TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + i);
        rC[Bcol] += __shfl_down_sync(0xffffffff, rC[Bcol], 1);
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

__device__ __forceinline__ void ell_x_dcl_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j * 16 + Acol];
        }
    }

#pragma unroll
    for (int i = 0; i < cntB; ++i)
    {
        TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + i);
        rC[Bcol] += __shfl_down_sync(0xffffffff, rC[Bcol], 1);
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

__device__ __forceinline__ void ell_x_dcl_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j * 16 + Acol];
        }
    }

    #pragma unroll
    for (int i = 0; i < cntB; ++i)
    {
        TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + i);
        rC[Bcol] += __shfl_down_sync(0xffffffff, rC[Bcol], 1);
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

__device__ __forceinline__ void ell_x_dcl_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j * 16 + Acol];
        }
    }

#pragma unroll
    for (int i = 0; i < cntB; ++i)
    {
        TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + i);
        rC[Bcol] += __shfl_down_sync(0xffffffff, rC[Bcol], 1);
    }

    if (vlane_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, vwarp_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void ell_x_dcl_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j * 16 + Acol];
        }
    }

#pragma unroll
    for (int i = 0; i < cntB; ++i)
    {
        TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + i);
        rC[Bcol] += __shfl_down_sync(0xffffffff, rC[Bcol], 1);
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

__device__ __forceinline__ void ell_x_dcl_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;
        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j * 16 + Acol];
        }
    }

#pragma unroll
    for (int i = 0; i < cntB; ++i)
    {
        TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + i);
        rC[Bcol] += __shfl_down_sync(0xffffffff, rC[Bcol], 1);
    }

    if (vlane_id == 0)
    {
        #pragma unroll
        for (int i = 0; i < cntB; ++i)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + i);
            Cvals[vwarp_id * 16 + Bcol] += rC[Bcol];
        }
    }
}

__device__ __forceinline__ void ell_x_dns_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int Arow = vwarp_id, nzC = Cbits[0] + 1;
    MatValue rC[16] = {0};
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;

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
        for (int i = 0; i < 16; ++i)
        {
            if (rC[i] != 0)
            {
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | i);
                Cvals[idx] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void ell_x_dns_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rC[j] += Aval * Bvals[Acol * 16 + j];
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

__device__ __forceinline__ void ell_x_dns_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rC[j] += Aval * Bvals[Acol * 16 + j];
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

__device__ __forceinline__ void ell_x_dns_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;

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

__device__ __forceinline__ void ell_x_dns_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;

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
        int idx = binary_find_index_4bit(Cbits, vwarp_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void ell_x_dns_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;

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
        for (int i = 0; i < cntC; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[vwarp_id + 16 * i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void ell_x_dns_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rC[16] = {0};
    int max_row_len_A = TileIndex_CSR_Get(Abits, 0) + 1;

    for (int i = vwarp_id * max_row_len_A + vlane_id; i < (vwarp_id + 1) * max_row_len_A; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i];
        if (Aval == 0)
            continue;

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
        // memcpy(Cvals + vwarp_id * 16, rC, 16 * sizeof(MatValue));
        for (int i = 0; i < 16; ++i)
        {
            Cvals[vwarp_id * 16 + i] += rC[i];
        }
    }
}