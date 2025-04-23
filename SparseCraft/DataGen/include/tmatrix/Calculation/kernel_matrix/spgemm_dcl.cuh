#pragma once
#include <tmatrix/Calculation/kernel_matrix/common.cuh>

__device__ __forceinline__ void dcl_x_coo_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int nzB = Bbits[0] + 1, nzC = Cbits[0] + 1;
    MatValue rC[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];
        if (Aval == 0) continue;

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

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 16);
    }

    if (hwarp_id == 0)
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

__device__ __forceinline__ void dcl_x_coo_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int nzB = Bbits[0] + 1;
    MatValue rC[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

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

    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = Cbits[hlane_id]; i < Cbits[hlane_id + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rC[col];
        }
    }
}

__device__ __forceinline__ void dcl_x_coo_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int nzB = Bbits[0] + 1;
    MatValue rC[16] = {0};

    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

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

    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = hlane_id * max_row_len_C; i < (hlane_id + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            if (i > hlane_id * max_row_len_C && Ccol == 0) break;
            Cvals[i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void dcl_x_coo_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int nzB = Bbits[0] + 1;
    MatValue rC[16] = {0};

    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

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

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = hlane_id * min_row_len_C; i < (hlane_id + 1) * min_row_len_C; ++i)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits + 2, i);
            Cvals[i] += rC[col];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (hlane_id == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && hlane_id == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rC[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void dcl_x_coo_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int nzB = Bbits[0] + 1;
    MatValue rC[16] = {0};
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

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

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 16);
    }

    if (hwarp_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, hlane_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void dcl_x_coo_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int nzB = Bbits[0] + 1;
    MatValue rC[16] = {0};

    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

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

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rC[Ccol] += __shfl_down_sync(0xffffffff, rC[Ccol], 16);
        if (hwarp_id == 0) Cvals[Arow + i * 16] += rC[Ccol];
    }
}

__device__ __forceinline__ void dcl_x_coo_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int nzB = Bbits[0] + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        int j = 0;
        for (; j < nzB; ++j)
            if (Acol == Bbits[1 + j] >> 4)
                break;
        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rc[Bcol] += Aval * Bvals[j];
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
        if (hwarp_id == 0)
            Cvals[Arow * 16 + i] += rc[i];
    }
}

__device__ __forceinline__ void dcl_x_csr_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, nzC = Cbits[0] + 1;
    MatValue rC[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = 0; i < 16; ++i)
        {
            if (rC[i] != 0)
            {
                // MatIndex idx = check_bitmap_idx(Cbitmap, Arow << 4 | i);
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | i);
                if (idx != -1) Cvals[idx] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void dcl_x_csr_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = Cbits[hlane_id]; i < Cbits[hlane_id + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rC[col];
        }
    }
}

__device__ __forceinline__ void dcl_x_csr_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    MatValue rC[16] = {0};

    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = hlane_id * max_row_len_C; i < (hlane_id + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void dcl_x_csr_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    MatValue rC[16] = {0};

    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = hlane_id * min_row_len_C; i < (hlane_id + 1) * min_row_len_C; ++i)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits + 2, i);
            Cvals[i] += rC[col];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (hlane_id == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && hlane_id == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rC[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void dcl_x_csr_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    MatValue rC[16] = {0};
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 16);
    }

    if (hwarp_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, hlane_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void dcl_x_csr_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    MatValue rC[16] = {0};
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rC[Ccol] += __shfl_down_sync(0xffffffff, rC[Ccol], 16);
        if (hwarp_id == 0) Cvals[Arow + i * 16] += rC[Ccol];
    }
}

__device__ __forceinline__ void dcl_x_csr_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rc[Bcol] += Aval * Bvals[j];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
        if (hwarp_id == 0)
            Cvals[Arow * 16 + i] += rc[i];
    }
}

__device__ __forceinline__ void dcl_x_ell_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id, max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, nzC = Cbits[0] + 1;
    MatValue rC[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 16);
    }

    if (hwarp_id == 0)
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

__device__ __forceinline__ void dcl_x_ell_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = Cbits[hlane_id]; i < Cbits[hlane_id + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rC[col];
        }
    }
}

__device__ __forceinline__ void dcl_x_ell_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    MatValue rC[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = hlane_id * max_row_len_C; i < (hlane_id + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[i] += rC[Ccol];
        }
    }
}

__device__ __forceinline__ void dcl_x_ell_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    MatValue rC[16] = {0};

    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = hlane_id * min_row_len_C; i < (hlane_id + 1) * min_row_len_C; ++i)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits + 2, i);
            Cvals[i] += rC[col];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (hlane_id == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && hlane_id == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rC[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void dcl_x_ell_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    MatValue rC[16] = {0};
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rC[i] += __shfl_down_sync(0xffffffff, rC[i], 16);
    }

    if (hwarp_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, hlane_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rC[i];
            }
        }
    }
}

__device__ __forceinline__ void dcl_x_ell_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id, max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    MatValue rC[16] = {0};
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rC[Bcol] += Aval * Bvals[j];
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rC[Ccol] += __shfl_down_sync(0xffffffff, rC[Ccol], 16);
        if (hwarp_id == 0) Cvals[Arow + i * 16] += rC[Ccol];
    }
}

__device__ __forceinline__ void dcl_x_ell_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id, max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rc[Bcol] += Aval * Bvals[j];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
        if (hwarp_id == 0)
            Cvals[Arow * 16 + i] += rc[i];
    }
}

__device__ __forceinline__ void dcl_x_hyb_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, nzC = Cbits[0] + 1;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            rc[Bcol] += Aval * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rc[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
        if (hwarp_id == 0)
        {
            MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | i);
            if (idx != -1) Cvals[idx] += rc[i];
        }
    }
}

__device__ __forceinline__ void dcl_x_hyb_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            rc[Bcol] += Aval * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rc[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = Cbits[hlane_id]; i < Cbits[hlane_id + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rc[col];
        }
    }
}

__device__ __forceinline__ void dcl_x_hyb_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            rc[Bcol] += Aval * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rc[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = hlane_id * max_row_len_C; i < (hlane_id + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[i] += rc[Ccol];
        }
    }
}

__device__ __forceinline__ void dcl_x_hyb_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            rc[Bcol] += Aval * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rc[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = hlane_id * min_row_len_C; i < (hlane_id + 1) * min_row_len_C; ++i)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits + 2, i);
            Cvals[i] += rc[col];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (hlane_id == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && hlane_id == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rc[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void dcl_x_hyb_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            rc[Bcol] += Aval * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rc[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dcl_x_hyb_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            rc[Bcol] += Aval * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rc[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rc[Ccol] += __shfl_down_sync(0xffffffff, rc[Ccol], 16);
        if (hwarp_id == 0) Cvals[Arow + i * 16] += rc[Ccol];
    }
}

__device__ __forceinline__ void dcl_x_hyb_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            rc[Bcol] += Aval * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rc[Bcol] += Aval * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
        if (hwarp_id == 0)
            Cvals[Arow * 16 + i] += rc[i];
    }
}

__device__ __forceinline__ void dcl_x_drw_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1, nzC = Cbits[0] + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k != -1)
        {
            k -= 1;
            #pragma unroll
            for (int j = 0; j < 16; ++j)
                rc[j] += Aval * Bvals[k * 16 + j];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = 0; i < 16; ++i)
        {
            if (rc[i] != 0)
            {
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | i);
                Cvals[idx] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dcl_x_drw_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k != -1)
        {
            k -= 1;
            #pragma unroll
            for (int j = 0; j < 16; ++j)
                rc[j] += Aval * Bvals[k * 16 + j];
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = Cbits[Arow]; i < Cbits[Arow + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rc[col];
        }
    }
}

__device__ __forceinline__ void dcl_x_drw_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k != -1)
        {
            k -= 1;
            #pragma unroll
            for (int j = 0; j < 16; ++j)
                rc[j] += Aval * Bvals[k * 16 + j];
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = Arow * max_row_len_C; i < (Arow + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[i] += rc[Ccol];
        }
    }
}

__device__ __forceinline__ void dcl_x_drw_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k != -1)
        {
            k -= 1;
            #pragma unroll
            for (int j = 0; j < 16; ++j)
                rc[j] += Aval * Bvals[k * 16 + j];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = hlane_id * min_row_len_C; i < (hlane_id + 1) * min_row_len_C; ++i)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits + 2, i);
            Cvals[i] += rc[col];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (Arow == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && Arow == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rc[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void dcl_x_drw_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k != -1)
        {
            k -= 1;
            #pragma unroll
            for (int j = 0; j < 16; ++j)
                rc[j] += Aval * Bvals[k * 16 + j];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dcl_x_drw_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k != -1)
        {
            k -= 1;
            for (int j = 0; j < cntC; ++j)
            {
                TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + j);
                rc[Ccol] += Aval * Bvals[k * 16 + Ccol];
            }
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rc[Ccol] += __shfl_down_sync(0xffffffff, rc[Ccol], 16);
        if (hwarp_id == 0) Cvals[Arow + i * 16] += rc[Ccol];
    }
}

__device__ __forceinline__ void dcl_x_drw_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k != -1)
        {
            k -= 1;
            #pragma unroll
            for (int j = 0; j < 16; ++j)
                rc[j] += Aval * Bvals[k * 16 + j];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
        if (hwarp_id == 0)
            Cvals[Arow * 16 + i] += rc[i];
    }
}

__device__ __forceinline__ void dcl_x_dcl_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1, nzC = Cbits[0] + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rc[Bcol] += Aval * Bvals[j * 16 + Acol];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = 0; i < 16; ++i)
        {
            if (rc[i] != 0)
            {
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | i);
                Cvals[idx] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dcl_x_dcl_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rc[Bcol] += Aval * Bvals[j * 16 + Acol];
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = Cbits[Arow]; i < Cbits[Arow + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rc[col];
        }
    }
}

__device__ __forceinline__ void dcl_x_dcl_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rc[Bcol] += Aval * Bvals[j * 16 + Acol];
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = hlane_id * max_row_len_C; i < (hlane_id + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[i] += rc[Ccol];
        }
    }
}

__device__ __forceinline__ void dcl_x_dcl_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rc[Bcol] += Aval * Bvals[j * 16 + Acol];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = hlane_id * min_row_len_C; i < (hlane_id + 1) * min_row_len_C; ++i)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits + 2, i);
            Cvals[i] += rc[col];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (hlane_id == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && hlane_id == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rc[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void dcl_x_dcl_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rc[Bcol] += Aval * Bvals[j * 16 + Acol];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dcl_x_dcl_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rc[Bcol] += Aval * Bvals[j * 16 + Acol];
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rc[Ccol] += __shfl_down_sync(0xffffffff, rc[Ccol], 16);
        if (hwarp_id == 0) Cvals[Arow + i * 16] += rc[Ccol];
    }
}

__device__ __forceinline__ void dcl_x_dcl_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rc[Bcol] += Aval * Bvals[j * 16 + Acol];
        }
    }

#pragma unroll
    for (int i = 0; i < cntB; ++i)
    {
        TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + i);
        rc[Bcol] += __shfl_down_sync(0xffffffff, rc[Bcol], 16);
        if (hwarp_id == 0)
            Cvals[Arow * 16 + Bcol] += rc[Bcol];
    }
}

__device__ __forceinline__ void dcl_x_dns_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, nzC = Cbits[0] + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rc[j] += Aval * Bvals[Acol * 16 + j];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = 0; i < 16; ++i)
        {
            if (rc[i] != 0)
            {
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | i);
                Cvals[idx] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dcl_x_dns_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rc[j] += Aval * Bvals[Acol * 16 + j];
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = Cbits[hlane_id]; i < Cbits[hlane_id + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rc[col];
        }
    }
}

__device__ __forceinline__ void dcl_x_dns_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rc[j] += Aval * Bvals[Acol * 16 + j];
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = hlane_id * max_row_len_C; i < (hlane_id + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[i] += rc[Ccol];
        }
    }
}

__device__ __forceinline__ void dcl_x_dns_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rc[j] += Aval * Bvals[Acol * 16 + j];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        for (int i = hlane_id * min_row_len_C; i < (hlane_id + 1) * min_row_len_C; ++i)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits + 2, i);
            Cvals[i] += rc[col];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (hlane_id == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && hlane_id == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rc[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void dcl_x_dns_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rc[j] += Aval * Bvals[Acol * 16 + j];
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
    }

    if (hwarp_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, hlane_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dcl_x_dns_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rc[j] += Aval * Bvals[Acol * 16 + j];
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rc[Ccol] += __shfl_down_sync(0xffffffff, rc[Ccol], 16);
        if (hwarp_id == 0) Cvals[Arow + i * 16] += rc[Ccol];
    }
}

__device__ __forceinline__ void dcl_x_dns_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Arow = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Acol = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rc[j] += Aval * Bvals[Acol * 16 + j];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 16);
        if (hwarp_id == 0)
            Cvals[Arow * 16 + i] += rc[i];
    }
}