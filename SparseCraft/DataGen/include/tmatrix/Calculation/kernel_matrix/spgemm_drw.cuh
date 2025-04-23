#pragma once
#include <tmatrix/Calculation/kernel_matrix/common.cuh>

__device__ __forceinline__ void drw_x_coo_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = Bbits[0] + 1;
    int nzC = Cbits[0] + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];
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

        j = 0;
        for (; j < nzC; ++j) if (Arow == Cbits[1 + j] >> 4) break;

        while (j < nzC && Arow == Cbits[1 + j] >> 4)
        {
            TileIndex Ccol = Cbits[1 + j] & 15;
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 1);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 2);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 4);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 8);
            if (hlane_id == 0) Cvals[j] += rC[Ccol];
            ++j;
        }
    }
}

__device__ __forceinline__ void drw_x_coo_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = Bbits[0] + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        uint16_t row_map = get_row_map(Cbitmap, Arow);
        MatValue rC[16] = {0};
        MatValue Aval = Avals[i * 16 + hlane_id];
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

#pragma unroll
        for (int j = Cbits[Arow]; j < Cbits[Arow + 1]; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits + 17, j);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 1);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 2);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 4);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 8);
            if (hlane_id == 0)
                Cvals[j] += rC[col];
        }
    }
}

__device__ __forceinline__ void drw_x_coo_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = Bbits[0] + 1;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        uint16_t row_map = get_row_map(Cbitmap, Arow);
        MatValue Aval = Avals[i * 16 + hlane_id];
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

        for (int j = Arow * max_row_len_C; j < (Arow + 1) * max_row_len_C; ++j)
        {
            MatIndex idx = TileIndex_CSR_Get(Cbits, 1 + j);
            if (j > Arow * max_row_len_C && idx == 0) break;
            rC[idx] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[idx], 1);
            rC[idx] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[idx], 2);
            rC[idx] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[idx], 4);
            rC[idx] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[idx], 8);
            if (hlane_id == 0)
                Cvals[j] += rC[idx];
        }
    }
}

__device__ __forceinline__ void drw_x_coo_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = Bbits[0] + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];
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

        for (int j = Arow * min_row_len_C; j < (Arow + 1) * min_row_len_C; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, j);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 1);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 2);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 4);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 8);
            if (hlane_id == 0)
                Cvals[j] += rC[Ccol];
        }

        j = 0;
        for (; j < coo_cnt_C; ++j) if (Arow == Cbits[Celllen + j] >> 4) break;
        while (j < coo_cnt_C && Arow == Cbits[Celllen + j] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + j] & 15;
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 1);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 2);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 4);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 8);
            if (hlane_id == 0)
                Cvals[16 * min_row_len_C + j] += rC[Ccol];
            ++j;
        }
    }
}

__device__ __forceinline__ void drw_x_coo_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = Bbits[0] + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        int idx = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (idx == -1) continue;
        idx -= 1;

        MatValue Aval = Avals[i * 16 + hlane_id];
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

        #pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 1);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 2);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 4);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 8);
            if (hlane_id == 0)
                Cvals[idx * 16 + j] += rC[j];
        }
    }
}

__device__ __forceinline__ void drw_x_coo_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = Bbits[0] + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];
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

        #pragma unroll
        for (int j = 0; j < cntC; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits, 1 + j);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 1);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 2);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 4);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 8);
            if (hlane_id == 0)
                Cvals[Arow + j * 16] += rC[col];
        }
    }
}

__device__ __forceinline__ void drw_x_coo_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = Bbits[0] + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];
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

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 1);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 2);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 4);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 8);

            if (hlane_id == 0)
                Cvals[Arow * 16 + j] += rC[j];
        }
    }
}

__device__ __forceinline__ void drw_x_csr_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int nzC = Cbits[0] + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
        }

        int j = 0;
        for (; j < nzC; ++j) if (Arow == Cbits[1 + j] >> 4) break;

        while (j < nzC && Arow == Cbits[1 + j] >> 4)
        {
            TileIndex Ccol = Cbits[1 + j] & 15;
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 1);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 2);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 4);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 8);
            if (hlane_id == 0) Cvals[j] += rC[Ccol];
            ++j;
        }
    }
}

__device__ __forceinline__ void drw_x_csr_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        uint16_t row_map = get_row_map(Cbitmap, Arow);
        MatValue rC[16] = {0};
        MatValue Aval = Avals[i * 16 + hlane_id];
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
        }

#pragma unroll
        for (int j = Cbits[Arow]; j < Cbits[Arow + 1]; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits + 17, j);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 1);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 2);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 4);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 8);
            if (hlane_id == 0)
                Cvals[j] += rC[col];
        }
    }
}

__device__ __forceinline__ void drw_x_csr_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        uint16_t row_map = get_row_map(Cbitmap, Arow);
        MatValue Aval = Avals[i * 16 + hlane_id];
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
        }

        for (int j = Arow * max_row_len_C; j < (Arow + 1) * max_row_len_C; ++j)
        {
            MatIndex idx = TileIndex_CSR_Get(Cbits, 1 + j);
            rC[idx] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[idx], 1);
            rC[idx] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[idx], 2);
            rC[idx] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[idx], 4);
            rC[idx] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[idx], 8);
            if (hlane_id == 0)
                Cvals[j] += rC[idx];
        }
    }
}

__device__ __forceinline__ void drw_x_csr_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);

        MatValue Aval = Avals[i * 16 + hlane_id];
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
        }

        for (int j = Arow * min_row_len_C; j < (Arow + 1) * min_row_len_C; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, j);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 1);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 2);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 4);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 8);
            if (hlane_id == 0)
                Cvals[j] += rC[Ccol];
        }

        int j = 0;
        for (; j < coo_cnt_C; ++j) if (Arow == Cbits[Celllen + j] >> 4) break;
        while (j < coo_cnt_C && Arow == Cbits[Celllen + j] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + j] & 15;
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 1);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 2);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 4);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 8);
            if (hlane_id == 0)
                Cvals[16 * min_row_len_C + j] += rC[Ccol];
            ++j;
        }
    }
}

__device__ __forceinline__ void drw_x_csr_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        int idx = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (idx == -1) continue;
        idx -= 1;

        MatValue Aval = Avals[i * 16 + hlane_id];
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
        }

        #pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 1);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 2);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 4);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 8);
            if (hlane_id == 0)
                Cvals[idx * 16 + j] += rC[j];
        }
    }
}

__device__ __forceinline__ void drw_x_csr_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            rC[Bcol] += Aval * Bvals[j];
        }

        #pragma unroll
        for (int j = 0; j < cntC; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits, 1 + j);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 1);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 2);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 4);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 8);
            if (hlane_id == 0)
                Cvals[Arow + j * 16] += rC[col];
        }
    }
}

__device__ __forceinline__ void drw_x_csr_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];
        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rC[Bcol] += Aval * Bvals[j];
        }

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 1);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 2);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 4);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 8);

            if (hlane_id == 0)
                Cvals[Arow * 16 + j] += rC[j];
        }
    }
}

__device__ __forceinline__ void drw_x_ell_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int nzC = Cbits[0] + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (j > Acol * max_row_len_B && Bcol == 0) break;
            rC[Bcol] += Aval * Bvals[j];
        }
        int j = 0;
        for (; j < nzC; ++j) if (Arow == Cbits[1 + j] >> 4) break;

        while (j < nzC && Arow == Cbits[1 + j] >> 4)
        {
            TileIndex Ccol = Cbits[1 + j] & 15;
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 1);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 2);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 4);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 8);
            if (hlane_id == 0) Cvals[j] += rC[Ccol];
            ++j;
        }
    }
}

__device__ __forceinline__ void drw_x_ell_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        uint16_t row_map = get_row_map(Cbitmap, Arow);
        MatValue rC[16] = {0};
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (j > Acol * max_row_len_B && Bcol == 0) break;
            rC[Bcol] += Aval * Bvals[j];
        }

        #pragma unroll
        for (int j = Cbits[Arow]; j < Cbits[Arow + 1]; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits + 17, j);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 1);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 2);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 4);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 8);
            if (hlane_id == 0)
                Cvals[j] += rC[col];
        }
    }
}

__device__ __forceinline__ void drw_x_ell_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue rC[16] = {0};
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (j > Acol * max_row_len_B && Bcol == 0) break;
            rC[Bcol] += Aval * Bvals[j];
        }

        for (int j = Arow * max_row_len_C; j < (Arow + 1) * max_row_len_C; ++j)
        {
            MatIndex idx = TileIndex_CSR_Get(Cbits, 1 + j);
            rC[idx] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[idx], 1);
            rC[idx] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[idx], 2);
            rC[idx] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[idx], 4);
            rC[idx] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[idx], 8);
            if (hlane_id == 0)
                Cvals[j] += rC[idx];
        }
    }
}

__device__ __forceinline__ void drw_x_ell_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue rC[16] = {0};
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (j > Acol * max_row_len_B && Bcol == 0) break;
            rC[Bcol] += Aval * Bvals[j];
        }

        for (int j = Arow * min_row_len_C; j < (Arow + 1) * min_row_len_C; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, j);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 1);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 2);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 4);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 8);
            if (hlane_id == 0)
                Cvals[j] += rC[Ccol];
        }

        int j = 0;
        for (; j < coo_cnt_C; ++j) if (Arow == Cbits[Celllen + j] >> 4) break;
        while (j < coo_cnt_C && Arow == Cbits[Celllen + j] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + j] & 15;
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 1);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 2);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 4);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 8);
            if (hlane_id == 0)
                Cvals[16 * min_row_len_C + j] += rC[Ccol];
            ++j;
        }
    }
}

__device__ __forceinline__ void drw_x_ell_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue rC[16] = {0};
        MatValue Aval = Avals[i * 16 + hlane_id];

        int idx = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (idx == -1) continue;
        idx -= 1;

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (j > Acol * max_row_len_B && Bcol == 0) break;
            rC[Bcol] += Aval * Bvals[j];
        }

        #pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 1);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 2);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 4);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 8);
            if (hlane_id == 0)
                Cvals[idx * 16 + j] += rC[j];
        }
    }
}

__device__ __forceinline__ void drw_x_ell_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue rC[16] = {0};
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (j > Acol * max_row_len_B && Bcol == 0) break;
            rC[Bcol] += Aval * Bvals[j];
        }

        #pragma unroll
        for (int j = 0; j < cntC; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits, 1 + j);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 1);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 2);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 4);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 8);
            if (hlane_id == 0)
                Cvals[Arow + j * 16] += rC[col];
        }
    }
}

__device__ __forceinline__ void drw_x_ell_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (j > Acol * max_row_len_B && Bcol == 0) break;
            rC[Bcol] += Aval * Bvals[j];
        }

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 1);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 2);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 4);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 8);

            if (hlane_id == 0)
                Cvals[Arow * 16 + j] += rC[j];
        }
    }
}

__device__ __forceinline__ void drw_x_hyb_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int nzC = Cbits[0] + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

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

        j = 0;
        for (; j < nzC; ++j) if (Arow == Cbits[1 + j] >> 4) break;

        while (j < nzC && Arow == Cbits[1 + j] >> 4)
        {
            TileIndex Ccol = Cbits[1 + j] & 15;
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 1);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 2);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 4);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 8);
            if (hlane_id == 0) Cvals[j] += rC[Ccol];
            ++j;
        }
    }
}

__device__ __forceinline__ void drw_x_hyb_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

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

        #pragma unroll
        for (int j = Cbits[Arow]; j < Cbits[Arow + 1]; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits + 17, j);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 1);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 2);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 4);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 8);
            if (hlane_id == 0)
                Cvals[j] += rC[col];
        }
    }
}

__device__ __forceinline__ void drw_x_hyb_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

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

        for (int j = Arow * max_row_len_C; j < (Arow + 1) * max_row_len_C; ++j)
        {
            MatIndex idx = TileIndex_CSR_Get(Cbits, 1 + j);
            rC[idx] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[idx], 1);
            rC[idx] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[idx], 2);
            rC[idx] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[idx], 4);
            rC[idx] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[idx], 8);
            if (hlane_id == 0)
                Cvals[j] += rC[idx];
        }
    }
}

__device__ __forceinline__ void drw_x_hyb_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue rC[16] = {0};
        MatValue Aval = Avals[i * 16 + hlane_id];

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

        for (int j = Arow * min_row_len_C; j < (Arow + 1) * min_row_len_C; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, j);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 1);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 2);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 4);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 8);
            if (hlane_id == 0)
                Cvals[j] += rC[Ccol];
        }

        j = 0;
        for (; j < coo_cnt_C; ++j) if (Arow == Cbits[Celllen + j] >> 4) break;
        while (j < coo_cnt_C && Arow == Cbits[Celllen + j] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + j] & 15;
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 1);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 2);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 4);
            rC[Ccol] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[Ccol], 8);
            if (hlane_id == 0)
                Cvals[16 * min_row_len_C + j] += rC[Ccol];
            ++j;
        }
    }
}

__device__ __forceinline__ void drw_x_hyb_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        int idx = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (idx == -1) continue;
        idx -= 1;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, j);
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

        #pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 1);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 2);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 4);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 8);
            if (hlane_id == 0)
                Cvals[idx * 16 + j] += rC[j];
        }
    }
}

__device__ __forceinline__ void drw_x_hyb_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

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

        #pragma unroll
        for (int j = 0; j < cntC; ++j)
        {
            TileIndex col = TileIndex_CSR_Get(Cbits, 1 + j);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 1);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 2);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 4);
            rC[col] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[col], 8);
            if (hlane_id == 0)
                Cvals[Arow + j * 16] += rC[col];
        }
    }
}

__device__ __forceinline__ void drw_x_hyb_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        MatValue rC[16] = {0};
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

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

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 1);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 2);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 4);
            rC[j] += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC[j], 8);

            if (hlane_id == 0)
                Cvals[Arow * 16 + j] += rC[j];
        }
    }
}

__device__ __forceinline__ void drw_x_drw_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int nzC = Cbits[0] + 1;

    for (int i = hwarp_id; i < cntB; i += 2)
    {
        TileIndex Brow = TileIndex_CSR_Get(Bbits, 1 + i);
        MatValue Bval = Bvals[i * 16 + hlane_id];
        bool flag = (hwarp_id == 0 && (i + 1 < cntB)) || (hwarp_id == 1);

        for (int j = 0; j < cntA; ++j)
        {
            TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + j);
            MatValue part = Avals[j * 16 + Brow] * Bval;
            if (flag) part += __shfl_down_sync(0xffffffff, part, 16);
            if (hwarp_id == 0) {
                int idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow << 4 | hlane_id);
                // int idx_map = check_bitmap_idx_slow(Cbitmap, Arow << 4 | hlane_id);
                // if (idx_map != idx) printf("idx_map: %d, idx: %d\n", idx_map, idx);
                if (idx != -1) Cvals[idx] += part;
            }
        }
    }
}

__device__ __forceinline__ void drw_x_drw_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = hwarp_id; i < cntB; i += 2)
    {
        TileIndex Brow = TileIndex_CSR_Get(Bbits, 1 + i);
        MatValue Bval = Bvals[i * 16 + hlane_id];
        bool flag = (hwarp_id == 0 && (i + 1 < cntB)) || (hwarp_id == 1);

        for (int j = 0; j < cntA; ++j)
        {
            TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + j);
            MatValue part = Avals[j * 16 + Brow] * Bval;
            if (flag) part += __shfl_down_sync(0xffffffff, part, 16);
            if (hwarp_id == 0) {
                uint16_t row_map = get_row_map(Cbitmap, Arow);
                int idx = check_bitmap_row_idx(row_map, hlane_id);
                if (idx != -1) Cvals[Cbits[Arow] + idx] += part;
            }
        }
    }
}

__device__ __forceinline__ void drw_x_drw_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntB; i += 2)
    {
        TileIndex Brow = TileIndex_CSR_Get(Bbits, 1 + i);
        MatValue Bval = Bvals[i * 16 + hlane_id];
        bool flag = (hwarp_id == 0 && (i + 1 < cntB)) || (hwarp_id == 1);

        for (int j = 0; j < cntA; ++j)
        {
            TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + j);
            MatValue part = Avals[j * 16 + Brow] * Bval;
            if (flag) part += __shfl_down_sync(0xffffffff, part, 16);
            if (hwarp_id == 0) {
                uint16_t row_map = get_row_map(Cbitmap, Arow);
                int idx = check_bitmap_row_idx(row_map, hlane_id);
                if (idx != -1) Cvals[Arow * max_row_len_C + idx] += part;
            }
        }
    }
}

__device__ __forceinline__ void drw_x_drw_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = hwarp_id; i < cntB; i += 2)
    {
        TileIndex Brow = TileIndex_CSR_Get(Bbits, 1 + i);
        MatValue Bval = Bvals[i * 16 + hlane_id];
        bool flag = (hwarp_id == 0 && (i + 1 < cntB)) || (hwarp_id == 1); // both warps working

        for (int j = 0; j < cntA; ++j)
        {
            TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + j);
            MatValue part = Avals[j * 16 + Brow] * Bval;
            if (flag) part += __shfl_down_sync(0xffffffff, part, 16);
            if (hwarp_id == 0 && part != 0) {
                int idx_ell = binary_find_index_4bit(Cbits + 2, hlane_id, Arow * min_row_len_C, (Arow + 1) * min_row_len_C);
                int idx_coo = binary_find_index<TileIndex>(Cbits + Celllen, coo_cnt_C, Arow << 4 | hlane_id);
                if (idx_ell != -1) Cvals[idx_ell] += part;
                if (idx_coo != -1) Cvals[16 * min_row_len_C + idx_coo] += part;
            }
        }
    }
}

__device__ __forceinline__ void drw_x_drw_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    uint16_t c_row_map = 0;
    for (int i = 0; i < cntC; ++i) c_row_map |= 1 << TileIndex_CSR_Get(Cbits, 1 + i);

    for (int i = hwarp_id; i < cntB; i += 2)
    {
        TileIndex Brow = TileIndex_CSR_Get(Bbits, 1 + i);
        MatValue Bval = Bvals[i * 16 + hlane_id];
        bool flag = (hwarp_id == 0 && (i + 1 < cntB)) || (hwarp_id == 1);

        for (int j = 0; j < cntA; ++j)
        {
            TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + j);
            uint16_t a_row_map = 1 << Arow;
            if (a_row_map & c_row_map) {
                MatValue part = Avals[j * 16 + Brow] * Bval;
                if (flag) part += __shfl_down_sync(0xffffffff, part, 16);
                if (hwarp_id == 0) {
                    int idx = __popc((c_row_map & (a_row_map - 1)));
                    Cvals[idx * 16 + hlane_id] += part;
                }
            }
        }
    }
}

__device__ __forceinline__ void drw_x_drw_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    uint16_t c_col_map = 0, hlane_mask = 1 << hlane_id;
    for (int i = 0; i < cntC; ++i) c_col_map |= 1 << TileIndex_CSR_Get(Cbits, 1 + i);
    bool active = c_col_map & hlane_mask;
    int active_warp_mask = __ballot_sync(0xffffffff, active);

    if (active)
    for (int i = hwarp_id; i < cntB; i += 2)
    {
        TileIndex Brow = TileIndex_CSR_Get(Bbits, 1 + i);
        MatValue Bval = Bvals[i * 16 + hlane_id];
        bool flag = (hwarp_id == 0 && (i + 1 < cntB)) || (hwarp_id == 1);

        for (int j = 0; j < cntA; ++j)
        {
            TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + j);
            MatValue part = Avals[j * 16 + Brow] * Bval;
            if (flag) part += __shfl_down_sync(active_warp_mask, part, 16);
            if (hwarp_id == 0) {
                int idx = __popc((c_col_map & (hlane_mask - 1)));
                Cvals[Arow + idx * 16] += part;
            }
        }
    }
}

__device__ __forceinline__ void drw_x_drw_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = hwarp_id; i < cntB; i += 2)
    {
        TileIndex Brow = TileIndex_CSR_Get(Bbits, 1 + i);
        MatValue Bval = Bvals[i * 16 + Acol];
        bool flag = (hwarp_id == 0 && (i + 1 < cntB)) || (hwarp_id == 1);

        for (int j = 0; j < cntA; ++j)
        {
            TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + j);
            MatValue part = Avals[j * 16 + Brow] * Bval;
            if (flag) part += __shfl_down_sync(0xffffffff, part, 16);
            if (hwarp_id == 0) Cvals[Arow * 16 + Acol] += part;
        }
    }
}

__device__ __forceinline__ void drw_x_dcl_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1, nzC = Cbits[0] + 1;
    uint16_t row_map = 0, col_map = 0;
    for (int i = 0; i < cntA; ++i)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        row_map |= 1 << Arow;
    }
    for (int i = 0; i < cntB; ++i)
    {
        TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + i);
        col_map |= 1 << Bcol;
    }

    for (int i = hwarp_id; i < nzC; i += 2)
    {
        TileIndex idx = Cbits[1 + i], Ccol = idx & 15, Crow = idx >> 4;
        uint16_t row_mask = 1 << Crow, col_mask = 1 << Ccol;
        bool flag = row_map & row_mask && col_map & col_mask;
        int active_mask = __ballot_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, flag);
        if (flag)
        {
            int idxA = __popc(row_map & (row_mask - 1));
            int idxB = __popc(col_map & (col_mask - 1));

            MatValue rc = Avals[idxA * 16 + hlane_id] * Bvals[idxB * 16 + hlane_id];
            rc += __shfl_down_sync(active_mask, rc, 1);
            rc += __shfl_down_sync(active_mask, rc, 2);
            rc += __shfl_down_sync(active_mask, rc, 4);
            rc += __shfl_down_sync(active_mask, rc, 8);
            if (hlane_id == 0) Cvals[i] += rc;
        }
    }
}

__device__ __forceinline__ void drw_x_dcl_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    uint16_t col_map = 0;
    for (int i = 0; i < cntB; ++i)
    {
        TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + i);
        col_map |= 1 << Bcol;
    }
    
    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);

        for (int j = Cbits[Arow]; j < Cbits[Arow + 1]; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 17, j);
            uint16_t col_mask = 1 << Ccol;
            bool flag = col_map & col_mask;
            int active_mask = __ballot_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, flag);
            if (col_map & col_mask)
            {
                int idxB = __popc(col_map & (col_mask - 1));
                MatValue rc = Avals[i * 16 + hlane_id] * Bvals[idxB * 16 + hlane_id];
                rc += __shfl_down_sync(active_mask, rc, 1);
                rc += __shfl_down_sync(active_mask, rc, 2);
                rc += __shfl_down_sync(active_mask, rc, 4);
                rc += __shfl_down_sync(active_mask, rc, 8);
                if (hlane_id == 0) Cvals[j] += rc;
            }
        }
    }
}

__device__ __forceinline__ void drw_x_dcl_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    uint16_t col_map = 0;
    for (int i = 0; i < cntB; ++i)
    {
        TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + i);
        col_map |= 1 << Bcol;
    }

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);

        for (int j = Arow * max_row_len_C; j < (Arow + 1) * max_row_len_C; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + j);
            uint16_t col_mask = 1 << Ccol;
            bool flag = col_map & col_mask;
            int active_mask = __ballot_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, flag);
            if (col_map & col_mask)
            {
                int idxB = __popc(col_map & (col_mask - 1));
                MatValue rc = Avals[i * 16 + hlane_id] * Bvals[idxB * 16 + hlane_id];
                rc += __shfl_down_sync(active_mask, rc, 1);
                rc += __shfl_down_sync(active_mask, rc, 2);
                rc += __shfl_down_sync(active_mask, rc, 4);
                rc += __shfl_down_sync(active_mask, rc, 8);
                if (hlane_id == 0) Cvals[j] += rc;
            }
        }
    }
}

__device__ __forceinline__ void drw_x_dcl_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    uint16_t col_map = 0;
    for (int i = 0; i < cntB; ++i)
    {
        TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + i);
        col_map |= 1 << Bcol;
    }

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);

        for (int j = Arow * min_row_len_C; j < (Arow + 1) * min_row_len_C; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, j);
            uint16_t col_mask = 1 << Ccol;
            bool flag = col_map & col_mask;
            int active_mask = __ballot_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, flag);
            if (col_map & col_mask)
            {
                int idxB = __popc(col_map & (col_mask - 1));
                MatValue rc = Avals[i * 16 + hlane_id] * Bvals[idxB * 16 + hlane_id];
                rc += __shfl_down_sync(active_mask, rc, 1);
                rc += __shfl_down_sync(active_mask, rc, 2);
                rc += __shfl_down_sync(active_mask, rc, 4);
                rc += __shfl_down_sync(active_mask, rc, 8);
                if (hlane_id == 0) Cvals[j] += rc;
            }
        }

        int j = 0;
        for (; j < coo_cnt_C; ++j) if (Arow == Cbits[Celllen + j] >> 4) break;
        while (j < coo_cnt_C && Arow == Cbits[Celllen + j] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + j] & 15;
            uint16_t col_mask = 1 << Ccol;
            bool flag = col_map & col_mask;
            int active_mask = __ballot_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, flag);
            if (flag)
            {
                int idxB = __popc(col_map & (col_mask - 1));
                MatValue rc = Avals[i * 16 + hlane_id] * Bvals[idxB * 16 + hlane_id];
                rc += __shfl_down_sync(active_mask, rc, 1);
                rc += __shfl_down_sync(active_mask, rc, 2);
                rc += __shfl_down_sync(active_mask, rc, 4);
                rc += __shfl_down_sync(active_mask, rc, 8);
                if (hlane_id == 0) Cvals[16 * min_row_len_C + j] += rc;
            }
            ++j;
        }
    }
}

__device__ __forceinline__ void drw_x_dcl_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);

        int k = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        #pragma unroll
        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            MatValue rc = Avals[i * 16 + hlane_id] * Bvals[j * 16 + hlane_id];
            rc += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rc, 1);
            rc += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rc, 2);
            rc += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rc, 4);
            rc += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rc, 8);
            if (hlane_id == 0) Cvals[k + Bcol] += rc;
        }
    }
}

__device__ __forceinline__ void drw_x_dcl_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    uint16_t c_col_map = 0;
    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        c_col_map |= 1 << Ccol;
    }

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            int col_mask = 1 << Bcol;
            bool flag = c_col_map & col_mask;
            int active_mask = __ballot_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, flag);
            if (flag) {
                int idx = __popc(c_col_map & (col_mask - 1));
                MatValue rc = Avals[i * 16 + hlane_id] * Bvals[j * 16 + hlane_id];
                rc += __shfl_down_sync(active_mask, rc, 1);
                rc += __shfl_down_sync(active_mask, rc, 2);
                rc += __shfl_down_sync(active_mask, rc, 4);
                rc += __shfl_down_sync(active_mask, rc, 8);
                if (hlane_id == 0) Cvals[idx * 16 + Arow] += rc;
            }
        }
    }
}

__device__ __forceinline__ void drw_x_dcl_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    TileIndex Acol = hlane_id;
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            MatValue rC = Aval * Bvals[j * 16 + Acol];
            rC += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC, 1);
            rC += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC, 2);
            rC += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC, 4);
            rC += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, rC, 8);
            if (hlane_id == 0) Cvals[Arow * 16 + Bcol] += rC;
        }
    }
}

__device__ __forceinline__ void drw_x_dns_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, nzC = Cbits[0] + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval  = Avals[i * 16 + hlane_id];

        int j = 0;
        for (;j < nzC; ++j) if (Arow == Cbits[1 + j] >> 4) break;
        while (j < nzC && Arow == Cbits[1 + j] >> 4)
        {
            TileIndex Ccol = Cbits[1 + j] & 15;
            MatValue res = Aval * Bvals[Ccol + hlane_id * 16];
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 1);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 2);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 4);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 8);
            if (hlane_id == 0)
                Cvals[j] += res;
            ++j;
        }
    }
}

__device__ __forceinline__ void drw_x_dns_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;
    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Cbits[Arow]; j < Cbits[Arow + 1]; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 17, j);
            MatValue res = Aval * Bvals[Ccol + hlane_id * 16];
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 1);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 2);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 4);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 8);
            if (hlane_id == 0)
                Cvals[j] += res;
        }
    }
}

__device__ __forceinline__ void drw_x_dns_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = Arow * max_row_len_C; j < (Arow + 1) * max_row_len_C; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + j);
            if (j > Arow * max_row_len_C && Ccol == 0) break;
            MatValue res = Aval * Bvals[Ccol + hlane_id * 16];
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 1);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 2);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 4);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 8);
            if (hlane_id == 0) Cvals[j] += res;
        }
    }
}

__device__ __forceinline__ void drw_x_dns_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = 0; j < min_row_len_C; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, Arow * min_row_len_C + j);
            MatValue res = Aval * Bvals[Ccol + hlane_id * 16];
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 1);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 2);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 4);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 8);
            if (hlane_id == 0)
                Cvals[Arow * min_row_len_C + j] += res;
        }

        int  j = 0;
        for (; j < coo_cnt_C; ++j) if (Arow == Cbits[Celllen + j] >> 4) break;
        while (j < coo_cnt_C && Arow == Cbits[Celllen + j] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + j] & 15;
            MatValue res = Aval * Bvals[Ccol + hlane_id * 16];
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 1);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 2);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 4);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 8);
            if (hlane_id == 0)
                Cvals[16 * min_row_len_C + j] += res;
            ++j;
        }
    }
}

__device__ __forceinline__ void drw_x_dns_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        int k = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        #pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            MatValue res = Aval * Bvals[hlane_id * 16 + j];
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 1);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 2);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 4);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 8);
            if (hlane_id == 0) Cvals[k + j] += res;
        }
    }
}

__device__ __forceinline__ void drw_x_dns_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];
        
        for (int j = 0; j < cntC; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + j);
            MatValue res = Aval * Bvals[hlane_id * 16 + Ccol];
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 1);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 2);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 4);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 8);
            if (hlane_id == 0) Cvals[j * 16 + Arow] += res;
        }
    }
}

__device__ __forceinline__ void drw_x_dns_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntA = TileIndex_CSR_Get(Abits, 0) + 1;

    for (int i = hwarp_id; i < cntA; i += 2)
    {
        TileIndex Arow = TileIndex_CSR_Get(Abits, 1 + i);
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = 0; j < 16; ++j)
        {
            MatValue res = Aval * Bvals[j + hlane_id * 16];
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 1);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 2);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 4);
            res += __shfl_down_sync(hwarp_id ? 0xffff0000 : 0x0000ffff, res, 8);
            if (hlane_id == 0) Cvals[Arow * 16 + j] += res;
        }
    }
}