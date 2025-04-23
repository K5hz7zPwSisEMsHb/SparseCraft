#pragma once
#include "./common.cuh"
#include <tmatrix/DataStructure/TileMatrix.h>

__device__ __forceinline__ void coo_x_coo_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, nzB = Bbits[0] + 1, nzC = Cbits[0] + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

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
}

__device__ __forceinline__ void coo_x_coo_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, nzB = Bbits[0] + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int idx = Abits[i + 1], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i];
        uint16_t row_map = get_row_map(Cbitmap, Arow);
        int j = 0;
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;

        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Cbits[Arow] + idx, Aval * Bvals[j]);
            ++j;
        }
    }
}

__device__ __forceinline__ void coo_x_coo_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, nzB = Bbits[0] + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    // if (debug_flag && lane_id == 0) printf("0x%016llx 0x%016llx 0x%016llx 0x%016llx | Cvals: %0.lf\n", Cbitmap[0], Cbitmap[1], Cbitmap[2], Cbitmap[3], Cvals[12 * max_row_len_C + 6]);

    for (int i = lane_id; i < nzA; i += 32)
    {
        int idx = Abits[i + 1], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i];
        uint16_t row_map = get_row_map(Cbitmap, Arow);
        int j = 0;
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;

        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Arow * max_row_len_C + idx, Aval * Bvals[j]);
            // if (debug_flag && Arow == 12 && Bcol == 12) printf("[%02d] row map: 0x%04x, Arow: %d, Bcol: %d, Aval: %.0lf, Bval: %.0lf, Cval: %.0lf, idx: %d\n", lane_id, row_map, Arow, Bcol, Aval, Bvals[j], Cvals[Arow * max_row_len_C + idx], idx);
            ++j;
        }
    }
    // __syncwarp();
    // if (debug_flag && lane_id == 0) printf("Cvals: %0.lf\n", Cvals[12 * max_row_len_C + 6]);
}

__device__ __forceinline__ void coo_x_coo_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, nzB = Bbits[0] + 1, min_row_len_C = Cbits[0], C_coo_cnt = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int idx = Abits[i + 1], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i];
        int j = 0;
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;

        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            int idx_ell = binary_find_index_4bit(Cbits + 2, Bcol, Arow * min_row_len_C, (Arow + 1) * min_row_len_C);
            int idx_coo = binary_find_index<TileIndex>(Cbits + Celllen, C_coo_cnt, Arow * 16 + Bcol);
            if (idx_ell != -1) atomicAdd(Cvals + idx_ell, Aval * Bvals[j]);
            if (idx_coo != -1) atomicAdd(Cvals + min_row_len_C * 16 + idx_coo, Aval * Bvals[j]);
            ++j;
        }
    }
}

__device__ __forceinline__ void coo_x_coo_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, nzB = Bbits[0] + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int idx = Abits[i + 1], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i];
        
        int j = 0, k = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;
        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            atomicAdd(Cvals + k + Bcol, Aval * Bvals[j]);
            ++j;
        }
    }
}

__device__ __forceinline__ void coo_x_coo_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, nzB = Bbits[0] + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    uint16_t col_map = 0;
    for (int k = 0; k < cntC; ++k) col_map |= 1 << TileIndex_CSR_Get(Cbits, 1 + k);

    for (int i = lane_id; i < nzA; i += 32)
    {
        int idx = Abits[i + 1], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i];
        int j = 0;
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;

        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            uint16_t col_mask = 1 << Bcol;
            if (col_mask & col_map) {
                int col_idx = __popc(col_map & (col_mask - 1));
                atomicAdd(Cvals + Arow + 16 * col_idx, Aval * Bvals[j]);
            }
            ++j;
        }
    }
}

__device__ __forceinline__ void coo_x_coo_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, nzB = Bbits[0] + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int idx = Abits[i + 1], Acol = idx & 15, Arow = idx >> 4;
        MatValue Aval = Avals[i];

        int j = 0;
        for (; j < nzB; ++j) if (Acol == Bbits[1 + j] >> 4) break;
        while (j < nzB && Acol == Bbits[1 + j] >> 4)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            atomicAdd(Cvals + Arow * 16 + Bcol, Aval * Bvals[j]);
            ++j;
        }
    }
}

__device__ __forceinline__ void coo_x_csr_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, nzC = Cbits[0] + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            // int idx = check_bitmap_idx(Cbitmap, Arow * 16 + Bcol);
            int idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow * 16 + Bcol);
            if (idx != -1) atomicAdd(Cvals + idx, Aval * Bvals[j]);
        }
    }
}

__device__ __forceinline__ void coo_x_csr_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4, Cidx = Cbits[Arow];
        MatValue Aval = Avals[i];
        uint16_t row_map = get_row_map(Cbitmap, Arow);

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Cidx + idx, Aval * Bvals[j]);
        }
    }
}

__device__ __forceinline__ void coo_x_csr_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];
        uint16_t row_map = get_row_map(Cbitmap, Arow);

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Arow * max_row_len_C + idx, Aval * Bvals[j]);
        }
    }
}

__device__ __forceinline__ void coo_x_csr_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, min_row_len_C = Cbits[0], C_coo_cnt = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            int idx_ell = binary_find_index_4bit(Cbits + 2, Bcol, Arow * min_row_len_C, (Arow + 1) * min_row_len_C);
            int idx_coo = binary_find_index<TileIndex>(Cbits + Celllen, C_coo_cnt, Arow * 16 + Bcol);
            if (idx_ell != -1) atomicAdd(Cvals + idx_ell, Aval * Bvals[j]);
            if (idx_coo != -1) atomicAdd(Cvals + min_row_len_C * 16 + idx_coo, Aval * Bvals[j]);
        }
    }
}

__device__ __forceinline__ void coo_x_csr_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    uint16_t row_map = 0;
    for (int k = 0; k < cntC; ++k) row_map |= 1 << TileIndex_CSR_Get(Cbits, 1 + k);

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

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
}

__device__ __forceinline__ void coo_x_csr_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    unsigned short col_map = 0;
    for (int k = 0; k < cntC; ++k) col_map |= 1 << TileIndex_CSR_Get(Cbits, 1 + k);

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 17, j);
            uint16_t col_mask = 1 << Bcol;
            if (col_mask & col_map) {
                int col_idx = __popc(col_map & (col_mask - 1));
                atomicAdd(Cvals + Arow + 16 * col_idx, Aval * Bvals[j]);
            }
        }
    }
}

__device__ __forceinline__ void coo_x_csr_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        TileIndex idx = Abits[i + 1], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i];

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            atomicAdd(Cvals + Arow * 16 + Bcol, Aval * Bvals[j]);
        }
    }
}

__device__ __forceinline__ void coo_x_ell_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, nzC = Cbits[0] + 1, max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];
        int start = Acol * max_row_len_B, end = (Acol + 1) * max_row_len_B;

        for (int j = start; j < end; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (j > start && Bcol == 0) break;
            int idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow * 16 + Bcol);
            if (idx != -1) atomicAdd(Cvals + idx, Aval * Bvals[j]);
        }
    }
}

__device__ __forceinline__ void coo_x_ell_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];
        uint16_t row_map = get_row_map(Cbitmap, Arow);
        int start = Acol * max_row_len_B, end = (Acol + 1) * max_row_len_B;

        for (int j = start; j < end; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (j > start && Bcol == 0) break;
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Cbits[Arow] + idx, Aval * Bvals[j]);
        }
    }
}

__device__ __forceinline__ void coo_x_ell_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];
        uint16_t row_map = get_row_map(Cbitmap, Arow);
        int start = Acol * max_row_len_B, end = (Acol + 1) * max_row_len_B;

        for (int j = start; j < end; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (j > start && Bcol == 0) break;
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Arow * max_row_len_C + idx, Aval * Bvals[j]);
        }
    }
}

__device__ __forceinline__ void coo_x_ell_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1, min_row_len_C = Cbits[0], C_coo_cnt = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4, Cidx = Arow * min_row_len_C, C_end = (Arow + 1) * min_row_len_C;
        MatValue Aval = Avals[i];
        int start = Acol * max_row_len_B, end = (Acol + 1) * max_row_len_B;

        for (int j = start; j < end; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (j > start && Bcol == 0) break;
            int idx_ell = binary_find_index_4bit(Cbits + 2, Bcol, Cidx, C_end);
            int idx_coo = binary_find_index<TileIndex>(Cbits + Celllen, C_coo_cnt, Arow * 16 + Bcol);
            if (idx_ell != -1) atomicAdd(Cvals + idx_ell, Aval * Bvals[j]);
            if (idx_coo != -1) atomicAdd(Cvals + min_row_len_C * 16 + idx_coo, Aval * Bvals[j]);
        }
    }
}

__device__ __forceinline__ void coo_x_ell_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        int k = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        int start = Acol * max_row_len_B, end = (Acol + 1) * max_row_len_B;

        for (int j = start; j < end; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (j > start && Bcol == 0) break;
            atomicAdd(Cvals + k + Bcol, Aval * Bvals[j]);
        }
    }
}

__device__ __forceinline__ void coo_x_ell_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    unsigned short col_map = 0;
    for (int k = 0; k < cntC; ++k) col_map |= 1 << TileIndex_CSR_Get(Cbits, 1 + k);

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];
        int start = Acol * max_row_len_B, end = (Acol + 1) * max_row_len_B;

        for (int j = start; j < end; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (j > start && Bcol == 0) break;
            if ((1 << Bcol) & col_map) {
                int col_idx = __popc(col_map & ((1 << Bcol) - 1));
                atomicAdd(Cvals + Arow + 16 * col_idx, Aval * Bvals[j]);
            }
        }
    }
}

__device__ __forceinline__ void coo_x_ell_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        TileIndex idx = Abits[i + 1], Acol = idx & 15, Arow = idx ^ Acol;
        MatValue Aval = Avals[i];
        int start = Acol * max_row_len_B, end = (Acol + 1) * max_row_len_B;

        for (int j = start; j < end; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            if (j > start && Bcol == 0) break;
            atomicAdd(Cvals + Arow + Bcol, Aval * Bvals[j]);
        }
    }
}

__device__ __forceinline__ void coo_x_hyb_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, nzC = Cbits[0] + 1, min_row_len_B = Bbits[0], B_coo_cnt = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, j);
            int idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow * 16 + Bcol);
            if (idx != -1) atomicAdd(Cvals + idx, Aval * Bvals[j]);
        }

        int j = 0;
        for (; j < B_coo_cnt; ++j) if (Acol == Bbits[Belllen + j] >> 4) break;
        while (j < B_coo_cnt && Acol == Bbits[Belllen + j] >> 4)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            int idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow * 16 + Bcol);
            if (idx != -1) atomicAdd(Cvals + idx, Aval * Bvals[16 * min_row_len_B + j]);
            ++j;
        }
    }
}

__device__ __forceinline__ void coo_x_hyb_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    // if (debug_flag && lane_id == 0) printf("coo_x_hyb_2_csr, Cvals[2]: %0.lf\n", Cvals[2]);
    int nzA = Abits[0] + 1, min_row_len_B = Bbits[0], B_coo_cnt = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];
        uint16_t row_map = get_row_map(Cbitmap, Arow);

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, j);
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Cbits[Arow] + idx, Aval * Bvals[j]);
            // if (debug_flag && Arow == 0 && Bcol == 13) printf("[%d] (ELL) Aval: %.0lf Bval: %.0lf, idx: %d \n", lane_id, Aval, Bvals[j], idx);
        }

        int j = 0;
        for (; j < B_coo_cnt; ++j) if (Acol == Bbits[Belllen + j] >> 4) break;
        while (j < B_coo_cnt && Acol == Bbits[Belllen + j] >> 4)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Cbits[Arow] + idx, Aval * Bvals[16 * min_row_len_B + j]);
            // if (debug_flag && Arow == 0 && Bcol == 13) printf("[%d] (COO) Aval: %.0lf Bval: %.0lf, idx: %d, Cval[idx]: %.0lf \n", lane_id, Aval, Bvals[16 * min_row_len_B + j], idx, Cvals[idx]);
            ++j;
        }
    }
    // __syncwarp();
    // if (debug_flag && lane_id == 0) printf("coo_x_hyb_2_csr, Cvals[2]: %0.lf\n", Cvals[2]);
}

__device__ __forceinline__ void coo_x_hyb_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, min_row_len_B = Bbits[0], B_coo_cnt = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];
        uint16_t row_map = get_row_map(Cbitmap, Arow);

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, j);
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Arow * max_row_len_C + idx, Aval * Bvals[j]);
        }

        int j = 0;
        for (; j < B_coo_cnt; ++j) if (Acol == Bbits[Belllen + j] >> 4) break;
        while (j < B_coo_cnt && Acol == Bbits[Belllen + j] >> 4)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            int idx = check_bitmap_row_idx(row_map, Bcol);
            if (idx != -1) atomicAdd(Cvals + Arow * max_row_len_C + idx, Aval * Bvals[j + min_row_len_B * 16]);
            ++j;
        }
    }
}

__device__ __forceinline__ void coo_x_hyb_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, 
        min_row_len_B = Bbits[0], B_coo_cnt = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2, 
        min_row_len_C = Cbits[0], C_coo_cnt = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    // if (lane_id == 0) printf("%d %d %d\n", min_row_len_C, C_coo_cnt, Celllen);

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, j);
            int idx_ell = binary_find_index_4bit(Cbits + 2, Bcol, Arow * min_row_len_C, (Arow + 1) * min_row_len_C);
            int idx_coo = binary_find_index<TileIndex>(Cbits + Celllen, C_coo_cnt, Arow << 4 | Bcol);
            if (idx_ell != -1) atomicAdd(Cvals + idx_ell, Aval * Bvals[j]);
            if (idx_coo != -1) atomicAdd(Cvals + min_row_len_C * 16 + idx_coo, Aval * Bvals[j]);
        }

        int j = 0;
        for (; j < B_coo_cnt; ++j) if (Acol == Bbits[Belllen + j] >> 4) break;

        while (j < B_coo_cnt && Acol == Bbits[Belllen + j] >> 4)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            int idx_ell = binary_find_index_4bit(Cbits + 2, Bcol, Arow * min_row_len_C, (Arow + 1) * min_row_len_C);
            int idx_coo = binary_find_index<TileIndex>(Cbits + Celllen, C_coo_cnt, Arow << 4 | Bcol);
            if (idx_ell != -1) atomicAdd(Cvals + idx_ell, Aval * Bvals[j + min_row_len_B * 16]);
            if (idx_coo != -1) atomicAdd(Cvals + min_row_len_C * 16 + idx_coo, Aval * Bvals[j + min_row_len_B * 16]);
            ++j;
        }
    }
}

__device__ __forceinline__ void coo_x_hyb_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, min_row_len_B = Bbits[0], B_coo_cnt = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        int k = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, j);
            atomicAdd(Cvals + k + Bcol, Aval * Bvals[j]);
        }

        int j = 0;
        for (; j < B_coo_cnt; ++j) if (Acol == Bbits[Belllen + j] >> 4) break;
        while (j < B_coo_cnt && Acol == Bbits[Belllen + j] >> 4)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            atomicAdd(Cvals + k + Bcol, Aval * Bvals[j + min_row_len_B * 16]);
            ++j;
        }
    }
}

__device__ __forceinline__ void coo_x_hyb_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, min_row_len_B = Bbits[0], B_coo_cnt = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    unsigned short col_map = 0;
    for (int k = 0; k < cntC; ++k) col_map |= 1 << TileIndex_CSR_Get(Cbits, 1 + k);

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, j);
            if ((1 << Bcol) & col_map) {
                int col_idx = __popc(col_map & ((1 << Bcol) - 1));
                atomicAdd(Cvals + Arow + 16 * col_idx, Aval * Bvals[j]);
            }
        }

        int j = 0;
        for (; j < B_coo_cnt; ++j) if (Acol == Bbits[Belllen + j] >> 4) break;
        while (j < B_coo_cnt && Acol == Bbits[Belllen + j] >> 4)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            if ((1 << Bcol) & col_map) {
                int col_idx = __popc(col_map & ((1 << Bcol) - 1));
                atomicAdd(Cvals + Arow + 16 * col_idx, Aval * Bvals[j + min_row_len_B * 16]);
            }
            ++j;
        }
    }
}

__device__ __forceinline__ void coo_x_hyb_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, min_row_len_B = Bbits[0], B_coo_cnt = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;

    for (int i = lane_id; i < nzA; i += 32)
    {
        TileIndex idx = Abits[i + 1], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i];

        for (int j = min_row_len_B * Acol; j < min_row_len_B * (Acol + 1); ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            atomicAdd(Cvals + Arow * 16 + Bcol, Aval * Bvals[j]);
        }
    }

    for (int i = lane_id; i < nzA; i += 32)
    {
        TileIndex idx = Abits[i + 1], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i];
        int j = 0;
        for (; j < B_coo_cnt; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < B_coo_cnt)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            atomicAdd(Cvals + Arow * 16 + Bcol, Aval * Bvals[j + min_row_len_B * 16]);
            ++j;
        }
    }
}

__device__ __forceinline__ void coo_x_drw_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1, nzC = Cbits[0] + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

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
}

__device__ __forceinline__ void coo_x_drw_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

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
}

__device__ __forceinline__ void coo_x_drw_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        uint16_t row_map = get_row_map(Cbitmap, Arow);
        for (int j = Arow * max_row_len_C; j < (Arow + 1) * max_row_len_C; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + j);
            int idx = check_bitmap_row_idx(row_map, Ccol);
            if (idx != -1) atomicAdd(Cvals + j, Aval * Bvals[k + Ccol]);
        }
    }
}

__device__ __forceinline__ void coo_x_drw_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int min_row_len_C = Cbits[0], C_coo_cnt = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;
        
        uint16_t row_map = get_row_map(Cbitmap, Arow);
        for (int j = Arow * min_row_len_C; j < (Arow + 1) * min_row_len_C; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, j);
            int idx = check_bitmap_row_idx(row_map, Ccol);
            if (idx != -1) atomicAdd(Cvals + j, Aval * Bvals[k + Ccol]);
        }

        int j = 0;
        for (; j < C_coo_cnt; ++j) if (Arow == Cbits[Celllen + j] >> 4) break;

        while (j < C_coo_cnt && Arow == Cbits[Celllen + j] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + j] & 15;
            atomicAdd(Cvals + min_row_len_C * 16 + j, Aval * Bvals[k + Ccol]);
            ++j;
        }
    }
}

__device__ __forceinline__ void coo_x_drw_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        int cidx = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (cidx == -1) continue;
        cidx -= 1;
        cidx <<= 4;
        
        #pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            atomicAdd(Cvals + cidx + j, Aval * Bvals[k + j]);
        }
    }
}

__device__ __forceinline__ void coo_x_drw_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = 0; j < cntC; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + j);
            atomicAdd(Cvals + Arow + 16 * j, Aval * Bvals[k + Ccol]);
        }
    }
}

__device__ __forceinline__ void coo_x_drw_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, Bcnt = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        TileIndex idx = Abits[i + 1], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i];

        for (int j = 0; j < Bcnt; ++j)
        {
            TileIndex Brow = TileIndex_CSR_Get(Bbits, 1 + j);
            if (Brow == Acol)
                for (int k = 0; k < 16; ++k)
                    atomicAdd(Cvals + Arow * 16 + k, Aval * Bvals[j * 16 + k]);
        }
    }
}

__device__ __forceinline__ void coo_x_dcl_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1, nzC = Cbits[0] + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            int idx = binary_find_index<TileIndex>(Cbits + 1, nzC, Arow * 16 + Bcol);
            if (idx != -1) atomicAdd(Cvals + idx, Aval * Bvals[j * 16 + Acol]);
        }
    }
}

__device__ __forceinline__ void coo_x_dcl_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];
        uint16_t row_map = get_row_map(Cbitmap, Arow);

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Bbits, 1 + j);
            int idx = check_bitmap_row_idx(row_map, Ccol);
            if (idx != -1) atomicAdd(Cvals + Cbits[Arow] + idx, Aval * Bvals[j * 16 + Acol]);
        }
    }
}

__device__ __forceinline__ void coo_x_dcl_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];
        uint16_t row_map = get_row_map(Cbitmap, Arow);

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Bbits, 1 + j);
            int idx = check_bitmap_row_idx(row_map, Ccol);
            if (idx != -1) atomicAdd(Cvals + Arow * max_row_len_C + idx, Aval * Bvals[j * 16 + Acol]);
        }
    }
}

__device__ __forceinline__ void coo_x_dcl_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int min_row_len_C = Cbits[0], C_coo_cnt = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Bbits, 1 + j);
            int idx_ell = binary_find_index_4bit(Cbits + 2, Ccol, Arow * min_row_len_C, (Arow + 1) * min_row_len_C);
            int idx_coo = binary_find_index<TileIndex>(Cbits + Celllen, C_coo_cnt, Arow << 4 | Ccol);
            if (idx_ell != -1) atomicAdd(Cvals + idx_ell, Aval * Bvals[j * 16 + Acol]);
            if (idx_coo != -1) atomicAdd(Cvals + min_row_len_C * 16 + idx_coo, Aval * Bvals[j * 16 + Acol]);
        }
    }
}

__device__ __forceinline__ void coo_x_dcl_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        int k = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;        

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Bbits, 1 + j);
            atomicAdd(Cvals + k + Ccol, Aval * Bvals[j * 16 + Acol]);
        }
    }
}

__device__ __forceinline__ void coo_x_dcl_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    uint16_t col_map = 0;
    for (int k = 0; k < cntC; ++k) col_map |= 1 << TileIndex_CSR_Get(Cbits, 1 + k);

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Bbits, 1 + j);
            if ((1 << Ccol) & col_map) {
                int col_idx = __popc(col_map & ((1 << Ccol) - 1));
                atomicAdd(Cvals + Arow + 16 * col_idx, Aval * Bvals[j * 16 + Acol]);
            }
        }
    }
}

__device__ __forceinline__ void coo_x_dcl_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, Bcnt = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        TileIndex idx = Abits[i + 1], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i];

        for (int j = 0; j < Bcnt; ++j)
        {
            atomicAdd(Cvals + Arow * 16 + TileIndex_CSR_Get(Bbits, 1 + j), Aval * Bvals[j * 16 + Acol]);
        }
    }
}

__device__ __forceinline__ void coo_x_dns_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, nzC = Cbits[0] + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        int j = 0;
        for (; j < nzC; ++j) if (Arow == Cbits[1 + j] >> 4) break;
        while (j < nzC && Arow == Cbits[1 + j] >> 4)
        {
            TileIndex Ccol = Cbits[1 + j] & 15;
            atomicAdd(Cvals + j, Aval * Bvals[Acol * 16 + Ccol]);
            ++j;
        }
    }
}

__device__ __forceinline__ void coo_x_dns_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        for (int j = Cbits[Arow]; j < Cbits[Arow + 1]; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 17, j);
            atomicAdd(Cvals + j, Aval * Bvals[Acol * 16 + Ccol]);
        }
    }
}

__device__ __forceinline__ void coo_x_dns_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        for (int j = Arow * max_row_len_C; j < (Arow + 1) * max_row_len_C; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + j);
            atomicAdd(Cvals + j, Aval * Bvals[Acol * 16 + Ccol]);
        }
    }
}

__device__ __forceinline__ void coo_x_dns_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1;
    int min_row_len_C = Cbits[0], C_coo_cnt = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        for (int j = Arow * min_row_len_C; j < (Arow + 1) * min_row_len_C; ++j)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, j);
            atomicAdd(Cvals + j, Aval * Bvals[Acol * 16 + Ccol]);
        }

        int j = 0;
        for (; j < C_coo_cnt; ++j) if (Arow == Cbits[Celllen + j] >> 4) break;
        while (j < C_coo_cnt && Arow == Cbits[Celllen + j] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + j] & 15;
            atomicAdd(Cvals + min_row_len_C * 16 + j, Aval * Bvals[Acol * 16 + Ccol]);
            ++j;
        }
    }
}

__device__ __forceinline__ void coo_x_dns_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        int k = binary_find_index_4bit(Cbits, Arow, 1, cntC + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        #pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            atomicAdd(Cvals + k + j, Aval * Bvals[Acol * 16 + j]);
        }
    }
}

__device__ __forceinline__ void coo_x_dns_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        int Aidx = Abits[i + 1], Acol = Aidx & 15, Arow = Aidx >> 4;
        MatValue Aval = Avals[i];

        for (int j = 0; j < cntC; ++j)
        {
            atomicAdd(Cvals + Arow + 16 * j, Aval * Bvals[Acol * 16 + TileIndex_CSR_Get(Cbits, 1 + j)]);
        }
    }
}

__device__ __forceinline__ void coo_x_dns_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzA = Abits[0] + 1;

    for (int i = lane_id; i < nzA; i += 32)
    {
        TileIndex idx = Abits[i + 1], Arow = idx >> 4, Acol = idx & 15;
        MatValue Aval = Avals[i];

#pragma unroll
        for (int j = 0; j < 16; ++j)
        {
            atomicAdd(Cvals + Arow * 16 + j, Aval * Bvals[Acol * 16 + j]);
        }
    }
}