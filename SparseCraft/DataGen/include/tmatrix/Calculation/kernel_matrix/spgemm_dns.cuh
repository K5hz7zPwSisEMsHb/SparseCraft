#pragma once
#include <tmatrix/Calculation/kernel_matrix/common.cuh>

__device__ __forceinline__ void dns_x_coo_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzB = Bbits[0] + 1, nzC = Cbits[0] + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        int j = 0;
        for (; j < nzB; ++j)
            if (Acol == Bbits[1 + j] >> 4)
                break;
        while (Acol == Bbits[1 + j] >> 4 && j < nzB)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rc[Bcol] += ra * Bvals[j];
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < 16; ++i)
        {
            if (rc[i] != 0)
            {
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, vwarp_id << 4 | i);
                if (idx != -1) Cvals[idx] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dns_x_coo_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzB = Bbits[0] + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        int j = 0;
        for (; j < nzB; ++j)
            if (Acol == Bbits[1 + j] >> 4)
                break;
        while (Acol == Bbits[1 + j] >> 4 && j < nzB)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rc[Bcol] += ra * Bvals[j];
            ++j;
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = Cbits[vwarp_id]; i < Cbits[vwarp_id + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rc[col];
        }
    }
}

__device__ __forceinline__ void dns_x_coo_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzB = Bbits[0] + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        int j = 0;
        for (; j < nzB; ++j)
            if (Acol == Bbits[1 + j] >> 4)
                break;
        while (Acol == Bbits[1 + j] >> 4 && j < nzB)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rc[Bcol] += ra * Bvals[j];
            ++j;
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = vwarp_id * max_row_len_C; i < (vwarp_id + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[i] += rc[Ccol];
        }
    }
}

__device__ __forceinline__ void dns_x_coo_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzB = Bbits[0] + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        int j = 0;
        for (; j < nzB; ++j)
            if (Acol == Bbits[1 + j] >> 4)
                break;
        while (Acol == Bbits[1 + j] >> 4 && j < nzB)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rc[Bcol] += ra * Bvals[j];
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = vwarp_id * min_row_len_C; i < (vwarp_id + 1) * min_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, i);
            Cvals[i] += rc[Ccol];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (vwarp_id == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && vwarp_id == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rc[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void dns_x_coo_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzB = Bbits[0] + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        int j = 0;
        for (; j < nzB; ++j)
            if (Acol == Bbits[1 + j] >> 4)
                break;
        while (Acol == Bbits[1 + j] >> 4 && j < nzB)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rc[Bcol] += ra * Bvals[j];
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, vwarp_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dns_x_coo_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzB = Bbits[0] + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        int j = 0;
        for (; j < nzB; ++j)
            if (Acol == Bbits[1 + j] >> 4)
                break;
        while (Acol == Bbits[1 + j] >> 4 && j < nzB)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rc[Bcol] += ra * Bvals[j];
            ++j;
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rc[Ccol] += __shfl_down_sync(0xffffffff, rc[Ccol], 1);
        if (vlane_id == 0) Cvals[vwarp_id + i * 16] += rc[Ccol];
    }
}

__device__ __forceinline__ void dns_x_coo_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzB = Bbits[0] + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        int j = 0;
        for (; j < nzB; ++j)
            if (Acol == Bbits[1 + j] >> 4)
                break;
        while (Acol == Bbits[1 + j] >> 4 && j < nzB)
        {
            TileIndex Bcol = Bbits[1 + j] & 15;
            rc[Bcol] += ra * Bvals[j];
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
        if (vlane_id == 0)
            Cvals[vwarp_id * 16 + i] += rc[i];
    }
}

__device__ __forceinline__ void dns_x_csr_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzC = Cbits[0] + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rc[Bcol] += ra * Bvals[j];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < 16; ++i)
        {
            if (rc[i] != 0)
            {
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, vwarp_id << 4 | i);
                if (idx != -1) Cvals[idx] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dns_x_csr_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rc[Bcol] += ra * Bvals[j];
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = Cbits[vwarp_id]; i < Cbits[vwarp_id + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rc[col];
        }
    }
}

__device__ __forceinline__ void dns_x_csr_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rc[Bcol] += ra * Bvals[j];
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = vwarp_id * max_row_len_C; i < (vwarp_id + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[i] += rc[Ccol];
        }
    }
}

__device__ __forceinline__ void dns_x_csr_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rc[Bcol] += ra * Bvals[j];
        }
    }
    
    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = vwarp_id * min_row_len_C; i < (vwarp_id + 1) * min_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, i);
            Cvals[i] += rc[Ccol];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (vwarp_id == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && vwarp_id == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rc[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void dns_x_csr_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rc[Bcol] += ra * Bvals[j];
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, vwarp_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dns_x_csr_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rc[Bcol] += ra * Bvals[j];
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rc[Ccol] += __shfl_down_sync(0xffffffff, rc[Ccol], 1);
        if (vlane_id == 0) Cvals[vwarp_id + i * 16] += rc[Ccol];
    }
}

__device__ __forceinline__ void dns_x_csr_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Bbits[Acol]; j < Bbits[Acol + 1]; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 34 + j);
            rc[Bcol] += ra * Bvals[j];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
        if (vlane_id == 0)
            Cvals[vwarp_id * 16 + i] += rc[i];
    }
}

__device__ __forceinline__ void dns_x_ell_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1, nzC = Cbits[0] + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rc[Bcol] += ra * Bvals[j];
        }
    }
    
    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < 16; ++i)
        {
            if (rc[i] != 0)
            {
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, vwarp_id << 4 | i);
                if (idx != -1) Cvals[idx] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dns_x_ell_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rc[Bcol] += ra * Bvals[j];
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = Cbits[vwarp_id]; i < Cbits[vwarp_id + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rc[col];
        }
    }
}

__device__ __forceinline__ void dns_x_ell_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rc[Bcol] += ra * Bvals[j];
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = vwarp_id * max_row_len_C; i < (vwarp_id + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[i] += rc[Ccol];
        }
    }
}

__device__ __forceinline__ void dns_x_ell_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rc[Bcol] += ra * Bvals[j];
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = vwarp_id * min_row_len_C; i < (vwarp_id + 1) * min_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, i);
            Cvals[i] += rc[Ccol];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (vwarp_id == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && vwarp_id == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rc[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void dns_x_ell_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rc[Bcol] += ra * Bvals[j];
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, vwarp_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dns_x_ell_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rc[Bcol] += ra * Bvals[j];
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rc[Ccol] += __shfl_down_sync(0xffffffff, rc[Ccol], 1);
        if (vlane_id == 0) Cvals[vwarp_id + i * 16] += rc[Ccol];
    }
}

__device__ __forceinline__ void dns_x_ell_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rc[16] = {0};
    int max_row_len_B = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Acol * max_row_len_B; j < (Acol + 1) * max_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            rc[Bcol] += ra * Bvals[j];
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
        if (vlane_id == 0)
            Cvals[vwarp_id * 16 + i] += rc[i];
    }
}

__device__ __forceinline__ void dns_x_hyb_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2, nzC = Cbits[0] + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, j);
            rc[Bcol] += ra * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rc[Bcol] += ra * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < 16; ++i)
        {
            if (rc[i] != 0)
            {
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, vwarp_id << 4 | i);
                if (idx != -1) Cvals[idx] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dns_x_hyb_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, j);
            rc[Bcol] += ra * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rc[Bcol] += ra * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = Cbits[vwarp_id]; i < Cbits[vwarp_id + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rc[col];
        }
    }
}

__device__ __forceinline__ void dns_x_hyb_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, j);
            rc[Bcol] += ra * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rc[Bcol] += ra * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = vwarp_id * max_row_len_C; i < (vwarp_id + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            if (i > vwarp_id * max_row_len_C && Ccol == 0) break;
            Cvals[i] += rc[Ccol];
        }
    }
}

__device__ __forceinline__ void dns_x_hyb_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, j);
            rc[Bcol] += ra * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rc[Bcol] += ra * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = vwarp_id * min_row_len_C; i < (vwarp_id + 1) * min_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, i);
            Cvals[i] += rc[Ccol];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (vwarp_id == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && vwarp_id == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rc[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void dns_x_hyb_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, j);
            rc[Bcol] += ra * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rc[Bcol] += ra * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, vwarp_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dns_x_hyb_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits + 2, j);
            rc[Bcol] += ra * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rc[Bcol] += ra * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rc[Ccol] += __shfl_down_sync(0xffffffff, rc[Ccol], 1);
        if (vlane_id == 0) Cvals[vwarp_id + i * 16] += rc[Ccol];
    }
}

__device__ __forceinline__ void dns_x_hyb_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rc[16] = {0};
    int min_row_len_B = Bbits[0], coo_cnt_B = Bbits[1], Belllen = (min_row_len_B * 16 + 1) / 2 + 2;

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        for (int j = Acol * min_row_len_B; j < (Acol + 1) * min_row_len_B; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 4 + j);
            rc[Bcol] += ra * Bvals[j];
        }

        int j = 0;
        for (; j < coo_cnt_B; ++j)
            if (Acol == Bbits[Belllen + j] >> 4)
                break;
        while (Acol == Bbits[Belllen + j] >> 4 && j < coo_cnt_B)
        {
            TileIndex Bcol = Bbits[Belllen + j] & 15;
            rc[Bcol] += ra * Bvals[j + min_row_len_B * 16];
            ++j;
        }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
        if (vlane_id == 0)
            Cvals[vwarp_id * 16 + i] += rc[i];
    }
}

__device__ __forceinline__ void dns_x_drw_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rc[16] = {0};
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, nzC = Cbits[0] + 1;

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        #pragma unroll
        for (int j = 0; j < 16; ++j)
            rc[j] += ra * Bvals[k + j];
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = 0; i < 16; ++i)
        {
            if (rc[i] != 0)
            {
                MatIndex idx = binary_find_index<TileIndex>(Cbits + 1, nzC, vwarp_id << 4 | i);
                if (idx != -1) Cvals[idx] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dns_x_drw_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        #pragma unroll
        for (int j = 0; j < 16; ++j)
            rc[j] += ra * Bvals[k + j];
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = Cbits[vwarp_id]; i < Cbits[vwarp_id + 1]; ++i)
        {
            int col = TileIndex_CSR_Get(Cbits + 17, i);
            Cvals[i] += rc[col];
        }
    }
}

__device__ __forceinline__ void dns_x_drw_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        #pragma unroll
        for (int j = 0; j < 16; ++j)
            rc[j] += ra * Bvals[k + j];
    }

    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = vwarp_id * max_row_len_C; i < (vwarp_id + 1) * max_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
            Cvals[i] += rc[Ccol];
        }
    }
}

__device__ __forceinline__ void dns_x_drw_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        #pragma unroll
        for (int j = 0; j < 16; ++j)
            rc[j] += ra * Bvals[k + j];
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        for (int i = vwarp_id * min_row_len_C; i < (vwarp_id + 1) * min_row_len_C; ++i)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, i);
            Cvals[i] += rc[Ccol];
        }
        int i = 0;
        for (; i < coo_cnt_C; ++i)
            if (vwarp_id == Cbits[Celllen + i] >> 4)
                break;
        while (i < coo_cnt_C && vwarp_id == Cbits[Celllen + i] >> 4)
        {
            TileIndex Ccol = Cbits[Celllen + i] & 15;
            Cvals[i + min_row_len_C * 16] += rc[Ccol];
            ++i;
        }
    }
}

__device__ __forceinline__ void dns_x_drw_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        #pragma unroll
        for (int j = 0; j < 16; ++j)
            rc[j] += ra * Bvals[k + j];
    }

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
    }

    if (vlane_id == 0)
    {
        int idx = binary_find_index_4bit(Cbits, vwarp_id, 1, cntC + 1);
        if (idx != -1) {
            idx -= 1;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                Cvals[idx * 16 + i] += rc[i];
            }
        }
    }
}

__device__ __forceinline__ void dns_x_drw_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    MatValue rc[16] = {0};

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        #pragma unroll
        for (int j = 0; j < 16; ++j)
            rc[j] += ra * Bvals[k + j];
    }

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        rc[Ccol] += __shfl_down_sync(0xffffffff, rc[Ccol], 1);
        if (vlane_id == 0) Cvals[vwarp_id + i * 16] += rc[Ccol];
    }
}

__device__ __forceinline__ void dns_x_drw_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    MatValue rc[16] = {0};
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = vlane_id; i < 16; i += 2)
    {
        MatValue ra = Avals[vwarp_id * 16 + i];
        TileIndex Acol = i;

        int k = binary_find_index_4bit(Bbits, Acol, 1, cntB + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        #pragma unroll
        for (int j = 0; j < 16; ++j)
            rc[j] += ra * Bvals[k + j];
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
        if (vlane_id == 0)
            Cvals[vwarp_id * 16 + i] += rc[i];
    }
}

__device__ __forceinline__ void dns_x_dcl_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, nzC = Cbits[0] + 1;

    for (int i = hwarp_id; i < 16; i += 2)
    {
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            MatValue sum = Aval * Bvals[j * 16 + hlane_id];
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 1);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 2);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 4);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 8);
            if (hlane_id == 0) {
                int idx = binary_find_index<TileIndex>(Cbits + 1, nzC, i << 4 | Bcol);
                if (idx != -1) Cvals[idx] += sum;
            }
        }
    }
}

__device__ __forceinline__ void dns_x_dcl_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = hwarp_id; i < 16; i += 2)
    {
        MatValue Aval = Avals[i * 16 + hlane_id];
        uint16_t row_map = get_row_map(Cbitmap, i);

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            MatValue sum = Aval * Bvals[j * 16 + hlane_id];
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 1);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 2);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 4);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 8);
            if (hlane_id == 0) {
                int idx = check_bitmap_row_idx(row_map, Bcol);
                if (idx != -1) Cvals[Cbits[i] + idx] += sum;
            }
        }
    }
}

__device__ __forceinline__ void dns_x_dcl_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < 16; i += 2)
    {
        MatValue Aval = Avals[i * 16 + hlane_id];
        uint16_t row_map = get_row_map(Cbitmap, i);

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            MatValue sum = Aval * Bvals[j * 16 + hlane_id];
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 1);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 2);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 4);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 8);
            if (hlane_id == 0) {
                int idx = check_bitmap_row_idx(row_map, Bcol);
                if (idx != -1) Cvals[i * max_row_len_C + idx] += sum;
            }
        }
    }
}

__device__ __forceinline__ void dns_x_dcl_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = hwarp_id; i < 16; i += 2)
    {
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            MatValue sum = Aval * Bvals[j * 16 + hlane_id];
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 1);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 2);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 4);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 8);
            if (hlane_id == 0 && sum != 0) {
                int idx_ell = binary_find_index_4bit(Cbits + 2, Bcol, i * min_row_len_C, (i + 1) * min_row_len_C);
                int idx_coo = binary_find_index<TileIndex>(Cbits + Celllen, coo_cnt_C, i << 4 | Bcol);
                if (idx_ell != -1) Cvals[idx_ell] += sum;
                if (idx_coo != -1) Cvals[idx_coo + min_row_len_C * 16] += sum;
            }
        }
    }
}

__device__ __forceinline__ void dns_x_dcl_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = hwarp_id; i < 16; i += 2)
    {
        MatValue Aval = Avals[i * 16 + hlane_id];
        
        int k = binary_find_index_4bit(Cbits, i, 1, cntC + 1);
        if (k == -1) continue;
        k -= 1;
        k <<= 4;

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            MatValue sum = Aval * Bvals[j * 16 + hlane_id];
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 1);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 2);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 4);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 8);
            if (hlane_id == 0) Cvals[k + Bcol] += sum;
        }
    }
}

__device__ __forceinline__ void dns_x_dcl_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1, cntC = TileIndex_CSR_Get(Cbits, 0) + 1;
    uint16_t col_map = 0;
    for (int i = 0; i < cntC; ++i)
    {
        col_map |= 1 << TileIndex_CSR_Get(Cbits, 1 + i);
    }

    for (int i = hwarp_id; i < 16; i += 2)
    {
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            uint16_t Bcol_mask = 1 << Bcol;
            if (Bcol_mask & col_map) {
                MatValue sum = Aval * Bvals[j * 16 + hlane_id];
                sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 1);
                sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 2);
                sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 4);
                sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 8);
                if (hlane_id == 0) {
                    MatIndex idx = __popc(col_map & (Bcol_mask - 1));
                    Cvals[idx * 16 + i] += sum;
                }
            }
        }
    }
}

__device__ __forceinline__ void dns_x_dcl_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntB = TileIndex_CSR_Get(Bbits, 0) + 1;

    for (int i = hwarp_id; i < 16; i += 2)
    {
        MatValue Aval = Avals[i * 16 + hlane_id];

        for (int j = 0; j < cntB; ++j)
        {
            TileIndex Bcol = TileIndex_CSR_Get(Bbits, 1 + j);
            MatValue rc = Aval * Bvals[j * 16 + hlane_id];
            rc += __shfl_down_sync(0xffffffff, rc, 1);
            rc += __shfl_down_sync(0xffffffff, rc, 2);
            rc += __shfl_down_sync(0xffffffff, rc, 4);
            rc += __shfl_down_sync(0xffffffff, rc, 8);
            Cvals[i * 16 + Bcol] += rc;
        }
    }
}

__device__ __forceinline__ void dns_x_dns_2_coo(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int nzC = Cbits[0] + 1;

    for (int i = hwarp_id; i < nzC; i += 2)
    {
        TileIndex Cidx = Cbits[1 + i], Crow = Cidx >> 4, Ccol = Cidx & 15;
        MatValue sum = Avals[Crow * 16 + hlane_id] * Bvals[Ccol + 16 * hlane_id];
        sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 1);
        sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 2);
        sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 4);
        sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 8);
        if (hlane_id == 0) Cvals[i] += sum;
    }
}

__device__ __forceinline__ void dns_x_dns_2_csr(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    for (int i = 0; i < 16; ++i)
    {
        for (int j = Cbits[i] + hwarp_id; j < Cbits[i + 1]; j += 2)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 17, j);
            MatValue sum = Avals[i * 16 + hlane_id] * Bvals[Ccol + 16 * hlane_id];
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 1);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 2);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 4);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 8);
            if (hlane_id == 0) Cvals[j] += sum;
        }
    }
}

__device__ __forceinline__ void dns_x_dns_2_ell(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int max_row_len_C = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = 0; i < 16; ++i)
    {
        for (int j = i * max_row_len_C + hwarp_id; j < (i + 1) * max_row_len_C; j += 2)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + j);
            if (j > i * max_row_len_C && Ccol == 0) break;
            MatValue sum = Avals[i * 16 + hlane_id] * Bvals[Ccol + 16 * hlane_id];
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 1);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 2);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 4);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 8);
            if (hlane_id == 0) Cvals[j] += sum;
        }
    }
}

__device__ __forceinline__ void dns_x_dns_2_hyb(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int min_row_len_C = Cbits[0], coo_cnt_C = Cbits[1], Celllen = (min_row_len_C * 16 + 1) / 2 + 2;

    for (int i = 0; i < 16; ++i)
    {
        for (int j = i * min_row_len_C + hwarp_id; j < (i + 1) * min_row_len_C; j += 2)
        {
            TileIndex Ccol = TileIndex_CSR_Get(Cbits + 2, j);
            MatValue sum = Avals[i * 16 + hlane_id] * Bvals[Ccol + 16 * hlane_id];
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 1);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 2);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 4);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 8);
            if (hlane_id == 0) Cvals[j] += sum;
        }
    }

    for (int i = hwarp_id; i < coo_cnt_C; i += 2)
    {
        TileIndex Cidx = Cbits[Celllen + i], Crow = Cidx >> 4, Ccol = Cidx & 15;
        MatValue sum = Avals[Crow * 16 + hlane_id] * Bvals[Ccol + 16 * hlane_id];
        sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 1);
        sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 2);
        sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 4);
        sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 8);
        if (hlane_id == 0) Cvals[i + min_row_len_C * 16] += sum;
    }
}

__device__ __forceinline__ void dns_x_dns_2_drw(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Crow = TileIndex_CSR_Get(Cbits, 1 + i);
        for (int j = hwarp_id; j < 16; j += 2)
        {
            MatValue sum = Avals[Crow * 16 + hlane_id] * Bvals[j + 16 * hlane_id];
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 1);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 2);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 4);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 8);
            if (hlane_id == 0) Cvals[i * 16 + j] += sum;
        }
    }
}

__device__ __forceinline__ void dns_x_dns_2_dcl(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t*Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int cntC = TileIndex_CSR_Get(Cbits, 0) + 1;

    for (int i = 0; i < cntC; ++i)
    {
        TileIndex Ccol = TileIndex_CSR_Get(Cbits, 1 + i);
        for (int j = hwarp_id; j < 16; j += 2)
        {
            MatValue sum = Avals[j * 16 + hlane_id] * Bvals[Ccol + 16 * hlane_id];
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 1);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 2);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 4);
            sum += __shfl_down_sync(hwarp_id? 0xffff0000: 0x0000ffff, sum, 8);
            if (hlane_id == 0) Cvals[i * 16 + j] += sum;
        }
    }
}

__device__ __forceinline__ void dns_x_dns_2_dns(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag)
{
    int dnscol = vlane_id;
    int start = vwarp_id * 16, stop = (vwarp_id + 1) * 16;
    MatValue rc[16] = {0};

    for (int i = start + vlane_id; i < stop; i += 2)
    {
        MatValue val = Avals[i];
        for (int l = 0; l < 16; l++)
        {
            rc[l] += val * Bvals[dnscol * 16 + l];
        }
        dnscol = dnscol + 2;
    }

#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        rc[i] += __shfl_down_sync(0xffffffff, rc[i], 1);
        if (vlane_id == 0)
            Cvals[start + i] += rc[i];
    }
}