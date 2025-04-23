#include <tmatrix/Calculation/spmm.cuh>
#include <tmatrix/DataStructure/CSR2Tile.cuh>
#include <tmatrix/DataStructure/hd.cuh>
#include <tmatrix/Calculation/load_balance.h>
#include <tmatrix/Utils/timer.h>
#include <tmatrix/Utils/msg.h>

#include <tmatrix/MMIO/mmio_highlevel.h>

#include <cusparse.h>

__global__ void stir_spmm_cuda_kernel_v1(
    int rbnum, int cbnum, int rowA, int colA, int ldb, int ldc, MatIndex rowblks, MatIndex *block_row_ptr, MatIndex *block_col_start, MatIndex *block_col_end, MatIndex *A_rowptr, MatIndex *A_colind, Tile *A_tiles, char *A_data, MatValue *db, MatValue *dc
)
{
    const int thread_id = threadIdx.x;
    const int lwarp_id = thread_id >> 5;
    int dimN_index = blockIdx.y * 64;
    int blki_blc = blockIdx.x;
    const int lane_id = thread_id & 31;

    int tile_start = lwarp_id * 32;

    __shared__ MatValue s_dense_tile[16 * 64];
    __shared__ MatValue s_out_tile[16 * 64];
    MatValue rc[32] = {0};
    __shared__ MatValue s_value_tile[16 * 16];
    // __shared__ TileIndex s_index_tile[16 * 16];

    __shared__ int s_colidx[PREFETCH_SMEM_TH_SPMM];
    __shared__ uint8_t s_valoff[PREFETCH_SMEM_TH_SPMM];
    __shared__ uint64_t s_bitoff[PREFETCH_SMEM_TH_SPMM];
    __shared__ TileFormat s_format[PREFETCH_SMEM_TH_SPMM];

    if (blki_blc >= rowblks)
        return;
    
    MatIndex row16 = block_row_ptr[blki_blc];
    MatIndex signbit = row16 & 0x80000000;
    row16 ^= signbit;

    MatIndex start_Aj = signbit ? block_col_start[blki_blc] : A_rowptr[row16], end_Aj = signbit ? block_col_end[blki_blc] : A_rowptr[row16 + 1];

    if (thread_id < end_Aj - start_Aj) {
        s_colidx[thread_id] = A_colind[start_Aj + thread_id] << 4;
        s_valoff[thread_id] = A_tiles[start_Aj + thread_id].bitslen;
        s_bitoff[thread_id] = A_tiles[start_Aj + thread_id].bits_off;
        s_format[thread_id] = A_tiles[start_Aj + thread_id].fmt;
    }

    for (int i = thread_id; i < 16 * 64; i += blockDim.x)
    {
        s_out_tile[i] = 0;
    }
    __syncthreads();

    for (int Aj = start_Aj; Aj < end_Aj; ++Aj)
    {
        MatIndex off16 = s_colidx[Aj - start_Aj];
        TileFormat fmt = s_format[Aj - start_Aj];
        MatValue *A_val = (MatValue *)(A_data + s_bitoff[Aj - start_Aj] + s_valoff[Aj - start_Aj] * 16);
        TileIndex *bits = (TileIndex *)(A_data + s_bitoff[Aj - start_Aj]);

        #pragma unroll
        for (int i = 0; i < 16; ++i) s_dense_tile[i * 64 + thread_id] = db[(off16 + i) * ldb + dimN_index + thread_id];
        __syncthreads();

        switch (fmt)
        {
        case COO:
            {
                MatIndex nnz = bits[0] + 1;

                for (int i = 0; i < nnz; ++i)
                {
                    MatIndex idx = bits[i + 1], row = idx >> 4, col = idx & 15;
                    MatValue ra = A_val[i];
                    s_out_tile[row * 64 + tile_start + lane_id] += ra * s_dense_tile[col * 64 + tile_start + lane_id];
                }
            }
            break;
        case CSR:
            {
                TileIndex *colidx = bits + 17;
                int vwarp_id = lane_id & 15, vlane_id = lane_id >> 4;
                int start = bits[vwarp_id], stop = bits[vwarp_id + 1];

                for (int j = start + vlane_id; j < stop; j += 2)
                {
                    TileIndex col = TileIndex_CSR_Get(colidx, j);
                    MatValue val = A_val[j];
                    
                    #pragma unroll
                    for (int l = 0; l < 32; ++l) rc[l] += val * s_dense_tile[col * 64 + tile_start + l];
                }
            }
            break;
        case ELL:
            {
                int elllen = (TileIndex_CSR_Get(bits, 0) + 1);
                int vwarp_id = lane_id & 15, vlane_id = lane_id >> 4;
                int start = vwarp_id * elllen, stop = (vwarp_id + 1) * elllen;

                for (int j = start + vlane_id; j < stop; j += 2)
                {
                    TileIndex col = TileIndex_CSR_Get(bits, 1 + j);
                    MatValue val = A_val[j];

                    #pragma unroll
                    for (int l = 0; l < 32; ++l) 
                        rc[l] += val * s_dense_tile[col * 64 + tile_start + l];
                }
            }
            break;
        case HYB:
            {
                int min_row_len = bits[0], coo_cnt = bits[1], elllen = min_row_len * 16;
                int coo_start = (elllen + 1) / 2 + 2;
                int vwarp_id = lane_id & 15, vlane_id = lane_id >> 4;
                int start = vwarp_id * min_row_len, stop = (vwarp_id + 1) * min_row_len;

                for (int j = start + vlane_id; j < stop; j += 2)
                {
                    TileIndex col = TileIndex_CSR_Get(bits, 4 + j);
                    MatValue val = A_val[j];

                    #pragma unroll
                    for (int l = 0; l < 32; ++l) rc[l] += val * s_dense_tile[col * 64 + tile_start + l];
                }

                for (int j = 0; j < coo_cnt; ++j)
                {
                    TileIndex idx = bits[coo_start + j], row = idx >> 4, col = idx & 0xf;
                    MatValue ra = A_val[j + elllen];
                    s_out_tile[row * 64 + tile_start + lane_id] += ra * s_dense_tile[col * 64 + tile_start + lane_id];
                }
            }
            break;
        case DRW:
            {
                int cnt = TileIndex_CSR_Get(bits, 0) + 1, nnz = cnt * 16;
                for (int i = thread_id; i < nnz; i += blockDim.x)
                {
                    s_value_tile[i] = A_val[i];
                }
                __syncthreads();

                for (int i = 0; i < cnt; ++i)
                {
                    TileIndex row = TileIndex_CSR_Get(bits, 1 + i);
                    MatValue res = 0;

                    #pragma unroll
                    for (int j = 0; j < 16; ++j)
                    {
                        MatValue ra = s_value_tile[i * 16 + j];
                        MatValue rb = s_dense_tile[j * 64 + tile_start + lane_id];
                        res += ra * rb;
                    }

                    s_out_tile[row * 64 + tile_start + lane_id] += res;
                }
            }
            break;
        case DCL:
            {
                int cnt = TileIndex_CSR_Get(bits, 0) + 1, nnz = cnt * 16;

                for (int glid = lane_id; glid < nnz; glid += 32)
                {
                    int rj = glid >> 4, ri = (rj << 4) | (glid & 15);
                    rj = TileIndex_CSR_Get(bits, 1 + rj);
                    MatValue ra = A_val[ri];

                    #pragma unroll
                    for (int l = 0; l < 32; ++l) rc[l] += ra * s_dense_tile[rj * 64 + tile_start + l];
                }
            }
            break;
        case DNS:
            {
                for (int i = thread_id; i < 256; i += blockDim.x)
                {
                    s_value_tile[i] = A_val[i];
                }
                __syncthreads();

#ifndef PLATFORM
                int vwarp_id = lane_id & 15, vlane_id = lane_id >> 4;
                int start = vwarp_id * 16, stop = (vwarp_id + 1) * 16;

                int dnscol = lane_id >> 4;
                for (int i = start + vlane_id; i < stop; i += 2)
                {
                    MatValue val = s_value_tile[i];
                    #pragma unroll
                    for (int l = 0; l < 32; l++)
                    {
                        rc[l] += val * s_dense_tile[dnscol * 64 + tile_start + l];
                    }
                    dnscol = dnscol + 2;
                }
#elif PLATFORM == 80 // A100
                // *perform A (16 * 16) * B (16 * 32) = C (16 * 32) using tensor core | 884
                int a_offset = (lane_id >> 2) * 16 + (lane_id & 3);
                int b_offset = (lane_id >> 3) * 64 + (lane_id & 7);
                for (int m = 0; m < 16; m += 8)
                {
                    for (int n = 0; n < 32; n += 8)
                    {
                        for (int k = 0; k < 16; k += 4)
                        {
                            MatValue a = s_value_tile[m * 16 + k + a_offset];
                            MatValue b = s_dense_tile[k * 64 + tile_start + n + b_offset];
                        }
                    }
                }
#endif
            }
            
            break;
        default:
            break;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int l = 0; l < 32; l++)
    {
        int ri = lane_id & 15u;
        rc[l] += __shfl_down_sync(0xFFFFFFFF, rc[l], 16);
        if (lane_id < 16)
            s_out_tile[ri * 64 + l + tile_start] += rc[l];
    }
    
    if (signbit)
    {
        #pragma unroll
        for (int i = 0; i < 16; i++)
        {
            atomicAdd(&dc[(row16 * 16 + i) * ldb + dimN_index + thread_id], 
                    s_out_tile[i * 64 + thread_id]);
        }
    }
    else
    {
        #pragma unroll
        for (int i = 0; i < 16; i++)
        {
            dc[(row16 * 16 + i) * ldb + dimN_index + thread_id] = 
                s_out_tile[i * 64 + thread_id];
        }
    }
}

template<int SN, int TN, int STN, int SHAREB>
__global__ void stir_spmm_cuda_kernel_v2(
    int tilen_offset, int rbnum, int cbnum, int rowA, int colA, int ldb, int ldc, MatIndex rowblks, MatIndex *block_row_ptr, MatIndex *block_col_start, MatIndex *block_col_end, MatIndex *A_rowptr, MatIndex *A_colind, Tile *A_tiles, char *A_data, MatValue *db, MatValue *dc
)
{
    const int thread_id = threadIdx.x;
    const int lwarp_id = thread_id >> 5;
    int dimN_index = tilen_offset;
    int blki_blc = blockIdx.x;
    const int lane_id = thread_id & 31;

    int tile_start = lwarp_id * TN;

    __shared__ MatValue s_dense_tile[16 * SHAREB];
    __shared__ MatValue s_out_tile[16 * SHAREB];
    MatValue rc[TN] = {0};
    __shared__ MatValue s_value_tile[16 * 16];
    // __shared__ TileIndex s_index_tile[16 * 16];

    __shared__ int s_colidx[PREFETCH_SMEM_TH_SPMM];
    __shared__ uint8_t s_valoff[PREFETCH_SMEM_TH_SPMM];
    __shared__ uint64_t s_bitoff[PREFETCH_SMEM_TH_SPMM];
    __shared__ TileFormat s_format[PREFETCH_SMEM_TH_SPMM];

    if (blki_blc >= rowblks)
        return;
    
    MatIndex row16 = block_row_ptr[blki_blc];
    MatIndex signbit = row16 & 0x80000000;
    row16 ^= signbit;

    MatIndex start_Aj = signbit ? block_col_start[blki_blc] : A_rowptr[row16], end_Aj = signbit ? block_col_end[blki_blc] : A_rowptr[row16 + 1];

    if (thread_id < end_Aj - start_Aj) {
        s_colidx[thread_id] = A_colind[start_Aj + thread_id] << 4;
        s_valoff[thread_id] = A_tiles[start_Aj + thread_id].bitslen;
        s_bitoff[thread_id] = A_tiles[start_Aj + thread_id].bits_off;
        s_format[thread_id] = A_tiles[start_Aj + thread_id].fmt;
    }

    if (thread_id < STN)
    {
        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            s_out_tile[i * SHAREB + thread_id] = 0;
        }
    }
    if (thread_id < 16)
    {
        s_dense_tile[lane_id * SHAREB + SHAREB] = 0.0;
    }
    __syncthreads();

    for (int Aj = start_Aj; Aj < end_Aj; ++Aj)
    {
        MatIndex off16 = s_colidx[Aj - start_Aj];
        TileFormat fmt = s_format[Aj - start_Aj];
        MatValue *A_val = (MatValue *)(A_data + s_bitoff[Aj - start_Aj] + s_valoff[Aj - start_Aj] * 16);
        TileIndex *bits = (TileIndex *)(A_data + s_bitoff[Aj - start_Aj]);

        if (thread_id < STN) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) 
            s_dense_tile[i * SHAREB + thread_id] = db[(off16 + i) * ldb + dimN_index + thread_id];
        }
        __syncthreads();

        switch (fmt)
        {
        case COO:
            {
                MatIndex nnz = bits[0] + 1;

                for (int i = 0; i < nnz; ++i)
                {
                    MatIndex idx = bits[i + 1], row = idx >> 4, col = idx & 15;
                    MatValue ra = A_val[i];
                    if (lane_id < TN)
                        s_out_tile[row * SHAREB + tile_start + lane_id] += ra * s_dense_tile[col * SHAREB + tile_start + lane_id];
                }
            }
            break;
        case CSR:
            {
                TileIndex *colidx = bits + 17;
                int vwarp_id = lane_id & 15, vlane_id = lane_id >> 4;
                int start = bits[vwarp_id], stop = bits[vwarp_id + 1];

                for (int j = start + vlane_id; j < stop; j += 2)
                {
                    TileIndex col = TileIndex_CSR_Get(colidx, j);
                    MatValue val = A_val[j];
                    
                    #pragma unroll
                    for (int l = 0; l < TN; ++l) rc[l] += val * s_dense_tile[col * SHAREB + tile_start + l];
                }
            }
            break;
        case ELL:
            {
                int elllen = (TileIndex_CSR_Get(bits, 0) + 1);
                int vwarp_id = lane_id & 15, vlane_id = lane_id >> 4;
                int start = vwarp_id * elllen, stop = (vwarp_id + 1) * elllen;

                for (int j = start + vlane_id; j < stop; j += 2)
                {
                    TileIndex col = TileIndex_CSR_Get(bits, 1 + j);
                    MatValue val = A_val[j];
                    
                    #pragma unroll
                    for (int l = 0; l < TN; ++l) rc[l] += val * s_dense_tile[col * SHAREB + tile_start + l];
                }
            }
            break;
        case HYB:
            {
                int min_row_len = bits[0], coo_cnt = bits[1], elllen = min_row_len * 16;
                int coo_start = (elllen + 1) / 2 + 2;
                int vwarp_id = lane_id & 15, vlane_id = lane_id >> 4;
                int start = vwarp_id * min_row_len, stop = (vwarp_id + 1) * min_row_len;

                for (int j = start + vlane_id; j < stop; j += 2)
                {
                    TileIndex col = TileIndex_CSR_Get(bits, 4 + j);
                    MatValue val = A_val[j];

                    #pragma unroll
                    for (int l = 0; l < TN; ++l) rc[l] += val * s_dense_tile[col * SHAREB + tile_start + l];
                }

                for (int i = 0; i < coo_cnt; ++i)
                {
                    MatIndex idx = bits[coo_start + i], row = idx >> 4, col = idx & 15;
                    MatValue ra = A_val[i + elllen];
                    if (lane_id < TN)
                        s_out_tile[row * SHAREB + tile_start + lane_id] += ra * s_dense_tile[col * SHAREB + tile_start + lane_id];
                }
            }
            break;
        case DRW:
            {
                int cnt = TileIndex_CSR_Get(bits, 0) + 1, nnz = cnt * 16;
                for (int i = thread_id; i < nnz; i += blockDim.x)
                {
                    s_value_tile[i] = A_val[i];
                }
                __syncthreads();

                if (lane_id < TN)
                for (int i = 0; i < cnt; ++i)
                {
                    TileIndex row = TileIndex_CSR_Get(bits, 1 + i);
                    MatValue res = 0;

                    #pragma unroll
                    for (int j = 0; j < 16; ++j)
                    {
                        MatValue ra = s_value_tile[i * 16 + j];
                        MatValue rb = s_dense_tile[j * SHAREB + tile_start + lane_id];
                        res += ra * rb;
                    }

                    s_out_tile[row * SHAREB + tile_start + lane_id] += res;
                }
            }
            break;
        case DCL:
            {
                int cnt = TileIndex_CSR_Get(bits, 0) + 1, nnz = cnt * 16;

                for (int glid = lane_id; glid < nnz; glid += 32)
                {
                    int rj = glid >> 4, ri = (rj << 4) | (glid & 15);
                    rj = TileIndex_CSR_Get(bits, 1 + rj);
                    MatValue ra = A_val[ri];

                    #pragma unroll
                    for (int l = 0; l < TN; ++l) rc[l] += ra * s_dense_tile[rj * SHAREB + tile_start + l];
                }
            }
            break;
        case DNS:
            {
                // SN TN STN SHAREB
                // 16 17  33 34
                // for (int i = thread_id; i < 256; i += blockDim.x)
                // {
                //     s_value_tile[i] = A_val[i];
                // }
                // __syncthreads();

                int vwarp_id = lane_id & 15, vlane_id = lane_id >> 4;
                int start = vwarp_id * 16, stop = (vwarp_id + 1) * 16;

                int dnscol = lane_id >> 4;
                for (int i = start + vlane_id; i < stop; i += 2)
                {
                    MatValue val = A_val[i];
                    for (int l = 0; l < TN; l++)
                    {
                        rc[l] += val * s_dense_tile[dnscol * SHAREB + tile_start + l];
                    }
                    dnscol = dnscol + 2;
                }
            }
            break;
        default:
            break;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int l = 0; l < TN; l++)
    {
        rc[l] += __shfl_down_sync(0xFFFFFFFF, rc[l], 16);
        if (lane_id < 16)
            s_out_tile[lane_id * SHAREB + l + tile_start] += rc[l];
    }
    __syncthreads();
    
    if (signbit)
    {
        #pragma unroll
        for (int i = 0; i < 16; i++)
        {
            if (thread_id < STN) 
                atomicAdd(&dc[(row16 * 16 + i) * ldb + dimN_index + thread_id], s_out_tile[i * SHAREB + thread_id]);
        }
    }
    else
    {
        #pragma unroll
        for (int i = 0; i < 16; i++)
        {
            if (thread_id < STN)
            dc[(row16 * 16 + i) * ldb + dimN_index + thread_id] = 
                s_out_tile[i * SHAREB + thread_id];
        }
    }
}

template<int TN>
__global__ void stir_spmm_cuda_kernel_v3(
    int tilen_offset, int rbnum, int cbnum, int rowA, int colA, int ldb, int ldc, MatIndex rowblks, MatIndex *block_row_ptr, MatIndex *block_col_start, MatIndex *block_col_end, MatIndex *A_rowptr, MatIndex *A_colind, Tile *A_tiles, char *A_data, MatValue *db, MatValue *dc
)
{
    const int thread_id = threadIdx.x;
    const int lwarp_id = thread_id >> 5;
    int dimM_index = blockIdx.x * 2 + lwarp_id;
    int dimN_index = blockIdx.y * 32 + tilen_offset;
    int blki_blc = dimM_index;
    const int lane_id = thread_id & 31;

    __shared__ MatValue s_dense_tile_array[16 * TN * 2];
    __shared__ MatValue s_out_tile_array[16 * TN * 2];
    MatValue rc[TN] = {0};
    // __shared__ MatValue s_value_tile[16 * 16];
    // __shared__ TileIndex s_index_tile[16 * 16];

    __shared__ int s_colidx_array[PREFETCH_SMEM_TH_SPMM * 2];
    __shared__ uint8_t s_valoff_array[PREFETCH_SMEM_TH_SPMM * 2];
    __shared__ uint64_t s_bitoff_array[PREFETCH_SMEM_TH_SPMM * 2];
    __shared__ TileFormat s_format_array[PREFETCH_SMEM_TH_SPMM * 2];

    MatValue * s_dense_tile = s_dense_tile_array + lwarp_id * 16 * TN;
    MatValue * s_out_tile = s_out_tile_array + lwarp_id * 16 * TN;
    int* s_colidx = s_colidx_array + lwarp_id * PREFETCH_SMEM_TH_SPMM;
    uint8_t* s_valoff = s_valoff_array + lwarp_id * PREFETCH_SMEM_TH_SPMM;
    uint64_t* s_bitoff = s_bitoff_array + lwarp_id * PREFETCH_SMEM_TH_SPMM;
    TileFormat* s_format = s_format_array + lwarp_id * PREFETCH_SMEM_TH_SPMM;

    if (blki_blc >= rowblks)
        return;
    
    MatIndex row16 = block_row_ptr[blki_blc];
    MatIndex signbit = row16 & 0x80000000;
    row16 ^= signbit;

    MatIndex start_Aj = signbit ? block_col_start[blki_blc] : A_rowptr[row16], end_Aj = signbit ? block_col_end[blki_blc] : A_rowptr[row16 + 1];

    if (lane_id < end_Aj - start_Aj) {
        s_colidx[lane_id] = A_colind[start_Aj + lane_id] << 4;
        s_valoff[lane_id] = A_tiles[start_Aj + lane_id].bitslen;
        s_bitoff[lane_id] = A_tiles[start_Aj + lane_id].bits_off;
        s_format[lane_id] = A_tiles[start_Aj + lane_id].fmt;
    }

    if (lane_id < TN)
    {
        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            s_out_tile[i * TN + lane_id] = 0;
        }
    }
    __syncthreads();

    for (int Aj = start_Aj; Aj < end_Aj; ++Aj)
    {
        MatIndex off16 = s_colidx[Aj - start_Aj];
        TileFormat fmt = s_format[Aj - start_Aj];
        MatValue *A_val = (MatValue *)(A_data + s_bitoff[Aj - start_Aj] + s_valoff[Aj - start_Aj] * 16);
        TileIndex *bits = (TileIndex *)(A_data + s_bitoff[Aj - start_Aj]);
        
        if (lane_id < TN) {
            #pragma unroll
            for (int i = 0; i < 16; ++i) s_dense_tile[i * TN + lane_id] = db[(off16 + i) * ldb + dimN_index + lane_id];
        }
        __syncthreads();

        switch (fmt)
        {
        case COO:
            {
                MatIndex nnz = bits[0] + 1;

                for (int i = 0; i < nnz; ++i)
                {
                    MatIndex idx = bits[i + 1], row = idx >> 4, col = idx & 15;
                    MatValue ra = A_val[i];
                    if (lane_id < TN)
                    s_out_tile[row * TN + lane_id] += ra * s_dense_tile[col * TN + lane_id];
                }
            }
            break;
        case CSR:
            {
                TileIndex *colidx = bits + 17;
                int vwarp_id = lane_id & 15, vlane_id = lane_id >> 4;
                int start = bits[vwarp_id], stop = bits[vwarp_id + 1];

                for (int j = start + vlane_id; j < stop; j += 2)
                {
                    TileIndex col = TileIndex_CSR_Get(colidx, j);
                    MatValue val = A_val[j];
                    
                    #pragma unroll
                    for (int l = 0; l < TN; ++l)
                        rc[l] += val * s_dense_tile[col * TN + l];
                }
            }
            break;
        case ELL:
            {
                int elllen = (TileIndex_CSR_Get(bits, 0) + 1);
                int vwarp_id = lane_id & 15, vlane_id = lane_id >> 4;
                int start = vwarp_id * elllen, stop = (vwarp_id + 1) * elllen;

                for (int j = start + vlane_id; j < stop; j += 2)
                {
                    TileIndex col = TileIndex_CSR_Get(bits, 1 + j);
                    MatValue val = A_val[j];
                    
                    #pragma unroll
                    for (int l = 0; l < TN; ++l)
                        rc[l] += val * s_dense_tile[col * TN + l];
                }
            }
            break;
        case HYB:
            {
                int min_row_len = bits[0], coo_cnt = bits[1], elllen = min_row_len * 16;
                int coo_start = (elllen + 1) / 2 + 2;
                int vwarp_id = lane_id & 15, vlane_id = lane_id >> 4;
                int start = vwarp_id * min_row_len, stop = (vwarp_id + 1) * min_row_len;

                for (int j = start + vlane_id; j < stop; j += 2)
                {
                    TileIndex col = TileIndex_CSR_Get(bits, 4 + j);
                    MatValue val = A_val[j];

                    #pragma unroll
                    for (int l = 0; l < TN; ++l)
                        rc[l] += val * s_dense_tile[col * TN + l];
                }

                for (int i = 0; i < coo_cnt; ++i)
                {
                    MatIndex idx = bits[coo_start + i], row = idx >> 4, col = idx & 15;
                    MatValue ra = A_val[i + elllen];

                    if (lane_id < TN)
                        s_out_tile[row * TN + lane_id] += ra * s_dense_tile[col * TN + lane_id];
                }
            }
            break;
        case DRW:
            {
                int cnt = TileIndex_CSR_Get(bits, 0) + 1;

                if (lane_id < TN)
                    for (int i = 0; i < cnt; ++i)
                    {
                        TileIndex row = TileIndex_CSR_Get(bits, 1 + i);
                        MatValue res = 0;

                        #pragma unroll
                        for (int j = 0; j < 16; ++j)
                        {
                            MatValue ra = A_val[i * 16 + j];
                            MatValue rb = s_dense_tile[j * TN + lane_id];
                            res += ra * rb;
                        }

                        s_out_tile[row * TN + lane_id] += res;
                    }
            }
            break;
        case DCL:
            {
                int cnt = TileIndex_CSR_Get(bits, 0) + 1, nnz = cnt * 16;

                for (int glid = lane_id; glid < nnz; glid += 32)
                {
                    int rj = glid >> 4, ri = (rj << 4) | (glid & 15);
                    rj = TileIndex_CSR_Get(bits, 1 + rj);
                    MatValue ra = A_val[ri];

                    #pragma unroll
                    for (int l = 0; l < TN; ++l) rc[l] += ra * s_dense_tile[rj * TN + l];
                }
            }
            break;
        case DNS:
            {
                int vwarp_id = lane_id & 15, vlane_id = lane_id >> 4;
                int start = vwarp_id * 16, stop = (vwarp_id + 1) * 16;

                int dnscol = lane_id >> 4;
                for (int i = start + vlane_id; i < stop; i += 2)
                {
                    MatValue val = A_val[i];
                    for (int l = 0; l < TN; l++)
                    {
                        rc[l] += val * s_dense_tile[dnscol * TN + l];
                    }
                    dnscol = dnscol + 2;
                }
            }
            break;
        default:
            break;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int l = 0; l < TN; l++)
    {
        rc[l] += __shfl_down_sync(0xFFFFFFFF, rc[l], 16);
        if (lane_id < 16)
            s_out_tile[lane_id * TN + l] += rc[l];
    }
    __syncthreads();
    
    if (signbit)
    {
        if (lane_id < TN) {
            #pragma unroll
            for (int i = 0; i < 16; i++)
            {
                atomicAdd(&dc[(row16 * 16 + i) * ldb + dimN_index + lane_id], s_out_tile[i * TN + lane_id]);
            }
        }
    }
    else
    {
        if (lane_id < TN) {
            #pragma unroll
            for (int i = 0; i < 16; i++)
            {
                dc[(row16 * 16 + i) * ldb + dimN_index + lane_id] = 
                    s_out_tile[i * TN + lane_id];
            }
        }
    }
}

void switch_tilespmm_cuda(
    int tilem, int tilen, int rowA, int colA, int ldb, int ldc, MatIndex rowblks, MatIndex *block_row_ptr, MatIndex *block_col_start, MatIndex *block_col_end, MatIndex *A_rowptr, MatIndex *A_colind, Tile *A_tiles, char *A_data, MatValue *db, MatValue *dc)
{
    int ldc1 = (ldc / 64) * 64;
    int ldc2 = ldc % 64 - 32;
    int ldc3 = ldc % 64;
    // echo(debug, "%d %d %d", ldc1, ldc2, ldc3);
    if (ldc1 > 0)
    {
        int num_threads = 2 * 32;
        int num_blocks_y = ceil((float)ldc1 / (float)64);
        dim3 grid_dim(rowblks, num_blocks_y, 1);
        dim3 block_dim(num_threads, 1, 1);
        stir_spmm_cuda_kernel_v1<<<grid_dim, block_dim>>>(tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
    }

    if (ldc2 > 0)
    {
        int tilen_offset = ldc1;
        int num_threads = 2 * 32;
        dim3 grid_dim(rowblks, 1, 1);
        dim3 block_dim(num_threads, 1, 1);
        switch (ldc2)
        {
            case 1:
                stir_spmm_cuda_kernel_v2<16, 17, 33, 34><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 2:
                stir_spmm_cuda_kernel_v2<17, 17, 34, 34><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 3:
                stir_spmm_cuda_kernel_v2<17, 18, 35, 36><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 4:
                stir_spmm_cuda_kernel_v2<18, 18, 36, 36><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 5:
                stir_spmm_cuda_kernel_v2<18, 19, 37, 38><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 6:
                stir_spmm_cuda_kernel_v2<19, 19, 38, 38><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 7:
                stir_spmm_cuda_kernel_v2<19, 20, 39, 40><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 8:
                stir_spmm_cuda_kernel_v2<20, 20, 40, 40><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 9:
                stir_spmm_cuda_kernel_v2<20, 21, 41, 42><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 10:
                stir_spmm_cuda_kernel_v2<21, 21, 42, 42><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 11:
                stir_spmm_cuda_kernel_v2<21, 22, 43, 44><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 12:
                stir_spmm_cuda_kernel_v2<22, 22, 44, 44><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 13:
                stir_spmm_cuda_kernel_v2<22, 23, 45, 46><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 14:
                stir_spmm_cuda_kernel_v2<23, 23, 46, 46><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 15:
                stir_spmm_cuda_kernel_v2<23, 24, 47, 48><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 16:
                stir_spmm_cuda_kernel_v2<24, 24, 48, 48><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 17:
                stir_spmm_cuda_kernel_v2<24, 25, 49, 50><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 18:
                stir_spmm_cuda_kernel_v2<25, 25, 50, 50><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 19:
                stir_spmm_cuda_kernel_v2<25, 26, 51, 52><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 20:
                stir_spmm_cuda_kernel_v2<26, 26, 52, 52><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 21:
                stir_spmm_cuda_kernel_v2<26, 27, 53, 54><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 22:
                stir_spmm_cuda_kernel_v2<27, 27, 54, 54><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 23:
                stir_spmm_cuda_kernel_v2<27, 28, 55, 56><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 24:
                stir_spmm_cuda_kernel_v2<28, 28, 56, 56><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 25:
                stir_spmm_cuda_kernel_v2<28, 29, 57, 58><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 26:
                stir_spmm_cuda_kernel_v2<29, 29, 58, 58><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 27:
                stir_spmm_cuda_kernel_v2<29, 30, 59, 60><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 28:
                stir_spmm_cuda_kernel_v2<30, 30, 60, 60><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 29:
                stir_spmm_cuda_kernel_v2<30, 31, 61, 62><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 30:
                stir_spmm_cuda_kernel_v2<31, 31, 62, 62><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 31:
                stir_spmm_cuda_kernel_v2<31, 32, 63, 64><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
        }
    }
    else if (ldc3 > 0)
    {
        int tilen_offset = ldc1;
        int num_blocks_x = (float)(rowblks + 1) / 2;
        int num_threads = 2 * 32;
        dim3 grid_dim(num_blocks_x, 1, 1);
        dim3 block_dim(num_threads, 1, 1);
        switch (ldc3)
        {
            case 1:
                stir_spmm_cuda_kernel_v3<1><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 2:
                stir_spmm_cuda_kernel_v3<2><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 3:
                stir_spmm_cuda_kernel_v3<3><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 4:
                stir_spmm_cuda_kernel_v3<4><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 5:
                stir_spmm_cuda_kernel_v3<5><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 6:
                stir_spmm_cuda_kernel_v3<6><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 7:
                stir_spmm_cuda_kernel_v3<7><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 8:
                stir_spmm_cuda_kernel_v3<8><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 9:
                stir_spmm_cuda_kernel_v3<9><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 10:
                stir_spmm_cuda_kernel_v3<10><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 11:
                stir_spmm_cuda_kernel_v3<11><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 12:
                stir_spmm_cuda_kernel_v3<12><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 13:
                stir_spmm_cuda_kernel_v3<13><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 14:
                stir_spmm_cuda_kernel_v3<14><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 15:
                stir_spmm_cuda_kernel_v3<15><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 16:
                stir_spmm_cuda_kernel_v3<16><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 17:
                stir_spmm_cuda_kernel_v3<17><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 18:
                stir_spmm_cuda_kernel_v3<18><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 19:
                stir_spmm_cuda_kernel_v3<19><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 20:
                stir_spmm_cuda_kernel_v3<20><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 21:
                stir_spmm_cuda_kernel_v3<21><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 22:
                stir_spmm_cuda_kernel_v3<22><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 23:
                stir_spmm_cuda_kernel_v3<23><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 24:
                stir_spmm_cuda_kernel_v3<24><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 25:
                stir_spmm_cuda_kernel_v3<25><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 26:
                stir_spmm_cuda_kernel_v3<26><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 27:
                stir_spmm_cuda_kernel_v3<27><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 28:
                stir_spmm_cuda_kernel_v3<28><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 29:
                stir_spmm_cuda_kernel_v3<29><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 30:
                stir_spmm_cuda_kernel_v3<30><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 31:
                stir_spmm_cuda_kernel_v3<31><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
            case 32:
                stir_spmm_cuda_kernel_v3<32><<<grid_dim, block_dim>>>(tilen_offset, tilem, tilen, rowA, colA, ldb, ldc, rowblks, block_row_ptr, block_col_start, block_col_end, A_rowptr, A_colind, A_tiles, A_data, db, dc);
                break;
        }
    }
}

double*spmm_col_result(MatIndex m, MatIndex n, MatIndex nnz, MatIndex *row_ptr, MatIndex *col_idx, MatValue *csr_val)
{
    MatValue *c = (MatValue *)calloc(m, sizeof(MatValue));

    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        MatValue sum = 0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            sum += csr_val[j];
        }
        c[i] = sum;
    }

    return c;
}

int test_spmm(const char *filename, int repeat, int right_n, int device)
{
    echo(info, "Test SpMM with matrix file: %s, repeat: %d, right-n: %d", filename, repeat, right_n);

    MatIndex report_nnz;
    MatIndex m, n, nnz, *row_ptr, *col_idx;
    MatValue *csr_val;
    MatIndex isSymmetric;
    mmio_allinone(filename, &m, &n, &nnz, &isSymmetric, &row_ptr, &col_idx, &csr_val);
    double predict_time = 0;
    BaseMatrix* mat = load_mtx_2_tile(filename, &report_nnz, "pixel", m, n, nnz, row_ptr, col_idx, csr_val, predict_time);

    if (mat == nullptr)
    {
        echo(error, "Failed to load matrix file: %s", filename);
        return -1;
    }

    echo(success, "Matrix loaded from file: \"%s\", m: %d, n: %d, nnz: %d (report: %d)", filename, mat->meta_m, mat->meta_n, mat->meta_nnz, report_nnz);

    // 负载均衡，一个block算4个块
    cudaInit(device);
    Timer timer_pre;
    timer_start(timer_pre);
    MatIndex *block_row, *block_col_start, *block_col_end;
    MatIndex row_blocks = lb_spmm_coo_style(mat, &block_row, &block_col_start, &block_col_end);
    timer_end(timer_pre);
    echo(success, "Block partition time: %.3lf ms", timer_duration(timer_pre));

    MatValue *b = (MatValue *)malloc(mat->meta_n * right_n * sizeof(MatValue)), *c = (MatValue *)calloc(mat->meta_m * right_n, sizeof(MatValue));
    for (int i = 0; i < mat->meta_n * right_n; i++)
    {
        b[i] = 1.0;
    }

    BaseMatrix *d_mat = BaseMatrix_Host_to_Device(mat);
    MatValue *d_b, *d_c;
    cudaCheckError(cudaMalloc(&d_b, mat->meta_n * right_n * sizeof(MatValue)));
    cudaCheckError(cudaMalloc(&d_c, mat->meta_m * right_n * sizeof(MatValue)));
    cudaCheckError(cudaMemcpy(d_b, b, mat->meta_n * right_n * sizeof(MatValue), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemset(d_c, 0, mat->meta_m * right_n * sizeof(MatValue)));

    switch_tilespmm_cuda(
            d_mat->_m, d_mat->_n, d_mat->meta_m, d_mat->meta_n, right_n, right_n, row_blocks, block_row, block_col_start, block_col_end,
            d_mat->tile_row_ptr, d_mat->tile_col_idx, d_mat->tiles, d_mat->data, d_b, d_c);
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        echo(error, "CUDA error: %s", cudaGetErrorString(e));
    }
    cudaMemcpy(c, d_c, mat->meta_m * right_n * sizeof(MatValue), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 200; i++)
    {
        cudaMemset(d_c, 0, mat->meta_m * right_n * sizeof(MatValue));
        switch_tilespmm_cuda(
            d_mat->_m, d_mat->_n, d_mat->meta_m, d_mat->meta_n, right_n, right_n, row_blocks, block_row, block_col_start, block_col_end,
            d_mat->tile_row_ptr, d_mat->tile_col_idx, d_mat->tiles, d_mat->data, d_b, d_c);
        cudaDeviceSynchronize();
    }

    double duration = 1000000;
    for (int i = 0; i < repeat; i++)
    {
        cudaMemset(d_c, 0, mat->meta_m * right_n * sizeof(MatValue));
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        switch_tilespmm_cuda(
            d_mat->_m, d_mat->_n, d_mat->meta_m, d_mat->meta_n, right_n, right_n, row_blocks, block_row, block_col_start, block_col_end,
            d_mat->tile_row_ptr, d_mat->tile_col_idx, d_mat->tiles, d_mat->data, d_b, d_c);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaDeviceSynchronize();
        duration = min(milliseconds, duration);
    }

    MatValue *c_col_res = spmm_col_result(m, n, nnz, row_ptr, col_idx, csr_val);
    bool flag = true;
    // #pragma omp parallel for
    for (int i = 0; i < mat->meta_m * right_n; i++)
    {
        if (fabs(c[i] - c_col_res[i / right_n]) > 1e-6 && flag)
        {
            echo(error, "Error: c[%d][%d] (%.6lf) != c_col_res[%d][%d] (%.6lf)", i / right_n, i % right_n, c[i], i / right_n, i % right_n, c_col_res[i / right_n]);
            flag = false;
            break;
        }
    }

    if (flag)
    {
        echo(success, "Result is correct! Preprocess time: %.3lf ms, Calculate time: %.3lf ms, GFLOPS: %.3lf", timer_duration(timer_pre), duration, 2.0 * mat->meta_nnz / duration * right_n / 1e6);
    }
    /*2.0 * mat->meta_nnz / cusparse_duration * right_n / 1e6*/
    printf("%d,%d,%d,%d,%.3lf\n", mat->meta_m, mat->meta_n, mat->meta_nnz, right_n, mat->meta_nnz * 2.0 / duration * right_n / 1e6);
    
    cudaFree(block_row);
    cudaFree(block_col_start);
    cudaFree(block_col_end);
    cudaCheckError(cudaFree(d_b));
    cudaCheckError(cudaFree(d_c));
    DestroyBaseMatrixDevice(d_mat);
    DestroyBaseMatrixHost(mat);
    free(b);
    free(c);
    // free(c_cuda);
    free(row_ptr);
    free(col_idx);
    free(csr_val);
    free(c_col_res);
    return 0;
}