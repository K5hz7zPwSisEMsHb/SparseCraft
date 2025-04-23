#include <tmatrix/Calculation/spgemm_subkernels.cuh>
#include <tmatrix/Calculation/ShulkerBox/kernel_matrix/spgemm_kernels.cuh>
#include <tmatrix/Calculation/bin_spgemm.cuh>
#include <tmatrix/Utils/timer.h>
#include <tmatrix/Utils/msg.h>
#include <tmatrix/Calculation/nsparse_asm.cuh>

__device__ void (*kernel_matrix[7][7][7])(
    TileIndex *Abits, MatValue *Avals, TileIndex *Bbits, MatValue *Bvals, TileIndex *Cbits, MatValue *Cvals, uint64_t *Cbitmap,
    const TileIndex lane_id, const TileIndex vwarp_id, const TileIndex vlane_id, const TileIndex hwarp_id, const TileIndex hlane_id, bool debug_flag) = 
{
    {
        {coo_x_coo_2_coo, coo_x_coo_2_csr, coo_x_coo_2_ell, coo_x_coo_2_hyb, coo_x_coo_2_drw, coo_x_coo_2_dcl, coo_x_coo_2_dns},
        {coo_x_csr_2_coo, coo_x_csr_2_csr, coo_x_csr_2_ell, coo_x_csr_2_hyb, coo_x_csr_2_drw, coo_x_csr_2_dcl, coo_x_csr_2_dns},
        {coo_x_ell_2_coo, coo_x_ell_2_csr, coo_x_ell_2_ell, coo_x_ell_2_hyb, coo_x_ell_2_drw, coo_x_ell_2_dcl, coo_x_ell_2_dns},
        {coo_x_hyb_2_coo, coo_x_hyb_2_csr, coo_x_hyb_2_ell, coo_x_hyb_2_hyb, coo_x_hyb_2_drw, coo_x_hyb_2_dcl, coo_x_hyb_2_dns},
        {coo_x_drw_2_coo, coo_x_drw_2_csr, coo_x_drw_2_ell, coo_x_drw_2_hyb, coo_x_drw_2_drw, coo_x_drw_2_dcl, coo_x_drw_2_dns},
        {coo_x_dcl_2_coo, coo_x_dcl_2_csr, coo_x_dcl_2_ell, coo_x_dcl_2_hyb, coo_x_dcl_2_drw, coo_x_dcl_2_dcl, coo_x_dcl_2_dns},
        {coo_x_dns_2_coo, coo_x_dns_2_csr, coo_x_dns_2_ell, coo_x_dns_2_hyb, coo_x_dns_2_drw, coo_x_dns_2_dcl, coo_x_dns_2_dns},
    },
    {
        {csr_x_coo_2_coo, csr_x_coo_2_csr, csr_x_coo_2_ell, csr_x_coo_2_hyb, csr_x_coo_2_drw, csr_x_coo_2_dcl, csr_x_coo_2_dns},
        {csr_x_csr_2_coo, csr_x_csr_2_csr, csr_x_csr_2_ell, csr_x_csr_2_hyb, csr_x_csr_2_drw, csr_x_csr_2_dcl, csr_x_csr_2_dns},
        {csr_x_ell_2_coo, csr_x_ell_2_csr, csr_x_ell_2_ell, csr_x_ell_2_hyb, csr_x_ell_2_drw, csr_x_ell_2_dcl, csr_x_ell_2_dns},
        {csr_x_hyb_2_coo, csr_x_hyb_2_csr, csr_x_hyb_2_ell, csr_x_hyb_2_hyb, csr_x_hyb_2_drw, csr_x_hyb_2_dcl, csr_x_hyb_2_dns},
        {csr_x_drw_2_coo, csr_x_drw_2_csr, csr_x_drw_2_ell, csr_x_drw_2_hyb, csr_x_drw_2_drw, csr_x_drw_2_dcl, csr_x_drw_2_dns},
        {csr_x_dcl_2_coo, csr_x_dcl_2_csr, csr_x_dcl_2_ell, csr_x_dcl_2_hyb, csr_x_dcl_2_drw, csr_x_dcl_2_dcl, csr_x_dcl_2_dns},
        {csr_x_dns_2_coo, csr_x_dns_2_csr, csr_x_dns_2_ell, csr_x_dns_2_hyb, csr_x_dns_2_drw, csr_x_dns_2_dcl, csr_x_dns_2_dns},
    },
    {
        {ell_x_coo_2_coo, ell_x_coo_2_csr, ell_x_coo_2_ell, ell_x_coo_2_hyb, ell_x_coo_2_drw, ell_x_coo_2_dcl, ell_x_coo_2_dns},
        {ell_x_csr_2_coo, ell_x_csr_2_csr, ell_x_csr_2_ell, ell_x_csr_2_hyb, ell_x_csr_2_drw, ell_x_csr_2_dcl, ell_x_csr_2_dns},
        {ell_x_ell_2_coo, ell_x_ell_2_csr, ell_x_ell_2_ell, ell_x_ell_2_hyb, ell_x_ell_2_drw, ell_x_ell_2_dcl, ell_x_ell_2_dns},
        {ell_x_hyb_2_coo, ell_x_hyb_2_csr, ell_x_hyb_2_ell, ell_x_hyb_2_hyb, ell_x_hyb_2_drw, ell_x_hyb_2_dcl, ell_x_hyb_2_dns},
        {ell_x_drw_2_coo, ell_x_drw_2_csr, ell_x_drw_2_ell, ell_x_drw_2_hyb, ell_x_drw_2_drw, ell_x_drw_2_dcl, ell_x_drw_2_dns},
        {ell_x_dcl_2_coo, ell_x_dcl_2_csr, ell_x_dcl_2_ell, ell_x_dcl_2_hyb, ell_x_dcl_2_drw, ell_x_dcl_2_dcl, ell_x_dcl_2_dns},
        {ell_x_dns_2_coo, ell_x_dns_2_csr, ell_x_dns_2_ell, ell_x_dns_2_hyb, ell_x_dns_2_drw, ell_x_dns_2_dcl, ell_x_dns_2_dns},
    },
    {
        {hyb_x_coo_2_coo, hyb_x_coo_2_csr, hyb_x_coo_2_ell, hyb_x_coo_2_hyb, hyb_x_coo_2_drw, hyb_x_coo_2_dcl, hyb_x_coo_2_dns},
        {hyb_x_csr_2_coo, hyb_x_csr_2_csr, hyb_x_csr_2_ell, hyb_x_csr_2_hyb, hyb_x_csr_2_drw, hyb_x_csr_2_dcl, hyb_x_csr_2_dns},
        {hyb_x_ell_2_coo, hyb_x_ell_2_csr, hyb_x_ell_2_ell, hyb_x_ell_2_hyb, hyb_x_ell_2_drw, hyb_x_ell_2_dcl, hyb_x_ell_2_dns},
        {hyb_x_hyb_2_coo, hyb_x_hyb_2_csr, hyb_x_hyb_2_ell, hyb_x_hyb_2_hyb, hyb_x_hyb_2_drw, hyb_x_hyb_2_dcl, hyb_x_hyb_2_dns},
        {hyb_x_drw_2_coo, hyb_x_drw_2_csr, hyb_x_drw_2_ell, hyb_x_drw_2_hyb, hyb_x_drw_2_drw, hyb_x_drw_2_dcl, hyb_x_drw_2_dns},
        {hyb_x_dcl_2_coo, hyb_x_dcl_2_csr, hyb_x_dcl_2_ell, hyb_x_dcl_2_hyb, hyb_x_dcl_2_drw, hyb_x_dcl_2_dcl, hyb_x_dcl_2_dns},
        {hyb_x_dns_2_coo, hyb_x_dns_2_csr, hyb_x_dns_2_ell, hyb_x_dns_2_hyb, hyb_x_dns_2_drw, hyb_x_dns_2_dcl, hyb_x_dns_2_dns},
    },
    {
        {drw_x_coo_2_coo, drw_x_coo_2_csr, drw_x_coo_2_ell, drw_x_coo_2_hyb, drw_x_coo_2_drw, drw_x_coo_2_dcl, drw_x_coo_2_dns},
        {drw_x_csr_2_coo, drw_x_csr_2_csr, drw_x_csr_2_ell, drw_x_csr_2_hyb, drw_x_csr_2_drw, drw_x_csr_2_dcl, drw_x_csr_2_dns},
        {drw_x_ell_2_coo, drw_x_ell_2_csr, drw_x_ell_2_ell, drw_x_ell_2_hyb, drw_x_ell_2_drw, drw_x_ell_2_dcl, drw_x_ell_2_dns},
        {drw_x_hyb_2_coo, drw_x_hyb_2_csr, drw_x_hyb_2_ell, drw_x_hyb_2_hyb, drw_x_hyb_2_drw, drw_x_hyb_2_dcl, drw_x_hyb_2_dns},
        {drw_x_drw_2_coo, drw_x_drw_2_csr, drw_x_drw_2_ell, drw_x_drw_2_hyb, drw_x_drw_2_drw, drw_x_drw_2_dcl, drw_x_drw_2_dns},
        {drw_x_dcl_2_coo, drw_x_dcl_2_csr, drw_x_dcl_2_ell, drw_x_dcl_2_hyb, drw_x_dcl_2_drw, drw_x_dcl_2_dcl, drw_x_dcl_2_dns},
        {drw_x_dns_2_coo, drw_x_dns_2_csr, drw_x_dns_2_ell, drw_x_dns_2_hyb, drw_x_dns_2_drw, drw_x_dns_2_dcl, drw_x_dns_2_dns},
    },
    {
        {dcl_x_coo_2_coo, dcl_x_coo_2_csr, dcl_x_coo_2_ell, dcl_x_coo_2_hyb, dcl_x_coo_2_drw, dcl_x_coo_2_dcl, dcl_x_coo_2_dns},
        {dcl_x_csr_2_coo, dcl_x_csr_2_csr, dcl_x_csr_2_ell, dcl_x_csr_2_hyb, dcl_x_csr_2_drw, dcl_x_csr_2_dcl, dcl_x_csr_2_dns},
        {dcl_x_ell_2_coo, dcl_x_ell_2_csr, dcl_x_ell_2_ell, dcl_x_ell_2_hyb, dcl_x_ell_2_drw, dcl_x_ell_2_dcl, dcl_x_ell_2_dns},
        {dcl_x_hyb_2_coo, dcl_x_hyb_2_csr, dcl_x_hyb_2_ell, dcl_x_hyb_2_hyb, dcl_x_hyb_2_drw, dcl_x_hyb_2_dcl, dcl_x_hyb_2_dns},
        {dcl_x_drw_2_coo, dcl_x_drw_2_csr, dcl_x_drw_2_ell, dcl_x_drw_2_hyb, dcl_x_drw_2_drw, dcl_x_drw_2_dcl, dcl_x_drw_2_dns},
        {dcl_x_dcl_2_coo, dcl_x_dcl_2_csr, dcl_x_dcl_2_ell, dcl_x_dcl_2_hyb, dcl_x_dcl_2_drw, dcl_x_dcl_2_dcl, dcl_x_dcl_2_dns},
        {dcl_x_dns_2_coo, dcl_x_dns_2_csr, dcl_x_dns_2_ell, dcl_x_dns_2_hyb, dcl_x_dns_2_drw, dcl_x_dns_2_dcl, dcl_x_dns_2_dns},
    },
    {
        {dns_x_coo_2_coo, dns_x_coo_2_csr, dns_x_coo_2_ell, dns_x_coo_2_hyb, dns_x_coo_2_drw, dns_x_coo_2_dcl, dns_x_coo_2_dns},
        {dns_x_csr_2_coo, dns_x_csr_2_csr, dns_x_csr_2_ell, dns_x_csr_2_hyb, dns_x_csr_2_drw, dns_x_csr_2_dcl, dns_x_csr_2_dns},
        {dns_x_ell_2_coo, dns_x_ell_2_csr, dns_x_ell_2_ell, dns_x_ell_2_hyb, dns_x_ell_2_drw, dns_x_ell_2_dcl, dns_x_ell_2_dns},
        {dns_x_hyb_2_coo, dns_x_hyb_2_csr, dns_x_hyb_2_ell, dns_x_hyb_2_hyb, dns_x_hyb_2_drw, dns_x_hyb_2_dcl, dns_x_hyb_2_dns},
        {dns_x_drw_2_coo, dns_x_drw_2_csr, dns_x_drw_2_ell, dns_x_drw_2_hyb, dns_x_drw_2_drw, dns_x_drw_2_dcl, dns_x_drw_2_dns},
        {dns_x_dcl_2_coo, dns_x_dcl_2_csr, dns_x_dcl_2_ell, dns_x_dcl_2_hyb, dns_x_dcl_2_drw, dns_x_dcl_2_dcl, dns_x_dcl_2_dns},
        {dns_x_dns_2_coo, dns_x_dns_2_csr, dns_x_dns_2_ell, dns_x_dns_2_hyb, dns_x_dns_2_drw, dns_x_dns_2_dcl, dns_x_dns_2_dns},
    },
};

template <int WARPS, int NBLKS>
__global__ void bin_spgemm_smem(
    const int *d_row_perm, int bin_offset,
    const MatIndex *A_row_ptr, const MatIndex *A_col_idx, const Tile *A_tiles, const char *A_data,
    const MatIndex *B_row_ptr, const MatIndex *B_col_idx, const Tile *B_tiles, const char *B_data,
    const MatIndex *C_row_ptr, const MatIndex *C_col_idx,       Tile *C_tiles,       char *C_data)
{
    int thread_id = threadIdx.x;
    int lwarp_id = thread_id >> 5, lane_id = thread_id & 31;
    int vwarp_id = lane_id >> 1, hwarp_id = lane_id >> 4;
    int vlane_id = lane_id & 1, hlane_id = lane_id & 15;

    __shared__ double psum[WARPS * 256], s_cval[NBLKS * 256];
    double* warp_psum = psum + lwarp_id * 256;

    int row = d_row_perm[blockIdx.x + bin_offset]; // row

    for (int i = thread_id; i < NBLKS * 256; i += blockDim.x)
    {
        s_cval[i] = 0;
    }

    __syncthreads();
    for (int j = A_row_ptr[row] + lwarp_id; j < A_row_ptr[row + 1]; j += WARPS)
    {
        int acol = ld_gbl_int32(A_col_idx + j);
        TileFormat Afmt = A_tiles[j].fmt;
        TileIndex* abits = (TileIndex*)(A_data + A_tiles[j].bits_off);
        MatValue* aval = (MatValue*)(A_data + A_tiles[j].bits_off + A_tiles[j].bitslen * 16);
        for (int k = B_row_ptr[acol]; k < B_row_ptr[acol + 1]; k++)
        {
            int bcol = B_col_idx[k];
            int cj = -1;
            if (lane_id == 0) cj = binary_find_index(C_col_idx + C_row_ptr[row], C_row_ptr[row + 1] - C_row_ptr[row], bcol);
            cj = __shfl_sync(0xffffffff, cj, 0);
            if (cj >= 0) 
            {
                int valslen = C_tiles[C_row_ptr[row] + cj].valslen + 1;
                for (int i = lane_id; i < valslen; i += 32) warp_psum[i] = 0;
                TileFormat Bfmt = B_tiles[k].fmt;
                TileIndex* bbits = (TileIndex*)(B_data + B_tiles[k].bits_off);
                MatValue* bval = (MatValue*)(B_data + B_tiles[k].bits_off + B_tiles[k].bitslen * 16);
                TileFormat Cfmt = C_tiles[C_row_ptr[row] + cj].fmt;
                TileIndex *cbits = (TileIndex *)(C_data + C_tiles[C_row_ptr[row] + cj].bits_off);
                uint64_t *c_bitmap = C_tiles[C_row_ptr[row] + cj].bitmap;
                
                __syncwarp();
                kernel_matrix[Afmt][Bfmt][Cfmt](abits, aval, bbits, bval, cbits, warp_psum, c_bitmap, lane_id, vwarp_id, vlane_id, hwarp_id, hlane_id, false);
                __syncwarp();
                for (int i = lane_id; i < valslen; i += 32) atomicAdd(s_cval + cj * 256 + i, warp_psum[i]);
            }
        }
    }

    __syncthreads();
    for (int i = C_row_ptr[row] + lwarp_id; i < C_row_ptr[row + 1]; i += WARPS)
    {
        int idx = i - C_row_ptr[row];
        int valslen = C_tiles[i].valslen + 1;
        for (int j = lane_id; j < valslen; j += 32)
        {
            MatValue* cval = (MatValue*)(C_data + C_tiles[i].bits_off + C_tiles[i].bitslen * 16);
            cval[j] = s_cval[idx * 256 + j];
        }
    }
}

template <int WARPS>
__global__ void bin_spgemm_gmem(
    const int *d_row_perm, int bin_offset, 
    const int *A_row_ptr, const int *A_col_idx, const Tile *A_tiles, const char *A_data,
    const int *B_row_ptr, const int *B_col_idx, const Tile *B_tiles, const char *B_data,
    const int *C_row_ptr, const int *C_col_idx,       Tile *C_tiles,       char *C_data)
{
    int row = d_row_perm[blockIdx.x + bin_offset];
    int lwarp_id = threadIdx.x >> 5, lane_id = threadIdx.x & 31;
    int vwarp_id = lane_id >> 1, hwarp_id = lane_id >> 4;
    int vlane_id = lane_id & 1, hlane_id = lane_id & 15;

    // if (row == 0 && threadIdx.x == 0) printf("[GMEM] blockIdx.x: %d, bin_offset: %d\n", blockIdx.x, bin_offset);

    __shared__ MatValue psum[WARPS * 256];
    double *warp_psum = psum + lwarp_id * 256;

    // __syncthreads();
    for (int j = A_row_ptr[row] + lwarp_id; j < A_row_ptr[row + 1]; j += WARPS)
    {
        int acol = ld_gbl_int32(A_col_idx + j);
        TileFormat Afmt = A_tiles[j].fmt;
        TileIndex* abits = (TileIndex*)(A_data + A_tiles[j].bits_off);
        MatValue* aval = (MatValue*)(A_data + A_tiles[j].bits_off + A_tiles[j].bitslen * 16);
        for (int k = B_row_ptr[acol]; k < B_row_ptr[acol + 1]; k++)
        {
            int bcol = B_col_idx[k];
            // int cj = binary_find_index_warp_level(C_col_idx + C_row_ptr[row], C_row_ptr[row + 1] - C_row_ptr[row], bcol, lane_id);
            int cj = -1;
            if (lane_id == 0) cj = binary_find_index(C_col_idx + C_row_ptr[row], C_row_ptr[row + 1] - C_row_ptr[row], bcol);
            cj = __shfl_sync(0xffffffff, cj, 0);
            if (cj != -1) 
            {
                cj += C_row_ptr[row];
                int valslen = C_tiles[cj].valslen + 1;
                for (int i = lane_id; i < valslen; i += 32) warp_psum[i] = 0;
                TileFormat Bfmt = B_tiles[k].fmt;
                TileIndex* bbits = (TileIndex*)(B_data + B_tiles[k].bits_off);
                MatValue* bval = (MatValue*)(B_data + B_tiles[k].bits_off + B_tiles[k].bitslen * 16);
                TileFormat Cfmt = C_tiles[cj].fmt;
                TileIndex *cbits = (TileIndex *)(C_data + C_tiles[cj].bits_off);
                uint64_t *c_bitmap = C_tiles[cj].bitmap;
                
                __syncwarp();
                kernel_matrix[Afmt][Bfmt][Cfmt](abits, aval, bbits, bval, cbits, warp_psum, c_bitmap, lane_id, vwarp_id, vlane_id, hwarp_id, hlane_id, false);
                __syncwarp();
                MatValue* cval = (MatValue*)(C_data + C_tiles[cj].bits_off + C_tiles[cj].bitslen * 16);
                for (int i = lane_id; i < valslen; i += 32) {
                    if (warp_psum[i] != 0) atomicAdd(cval + i, warp_psum[i]);
                }
            }
        }
    }
}

// double bin_spgemm_call(sfBIN&bin, BaseMatrix *A, BaseMatrix *B, BaseMatrix *C, int*d_check, int*d_index, double*d_value, int table_size)
// {
//     // alloc more shared data for kernels
//     cudaFuncSetAttribute(bin_spgemm_smem<4, 4>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
//     cudaFuncSetAttribute(bin_spgemm_smem<8, 8>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
//     cudaFuncSetAttribute(bin_spgemm_smem<8, 12>, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
//     Timer t;
//     timer_start(t);
//     for (int i = BIN_NUM_N - 1; i >= 0; i--)
//     {
//         int GS = bin.bin_size[i];
//         if (GS > 0)
//         {
//             switch (i)
//             {
//             case 0: // 0~4
//                 {
//                     int BS = 4 << 5;
//                     int bin_offset = bin.bin_offset[i];
//                     bin_spgemm_smem<4, 4><<<GS, BS, 0, bin.stream[i]>>>(
//                         bin.d_row_perm, bin_offset,
//                         A->tile_row_ptr, A->tile_col_idx, A->tiles, A->data,
//                         B->tile_row_ptr, B->tile_col_idx, B->tiles, B->data,
//                         C->tile_row_ptr, C->tile_col_idx, C->tiles, C->data);
//                 }
//                 break;
//             case 1: // 4~8
//                 {
//                     int BS = 8 << 5;
//                     int bin_offset = bin.bin_offset[i];
//                     bin_spgemm_smem<8, 8><<<GS, BS, 0, bin.stream[i]>>>(
//                         bin.d_row_perm, bin_offset,
//                         A->tile_row_ptr, A->tile_col_idx, A->tiles, A->data,
//                         B->tile_row_ptr, B->tile_col_idx, B->tiles, B->data,
//                         C->tile_row_ptr, C->tile_col_idx, C->tiles, C->data);
//                 }
//                 break;
//             case 2: // 8~12
//                 {
//                     int BS = 8 << 5;
//                     int bin_offset = bin.bin_offset[i];
//                     bin_spgemm_smem<8, 12><<<GS, BS, 0, bin.stream[i]>>>(
//                         bin.d_row_perm, bin_offset,
//                         A->tile_row_ptr, A->tile_col_idx, A->tiles, A->data,
//                         B->tile_row_ptr, B->tile_col_idx, B->tiles, B->data,
//                         C->tile_row_ptr, C->tile_col_idx, C->tiles, C->data);
//                 }
//                 break;
//             case 3:
//                 {
//                     int bin_offset = bin.bin_offset[i];
//                     int BS = 8 << 5;
//                     // d_check, d_index, d_value, table_size,
//                     bin_spgemm_gmem<8><<<GS, BS, 0, bin.stream[i]>>>(
//                         bin.d_row_perm, bin_offset, 
//                         A->tile_row_ptr, A->tile_col_idx, A->tiles, A->data,
//                         B->tile_row_ptr, B->tile_col_idx, B->tiles, B->data,
//                         C->tile_row_ptr, C->tile_col_idx, C->tiles, C->data);
//                 }
//                 break;
//             }
//         }
//         // cudaError_t err = cudaGetLastError();
//         // if (err != cudaSuccess)
//         // {
//         //     echo(error, "Error: %s", cudaGetErrorString(err));
//         //     return -1.0;
//         // }
//         // else echo(debug, "bin %d done", i);
//     }
//     cudaDeviceSynchronize();
//     timer_end(t);
//     return timer_duration(t);
// }



__device__ MatIndex lower_bound(const MatIndex *arr, const MatIndex len, const MatIndex val)
{
    MatIndex low = 0, high = len - 1;
    while (low < high)
    {
        MatIndex mid = (low + high) >> 1;
        if (arr[mid] < val) low = mid + 1;
        else high = mid;
    }
    return low;
}

template <int WARPS>
__global__ void spgemm_csr_x_csc_tile_kernel(
    const int m, const int n, const int nnz,
    const MatIndex *A_row_ptr, const MatIndex *A_col_idx, const Tile *A_tiles, const char *A_data,
    const MatIndex *B_col_ptr, const MatIndex *B_row_idx, const Tile *B_tiles, const char *B_data,
    const MatIndex *C_row_ptr, const MatIndex *C_col_idx,       Tile *C_tiles,       char *C_data)
{
    int gwarp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (gwarp_id >= nnz) return;
    int lane_id = threadIdx.x & 31, lwarp_id = threadIdx.x >> 5;
    int vwarp_id = lane_id >> 1, hwarp_id = lane_id >> 4;
    int vlane_id = lane_id & 1, hlane_id = lane_id & 15;

    __shared__ MatValue psum[WARPS * 256];
    double *warp_psum = psum + lwarp_id * 256;
    
    int row;
    if (lane_id == 0) row = lower_bound(C_row_ptr, m + 1, gwarp_id);
    row = __shfl_sync(0xffffffff, row, 0);
    int col = C_col_idx[gwarp_id];

    int astart = A_row_ptr[row];
    int aend = A_row_ptr[row + 1];
    int bstart = B_col_ptr[col];
    int bend = B_col_ptr[col + 1];

    TileFormat Cfmt = C_tiles[gwarp_id].fmt;
    TileIndex *cbits = (TileIndex *)(C_data + C_tiles[gwarp_id].bits_off);
    MatValue *cval = (MatValue *)(C_data + C_tiles[gwarp_id].bits_off + C_tiles[gwarp_id].bitslen * 16);
    uint64_t *c_bitmap = C_tiles[gwarp_id].bitmap;
    int valslen = C_tiles[gwarp_id].valslen + 1;

    for (int i = lane_id; i < 256; i += 32) warp_psum[i] = 0;
    for (int i = astart; i < aend; ++i)
    {
        int acol = A_col_idx[i];
        int bidx;
        if (lane_id == 0) bidx = binary_find_index(B_row_idx + bstart, bend - bstart, acol);
        bidx = __shfl_sync(0xffffffff, bidx, 0);
        if (bidx != -1) {
            bstart = bstart + bidx;

            TileFormat Afmt = A_tiles[i].fmt;
            TileIndex* abits = (TileIndex*)(A_data + A_tiles[i].bits_off);
            MatValue* aval = (MatValue*)(A_data + A_tiles[i].bits_off + A_tiles[i].bitslen * 16);
            TileFormat Bfmt = B_tiles[bstart].fmt;
            TileIndex* bbits = (TileIndex*)(B_data + B_tiles[bstart].bits_off);
            MatValue* bval = (MatValue*)(B_data + B_tiles[bstart].bits_off + B_tiles[bstart].bitslen * 16);
            
            kernel_matrix[Afmt][Bfmt][Cfmt](abits, aval, bbits, bval, cbits, warp_psum, c_bitmap, lane_id, vwarp_id, vlane_id, hwarp_id, hlane_id, false);
            __syncwarp();
        }
    }

    for (int i = lane_id; i < valslen; i += 32)
    {
        if (warp_psum[i] != 0) cval[i] = warp_psum[i];
    }
}

double spgemm_csr_x_csc_tile(BaseMatrix*A, BaseMatrixCSC*B, BaseMatrix*C)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    spgemm_csr_x_csc_tile_kernel<8><<<(C->_nnz + 7) / 8, 256>>>(
        A->_m, B->_n, C->_nnz,
        A->tile_row_ptr, A->tile_col_idx, A->tiles, A->data,
        B->tile_col_ptr, B->tile_row_idx, B->tiles, B->data,
        C->tile_row_ptr, C->tile_col_idx, C->tiles, C->data);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceSynchronize();
    return milliseconds;
}