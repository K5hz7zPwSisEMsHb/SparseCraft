#include <tmatrix/DataStructure/CSR2Tile.cuh>
#include <tmatrix/Calculation/nsparse_asm.cuh>
#include <tmatrix/Utils/msg.h>
#include <tmatrix/Utils/timer.h>
#include <tmatrix/Utils/bitmap_utils.cuh>
#define div_round_up(a, b) (((a) + ((b) - 1)) / (b))

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <tmatrix/Utils/omp_utils.h>
#include <tmatrix/Calculation/spgemm_subkernels.cuh>

#define bitmap8_col_x_row(col, row) (((col) * 0xff) & ((row) * 0x0101010101010101))

__host__ __device__ inline uint64_t bitmap88_x(uint64_t A, uint64_t B)
{
    uint64_t C = 0;

#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        C |= bitmap8_col_x_row(A & 0x0101010101010101, B & 0xff);
        A >>= 1;
        B >>= 8;
    }

    return C;
}

__host__ __device__ void b64_multiply(const uint64_t *A, const uint64_t *B, uint64_t *C)
{
    C[0] = bitmap88_x(A[0], B[0]);
    C[1] = bitmap88_x(A[0], B[1]);
    C[2] = bitmap88_x(A[2], B[0]);
    C[3] = bitmap88_x(A[2], B[1]);

    C[0] |= bitmap88_x(A[1], B[2]);
    C[1] |= bitmap88_x(A[1], B[3]);
    C[2] |= bitmap88_x(A[3], B[2]);
    C[3] |= bitmap88_x(A[3], B[3]);
}

void bitmap256_2_coo(const uint64_t *bitmap, TileIndex *bits)
{
    bits[0] = bitmap_count(bitmap);

#pragma unroll
    for (int i = 0, idx = 1; i < 256; ++i)
    {
        if (bitmap_check(bitmap, i))
        {
            bits[idx] = i;
            idx++;
        }
    }
}

void bitmap256_2_csr(const uint64_t *bitmap, TileIndex *bits)
{
    int nnz = 0;
    bits[0] = 0;

#pragma unroll
    for (MatIndex i = 0; i < TILE_N; ++i)
    {
        uint16_t bitrow = bitmap_get_row(bitmap, i), nzr = __builtin_popcount(bitrow);
        for (MatIndex j = 0; j < nzr; ++j)
        {
            uint16_t lowbit = bitrow & -bitrow, col = __builtin_ctz(lowbit);
            TileIndex_CSR_Set(bits, 2 + TILE_N * 2 + nnz + j, col);
            bitrow ^= lowbit;
        }
        nnz += nzr;
        bits[i + 1] = nnz;
    }
}

void bitmap256_2_ell(const uint64_t *bitmap, TileIndex *bits)
{
    MatIndex max_row_len = 0;
    ushort row_map[16] = {0};

#pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        row_map[i] = bitmap_get_row(bitmap, i);
        MatIndex row_len = __builtin_popcount(row_map[i]);
        max_row_len = std::max(max_row_len, row_len);
    }

    TileIndex_CSR_Set(bits, 0, max_row_len - 1);

#pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        MatIndex row_len = __builtin_popcount(row_map[i]);
        for (MatIndex j = 0; j < row_len; ++j)
        {
            uint16_t lowbit = row_map[i] & -row_map[i], col = __builtin_ctz(lowbit);
            TileIndex_CSR_Set(bits, 1 + i * max_row_len + j, col);
            row_map[i] ^= lowbit;
        }
    }
}

void bitmap256_2_hyb(const uint64_t *bitmap, TileIndex *bits)
{
    MatIndex min_row_len = 17, row_len[16] = {0}, cnt = 0;
    ushort row_map[16] = {0};

#pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        row_map[i] = bitmap_get_row(bitmap, i);
        row_len[i] = __builtin_popcount(row_map[i]);
        min_row_len = std::min(min_row_len, std::max(row_len[i], 1));
    }
    bits[0] = min_row_len;
    MatIndex ell_using_size = (min_row_len * 16 + 1) / 2 + 2;

#pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        MatIndex rest = std::max(row_len[i] - min_row_len, 0);
        for (MatIndex j = 0; j < min_row_len; ++j)
        {
            ushort lowbit = row_map[i] & -row_map[i], col = __builtin_ctz(lowbit);
            TileIndex_CSR_Set(bits, 4 + i * min_row_len + j, col);
            row_map[i] ^= lowbit;
        }

        for (MatIndex j = 0; j < rest; ++j)
        {
            ushort lowbit = row_map[i] & -row_map[i], col = __builtin_ctz(lowbit);
            bits[ell_using_size + cnt] = i << 4 | col;
            cnt++;
            row_map[i] ^= lowbit;
        }
    }
    bits[1] = cnt;
}

void bitmap256_2_drw(const uint64_t *bitmap, TileIndex *bits)
{
    TileIndex cnt = 0;

#pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        if (bitmap_get_row(bitmap, i))
        {
            TileIndex_CSR_Set(bits, cnt + 1, i);
            cnt++;
        }
    }
    TileIndex_CSR_Set(bits, 0, cnt - 1);
}

void bitmap256_2_dcl(const uint64_t *bitmap, TileIndex *bits)
{
    TileIndex cnt = 0;

#pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        if (bitmap_get_col(bitmap, i))
        {
            TileIndex_CSR_Set(bits, cnt + 1, i);
            cnt++;
        }
    }
    TileIndex_CSR_Set(bits, 0, cnt - 1);
}

void bitmap256_2_dns(const uint64_t *bitmap, TileIndex *bits)
{
}

void bitmap256_2_fmt(const uint64_t *bitmap, char *bits, TileFormat fmt)
{
    void (*bitmap2fmt[7])(const uint64_t *bitmap, TileIndex *bits) = {
        bitmap256_2_coo,
        bitmap256_2_csr,
        bitmap256_2_ell,
        bitmap256_2_hyb,
        bitmap256_2_drw,
        bitmap256_2_dcl,
        bitmap256_2_dns};
    bitmap2fmt[fmt](bitmap, (TileIndex *)bits);
}

void init_bin(sfBIN *bin, int M)
{
    for (int i = 0; i < BIN_NUM; i++)
    {
        cudaStreamCreate(&(bin->stream[i]));
    }

    cudaMalloc((void **)&(bin->d_row_perm), sizeof(int) * M);
    cudaMalloc((void **)&(bin->d_row_nz), sizeof(int) * (M + 1));
    cudaMalloc((void **)&(bin->d_max), sizeof(int));
    cudaMalloc((void **)&(bin->d_bin_size), sizeof(int) * BIN_NUM);
    cudaMalloc((void **)&(bin->d_bin_offset), sizeof(int) * BIN_NUM);

    bin->max_intprod = 0;
    bin->max_nz = 0;
    bin->inited = 1;
}

void release_bin(sfBIN bin)
{
    cudaFree(bin.d_row_nz);
    cudaFree(bin.d_row_perm);
    cudaFree(bin.d_max);
    cudaFree(bin.d_bin_size);
    cudaFree(bin.d_bin_offset);
    // free(bin.bin_size);
    // free(bin.bin_offset);
    for (int i = 0; i < BIN_NUM; i++)
    {
        cudaStreamDestroy(bin.stream[i]);
    }
}

__global__ void set_intprod_num(
    int *d_arpt, int *d_acol, const int *__restrict__ d_brpt, int *d_row_intprod, int *d_max_intprod, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M)
        return;

    int nz_per_row = 0;
    for (int j = d_arpt[i]; j < d_arpt[i + 1]; j++)
    {
        nz_per_row += d_brpt[d_acol[j] + 1] - d_brpt[d_acol[j]];
    }

    d_row_intprod[i] = nz_per_row;
    atomicMax(d_max_intprod, nz_per_row);
}

__global__ void set_bin(int *d_row_nz, int *d_bin_size, int *d_max, int M, int min, int mmin)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M)
        return;
    int nz_per_row = d_row_nz[i];

    atomicMax(d_max, nz_per_row);

    int j = 0;
    for (j = 0; j < BIN_NUM - 2; j++)
    {
        if (nz_per_row <= (min << j))
        {
            if (nz_per_row <= (mmin))
            {
                atomicAdd(d_bin_size + j, 1);
            }
            else
            {
                atomicAdd(d_bin_size + j + 1, 1);
            }
            return;
        }
    }
    atomicAdd(d_bin_size + BIN_NUM - 1, 1);
}

__global__ void init_row_perm(int *d_permutation, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= M)
    {
        return;
    }

    d_permutation[i] = i;
}

__global__ void set_row_perm(int *d_bin_size, int *d_bin_offset,
                             int *d_max_row_nz, int *d_row_perm,
                             int M, int min, int mmin)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= M)
    {
        return;
    }

    int nz_per_row = d_max_row_nz[i];
    int dest;

    int j = 0;
    for (j = 0; j < BIN_NUM - 2; j++)
    {
        if (nz_per_row <= (min << j))
        {
            if (nz_per_row <= mmin)
            {
                dest = atomicAdd(d_bin_size + j, 1);
                d_row_perm[d_bin_offset[j] + dest] = i;
            }
            else
            {
                dest = atomicAdd(d_bin_size + j + 1, 1);
                d_row_perm[d_bin_offset[j + 1] + dest] = i;
            }
            return;
        }
    }
    dest = atomicAdd(d_bin_size + BIN_NUM - 1, 1);
    d_row_perm[d_bin_offset[BIN_NUM - 1] + dest] = i;
}

__global__ void set_row_nz_bin_pwarp(const int *d_arpt, const int *d_acol, const Tile *d_atile,
                                     const int *__restrict__ d_brpt,
                                     const int *__restrict__ d_bcol,
                                     const Tile *__restrict__ d_btile,
                                     const int *d_row_perm,
                                     int *d_row_nz,
                                     int bin_offset, int M)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = i / PWARP;
    int tid = i % PWARP;
    int local_rid = rid % (blockDim.x / PWARP);

    int j, k;
    int soffset;
    int acol, bcol, key, hash, adr, nz, old;
    __shared__ int check[IMB_PW_SH_SIZE];

    soffset = local_rid * IMB_PWMIN;

    for (j = tid; j < IMB_PWMIN; j += PWARP)
    {
        check[soffset + j] = -1;
    }
    if (rid >= M)
    {
        return;
    }

    rid = d_row_perm[rid + bin_offset];
    nz = 0;
    for (j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += PWARP)
    {
        acol = ld_gbl_int32(d_acol + j);
        uint64_t a_bitmap[4];
        a_bitmap[0] = d_atile[j].bitmap[0];
        a_bitmap[1] = d_atile[j].bitmap[1];
        a_bitmap[2] = d_atile[j].bitmap[2];
        a_bitmap[3] = d_atile[j].bitmap[3];
        for (k = d_brpt[acol]; k < d_brpt[acol + 1]; k++)
        {
            bcol = d_bcol[k];
            uint64_t bitmap[4] = {0};
            b64_multiply(a_bitmap, d_btile[k].bitmap, bitmap);
            if (bitmap[0] | bitmap[1] | bitmap[2] | bitmap[3])
            {
                key = bcol;
                hash = (bcol * HASH_SCAL) & (IMB_PWMIN - 1);
                adr = soffset + hash;
                while (1)
                {
                    if (check[adr] == key)
                    {
                        break;
                    }
                    else if (check[adr] == -1)
                    {
                        old = atomicCAS(check + adr, -1, key);
                        if (old == -1)
                        {
                            nz++;
                            break;
                        }
                    }
                    else
                    {
                        hash = (hash + 1) & (IMB_PWMIN - 1);
                        adr = soffset + hash;
                    }
                }
            }
        } ////__syncwarp();
    }

    for (j = PWARP / 2; j >= 1; j /= 2)
    {
        nz += __shfl_xor_sync(0xffffffff, nz, j);
    }

    if (tid == 0)
    {
        d_row_nz[rid] = nz;
    }
}

template <int SH_ROW>
__global__ void set_row_nz_bin_each_tb(
    const int *d_arpt, const int *d_acol, const Tile *d_atile,
    const int *__restrict__ d_brpt,
    const int *__restrict__ d_bcol,
    const Tile *__restrict__ d_btile,
    int *d_row_perm, int *d_row_nz,
    int bin_offset, int M)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & 31;
    int wid = threadIdx.x / 32;
    int wnum = blockDim.x / 32;
    int j, k;
    int bcol, key, hash, old;
    int nz, adr;
    int acol;

    __shared__ int check[SH_ROW];
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x)
    {
        check[j] = -1;
    }

    if (rid >= M)
    {
        return;
    }

    __syncthreads();

    nz = 0;
    rid = d_row_perm[rid + bin_offset];
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum)
    {
        acol = ld_gbl_int32(d_acol + j);
        uint64_t a_bitmap[4];
        a_bitmap[0] = d_atile[j].bitmap[0];
        a_bitmap[1] = d_atile[j].bitmap[1];
        a_bitmap[2] = d_atile[j].bitmap[2];
        a_bitmap[3] = d_atile[j].bitmap[3];
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += 32)
        {
            bcol = d_bcol[k];
            uint64_t bitmap[4] = {0};
            b64_multiply(a_bitmap, d_btile[k].bitmap, bitmap);

            if (bitmap[0] | bitmap[1] | bitmap[2] | bitmap[3])
            {
                key = bcol;
                hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
                adr = hash;
                while (1)
                {
                    if (check[adr] == key)
                    {
                        break;
                    }
                    else if (check[adr] == -1)
                    {
                        old = atomicCAS(check + adr, -1, key);
                        if (old == -1)
                        {
                            nz++;
                            break;
                        }
                    }
                    else
                    {
                        hash = (hash + 1) & (SH_ROW - 1);
                        adr = hash;
                    }
                }
            }
        }
    }

    for (j = 32 / 2; j >= 1; j /= 2)
    {
        nz += __shfl_xor_sync(0xffffffff, nz, j);
    }

    __syncthreads();
    if (threadIdx.x == 0)
    {
        check[0] = 0;
    }
    __syncthreads();

    if (tid == 0)
    {
        atomicAdd(check, nz);
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        d_row_nz[rid] = check[0];
    }
}

template <int SH_ROW>
__global__ void set_row_nz_bin_each_tb_large(
    const int *d_arpt, const int *d_acol, const Tile *d_atile,
    const int *__restrict__ d_brpt,
    const int *__restrict__ d_bcol,
    const Tile *__restrict__ d_btile,
    int *d_row_perm, int *d_row_nz,
    int *d_fail_count, int *d_fail_perm,
    int bin_offset, int M)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & 31;
    int wid = threadIdx.x / 32;
    int wnum = blockDim.x / 32;
    int j, k;
    int bcol, key, hash, old;
    int adr;
    int acol;

    __shared__ int check[SH_ROW];
    __shared__ int snz[1];
    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x)
    {
        check[j] = -1;
    }
    if (threadIdx.x == 0)
    {
        snz[0] = 0;
    }

    if (rid >= M)
    {
        return;
    }

    __syncthreads();

    rid = d_row_perm[rid + bin_offset];
    int count = 0;
    int border = SH_ROW >> 1;
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum)
    {
        acol = ld_gbl_int32(d_acol + j);
        uint64_t a_bitmap[4];
        a_bitmap[0] = d_atile[j].bitmap[0];
        a_bitmap[1] = d_atile[j].bitmap[1];
        a_bitmap[2] = d_atile[j].bitmap[2];
        a_bitmap[3] = d_atile[j].bitmap[3];
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += 32)
        {
            bcol = d_bcol[k];
            uint64_t bitmap[4] = {0};
            b64_multiply(a_bitmap, d_btile[k].bitmap, bitmap);
            if (bitmap[0] | bitmap[1] | bitmap[2] | bitmap[3])
            {
                key = bcol;
                hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
                adr = hash;
                while (count < border && snz[0] < border)
                {
                    if (check[adr] == key)
                    {
                        break;
                    }
                    else if (check[adr] == -1)
                    {
                        old = atomicCAS(check + adr, -1, key);
                        if (old == -1)
                        {
                            atomicAdd(snz, 1);
                            break;
                        }
                    }
                    else
                    {
                        hash = (hash + 1) & (SH_ROW - 1);
                        adr = hash;
                        count++;
                    }
                }
            }
            if (count >= border || snz[0] >= border)
            {
                break;
            }
        }
        if (count >= border || snz[0] >= border)
        {
            break;
        }
    }

    __syncthreads();
    if (count >= border || snz[0] >= border)
    {
        if (threadIdx.x == 0)
        {
            int d = atomicAdd(d_fail_count, 1);
            d_fail_perm[d] = rid;
        }
    }
    else
    {
        if (threadIdx.x == 0)
        {
            d_row_nz[rid] = snz[0];
        }
    }
}

__global__ void set_row_nz_bin_each_gl(
    const int *d_arpt, const int *d_acol, const Tile *d_atile,
    const int *__restrict__ d_brpt,
    const int *__restrict__ d_bcol,
    const Tile *__restrict__ d_btile,
    const int *d_row_perm,
    int *d_row_nz, int *d_check,
    int max_row_nz, int bin_offset, int M)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & 31;
    int wid = threadIdx.x / 32;
    int wnum = blockDim.x / 32;
    int j, k;
    int bcol, key, hash, old;
    int nz, adr;
    int acol;
    int offset = rid * max_row_nz;

    __shared__ int snz[1];
    if (threadIdx.x == 0)
    {
        snz[0] = 0;
    }
    __syncthreads();

    if (rid >= M)
    {
        return;
    }

    nz = 0;
    rid = d_row_perm[rid + bin_offset];
    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum)
    {
        acol = ld_gbl_int32(d_acol + j);
        uint64_t a_bitmap[4];
        a_bitmap[0] = d_atile[j].bitmap[0];
        a_bitmap[1] = d_atile[j].bitmap[1];
        a_bitmap[2] = d_atile[j].bitmap[2];
        a_bitmap[3] = d_atile[j].bitmap[3];
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += 32)
        {
            bcol = d_bcol[k];
            uint64_t bitmap[4] = {0};
            b64_multiply(a_bitmap, d_btile[k].bitmap, bitmap);
            if (bitmap[0] | bitmap[1] | bitmap[2] | bitmap[3])
            {
                key = bcol;
                hash = (bcol * HASH_SCAL) % max_row_nz;
                adr = offset + hash;
                while (1)
                {
                    if (d_check[adr] == key)
                    {
                        break;
                    }
                    else if (d_check[adr] == -1)
                    {
                        old = atomicCAS(d_check + adr, -1, key); ////__syncwarp();
                        if (old == -1)
                        {
                            nz++;
                            break;
                        }
                    }
                    else
                    {
                        hash = (hash + 1) % max_row_nz;
                        adr = offset + hash;
                    }
                }
            }
        }
    }

    for (j = 32 / 2; j >= 1; j /= 2)
    {
        nz += __shfl_xor_sync(0xffffffff, nz, j);
    }

    if (tid == 0)
    {
        atomicAdd(snz, nz);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        d_row_nz[rid] = snz[0];
    }
}

double set_row_nnz(sfBIN *bin, int *d_arpt, int *d_acol, Tile *d_atile, int *d_brpt, int *d_bcol, Tile *d_btile, int *d_crpt, int M, int *nnz)
{
    // struct timeval tv;
    int i;
    int GS, BS;
    Timer t;
    timer_start(t);
    for (i = BIN_NUM - 1; i >= 0; i--)
    {
        if (bin->bin_size[i] > 0)
        {
            switch (i)
            {
            case 0:
                BS = 256;
                GS = div_round_up(bin->bin_size[i] * PWARP, BS);
                set_row_nz_bin_pwarp<<<GS, BS, 0, bin->stream[i]>>>(d_arpt, d_acol, d_atile,
                                                                    d_brpt, d_bcol, d_btile,
                                                                    bin->d_row_perm,
                                                                    bin->d_row_nz,
                                                                    bin->bin_offset[i],
                                                                    bin->bin_size[i]);
                break;
            case 1:
                BS = 64;
                GS = bin->bin_size[i];
                // echo(debug, "bin %d: %d\n", i, GS);
                set_row_nz_bin_each_tb<512><<<GS, BS, 0, bin->stream[i]>>>(d_arpt, d_acol, d_atile, d_brpt, d_bcol, d_btile,
                                                                           bin->d_row_perm, bin->d_row_nz,
                                                                           bin->bin_offset[i], bin->bin_size[i]);
                break;
            case 2:
                BS = 128;
                GS = bin->bin_size[i];
                set_row_nz_bin_each_tb<1024><<<GS, BS, 0, bin->stream[i]>>>(
                    d_arpt, d_acol, d_atile, d_brpt, d_bcol, d_btile,
                    bin->d_row_perm, bin->d_row_nz,
                    bin->bin_offset[i], bin->bin_size[i]);
                break;
            case 3:
                BS = 256;
                GS = bin->bin_size[i];
                set_row_nz_bin_each_tb<2048><<<GS, BS, 0, bin->stream[i]>>>(d_arpt, d_acol, d_atile, d_brpt, d_bcol, d_btile,
                                                                            bin->d_row_perm, bin->d_row_nz,
                                                                            bin->bin_offset[i], bin->bin_size[i]);
                break;
            case 4:
                BS = 512;
                GS = bin->bin_size[i];
                set_row_nz_bin_each_tb<4096><<<GS, BS, 0, bin->stream[i]>>>(d_arpt, d_acol, d_atile, d_brpt, d_bcol, d_btile,
                                                                            bin->d_row_perm, bin->d_row_nz,
                                                                            bin->bin_offset[i], bin->bin_size[i]);
                break;
            case 5:
                BS = 512;
                GS = bin->bin_size[i];
                set_row_nz_bin_each_tb<8192><<<GS, BS, 0, bin->stream[i]>>>(d_arpt, d_acol, d_atile, d_brpt, d_bcol, d_btile,
                                                                            bin->d_row_perm, bin->d_row_nz,
                                                                            bin->bin_offset[i], bin->bin_size[i]);
                break;
            case 6:
            {
                int fail_count;
                int *d_fail_count, *d_fail_perm;
                fail_count = 0;
                cudaMalloc((void **)&d_fail_count, sizeof(int));
                cudaMalloc((void **)&d_fail_perm, sizeof(int) * bin->bin_size[i]);

                cudaMemcpy(d_fail_count, &fail_count, sizeof(int), cudaMemcpyHostToDevice);
                BS = 512;
                GS = bin->bin_size[i];
                set_row_nz_bin_each_tb_large<8192><<<GS, BS, 0, bin->stream[i]>>>(d_arpt, d_acol, d_atile, d_brpt, d_bcol, d_btile,
                                                                                  bin->d_row_perm, bin->d_row_nz,
                                                                                  d_fail_count, d_fail_perm,
                                                                                  bin->bin_offset[i], bin->bin_size[i]);
                cudaMemcpy(&fail_count, d_fail_count, sizeof(int), cudaMemcpyDeviceToHost);
                if (fail_count > 0)
                {
                    int max_row_nz = bin->max_intprod;
                    size_t table_size = (size_t)max_row_nz * fail_count;
                    int *d_check;
                    cudaMalloc((void **)&(d_check), sizeof(int) * table_size);
                    cudaMemset(d_check, -1, sizeof(int) * table_size);

                    BS = 512;
                    GS = div_round_up(table_size, BS);
                    // init_check<<<GS, BS, 0, bin->stream[i]>>>(d_check, table_size);
                    GS = bin->bin_size[i];
                    set_row_nz_bin_each_gl<<<GS, BS, 0, bin->stream[i]>>>(d_arpt, d_acol, d_atile, d_brpt, d_bcol, d_btile,
                                                                          d_fail_perm, bin->d_row_nz, d_check,
                                                                          max_row_nz, 0, fail_count);
                    cudaFree(d_check);
                }
                cudaFree(d_fail_count);
                cudaFree(d_fail_perm);
                // cudaDeviceSynchronize();
            }
            break;
            default:
                exit(0);
            }
            // cudaDeviceSynchronize();
            // cudaError_t err = cudaGetLastError();
            // if (err != cudaSuccess)
            // {
            //     echo(error, "bin %d: %s", i, cudaGetErrorString(err));
            //     exit(1);
            // }
        }
        // echo(debug, "bin %d done\n", i);
    }
    cudaDeviceSynchronize();
    thrust::exclusive_scan(thrust::device, bin->d_row_nz, bin->d_row_nz + M + 1, d_crpt, 0);
    timer_end(t);
    cudaMemcpy(nnz, d_crpt + M, sizeof(int), cudaMemcpyDeviceToHost);
    return timer_duration(t);
}

__global__ void init_check(int *d_check, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nz)
    {
        return;
    }
    d_check[i] = -1;
}

double set_max_bin(sfBIN *bin, int *d_arpt, int *d_acol, int *dbrpt, int M)
{
    int i;
    int GS, BS;
    double used_time = 0;
    Timer t;

#pragma omp parallel for
    for (int i = 0; i < BIN_NUM; ++i)
    {
        bin->bin_size[i] = 0;
        bin->bin_offset[i] = 0;
    }
    cudaMemcpy(bin->d_bin_size, bin->bin_size, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(bin->d_max, &(bin->max_intprod), sizeof(int), cudaMemcpyHostToDevice);

    BS = 1024;
    GS = div_round_up(M, BS);
    timer_start(t);
    set_intprod_num<<<GS, BS>>>(d_arpt, d_acol, dbrpt, bin->d_row_nz, bin->d_max, M);
    cudaDeviceSynchronize();
    timer_end(t);
    used_time += timer_duration(t);

    cudaMemcpy(&(bin->max_intprod), bin->d_max, sizeof(int), cudaMemcpyDeviceToHost);
    if (bin->max_intprod > 4)
    {
        used_time += set_min_bin(bin, M);
    }
    else
    {
        bin->bin_size[0] = M;
        for (i = 1; i < BIN_NUM; i++)
        {
            bin->bin_size[i] = 0;
        }
        bin->bin_offset[0] = 0;
        for (i = 1; i < BIN_NUM; i++)
        {
            bin->bin_offset[i] = M;
        }
        BS = 1024;
        GS = div_round_up(M, BS);

        timer_start(t);
        init_row_perm<<<GS, BS>>>(bin->d_row_perm, M);
        cudaDeviceSynchronize();
        timer_end(t);
        used_time += timer_duration(t);
    }
    return used_time;
}

double set_min_bin(sfBIN *bin, int M)
{
    int i;
    int GS, BS;
    double used_time;
    Timer t;

    for (i = 0; i < BIN_NUM; i++)
    {
        bin->bin_size[i] = 0;
        bin->bin_offset[i] = 0;
    }

    cudaMemcpy(bin->d_bin_size, bin->bin_size, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);
    cudaMemcpy(bin->d_max, &(bin->max_nz), sizeof(int), cudaMemcpyHostToDevice);

    BS = 1024;
    GS = div_round_up(M, BS);
    timer_start(t);
    set_bin<<<GS, BS>>>(bin->d_row_nz, bin->d_bin_size, bin->d_max, M, B_MIN, B_PWMIN);
    cudaDeviceSynchronize();
    timer_end(t);
    used_time = timer_duration(t);

    cudaMemcpy(&(bin->max_nz), bin->d_max, sizeof(int), cudaMemcpyDeviceToHost);
    if (bin->max_nz > B_PWMIN)
    {
        cudaMemcpy(bin->bin_size, bin->d_bin_size, sizeof(int) * BIN_NUM, cudaMemcpyDeviceToHost);
        cudaMemcpy(bin->d_bin_size, bin->bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);

        for (i = 0; i < BIN_NUM - 1; i++)
        {
            bin->bin_offset[i + 1] = bin->bin_offset[i] + bin->bin_size[i];
        }
        cudaMemcpy(bin->d_bin_offset, bin->bin_offset, sizeof(int) * BIN_NUM, cudaMemcpyHostToDevice);

        timer_start(t);
        set_row_perm<<<GS, BS>>>(bin->d_bin_size, bin->d_bin_offset, bin->d_row_nz, bin->d_row_perm, M, B_MIN, B_PWMIN);
        cudaDeviceSynchronize();
        timer_end(t);
        used_time += timer_duration(t);
    }

    else
    {
        bin->bin_size[0] = M;
        for (i = 1; i < BIN_NUM; i++)
        {
            bin->bin_size[i] = 0;
        }
        bin->bin_offset[0] = 0;
        for (i = 1; i < BIN_NUM; i++)
        {
            bin->bin_offset[i] = M;
        }
        BS = 1024;
        GS = div_round_up(M, BS);

        timer_start(t);
        init_row_perm<<<GS, BS>>>(bin->d_row_perm, M);
        cudaDeviceSynchronize();
        timer_end(t);
        used_time += timer_duration(t);
    }
    return used_time;
}

__global__ void set_bin_for_numeric(int *d_row_nz, int *d_bin_size, int *d_max, int M, int min)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M)
    {
        return;
    }
    int nz_per_row = d_row_nz[i + 1] - d_row_nz[i];
    atomicMax(d_max, nz_per_row);
    int idx = nz_per_row / min;
    if (idx < BIN_NUM_N - 1)
    {
        atomicAdd(d_bin_size + idx, 1);
    }
    else
    {
        atomicAdd(d_bin_size + BIN_NUM_N - 1, 1);
    }
}

__global__ void set_row_perm_for_numeric(int *d_bin_size, int *d_bin_offset, int *d_row_nz, int *d_row_perm, int M, int min)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= M)
    {
        return;
    }

    int nz_per_row = d_row_nz[i + 1] - d_row_nz[i];
    int bin = nz_per_row / min;
    bin = (bin < BIN_NUM_N - 1) ? bin : BIN_NUM_N - 1;
    int dest = atomicAdd(d_bin_size + bin, 1);
    d_row_perm[d_bin_offset[bin] + dest] = i;
}

double set_min_bin_for_numeric(sfBIN *bin, int M)
{
    int i;
    int GS, BS;
    double used_time;
    Timer t;

    memset(bin->bin_size, 0, sizeof(int) * BIN_NUM_N);
    memset(bin->bin_offset, 0, sizeof(int) * BIN_NUM_N);
    cudaMemset(bin->d_bin_size, 0, sizeof(int) * BIN_NUM_N);
    cudaMemset(bin->d_max, 0, sizeof(int));

    BS = 1024;
    GS = div_round_up(M, BS);
    timer_start(t);
    set_bin_for_numeric<<<GS, BS>>>(bin->d_row_nz, bin->d_bin_size, bin->d_max, M, 4);
    cudaDeviceSynchronize();
    timer_end(t);
    used_time = timer_duration(t);

    cudaMemcpy(&(bin->max_nz), bin->d_max, sizeof(int), cudaMemcpyDeviceToHost);
    if (bin->max_nz > 4)
    {
        cudaMemcpy(bin->bin_size, bin->d_bin_size, sizeof(int) * BIN_NUM_N, cudaMemcpyDeviceToHost);
        cudaMemcpy(bin->d_bin_size, bin->bin_offset, sizeof(int) * BIN_NUM_N, cudaMemcpyHostToDevice);

        for (i = 0; i < BIN_NUM_N - 1; i++)
        {
            bin->bin_offset[i + 1] = bin->bin_offset[i] + bin->bin_size[i];
        }
        cudaMemcpy(bin->d_bin_offset, bin->bin_offset, sizeof(int) * BIN_NUM_N, cudaMemcpyHostToDevice);

        timer_start(t);
        set_row_perm_for_numeric<<<GS, BS>>>(bin->d_bin_size, bin->d_bin_offset, bin->d_row_nz, bin->d_row_perm, M, 4);
        cudaDeviceSynchronize();
        timer_end(t);
        used_time += timer_duration(t);
    }
    else
    {
        bin->bin_size[0] = M;
        for (i = 1; i < BIN_NUM; i++)
        {
            bin->bin_size[i] = 0;
        }
        bin->bin_offset[0] = 0;
        for (i = 1; i < BIN_NUM; i++)
        {
            bin->bin_offset[i] = M;
        }
        BS = 1024;
        GS = div_round_up(M, BS);

        timer_start(t);
        init_row_perm<<<GS, BS>>>(bin->d_row_perm, M);
        cudaDeviceSynchronize();
        timer_end(t);
        used_time += timer_duration(t);
    }
    return used_time;
}

__global__ void calculate_value_col_bin_pwarp(const int *d_arpt,
                                              const int *d_acol,
                                              const Tile *d_aval,
                                              const int *__restrict__ d_brpt,
                                              const int *__restrict__ d_bcol,
                                              const Tile *__restrict__ d_bval,
                                              int *d_crpt,
                                              int *d_ccol,
                                              Tile *d_cval,
                                              const int *d_row_perm,
                                              int *d_nz,
                                              int bin_offset,
                                              int bin_size)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int rid = i / PWARP;
    int tid = i % PWARP;
    int local_rid = rid % (blockDim.x / PWARP);
    int j;
    __shared__ int shared_check[B_PW_SH_SIZE];
    __shared__ uint64_t shared_value[B_PW_SH_SIZE * 4];

    int soffset = local_rid * (B_PWMIN);

    for (j = tid; j < (B_PWMIN); j += PWARP)
    {
        shared_check[soffset + j] = -1;
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            shared_value[soffset * 4 + k] = 0;
        }
    }

    if (rid >= bin_size)
    {
        return;
    }
    rid = d_row_perm[rid + bin_offset];

    if (tid == 0)
    {
        d_nz[rid] = 0;
    }
    int k;
    int acol, bcol, hash, key, adr;
    int offset = d_crpt[rid];
    int old, index;

    for (j = d_arpt[rid] + tid; j < d_arpt[rid + 1]; j += PWARP)
    {
        acol = ld_gbl_int32(d_acol + j);
        uint64_t bitmapA[4];
        bitmapA[0] = d_aval[j].bitmap[0];
        bitmapA[1] = d_aval[j].bitmap[1];
        bitmapA[2] = d_aval[j].bitmap[2];
        bitmapA[3] = d_aval[j].bitmap[3];
        for (k = d_brpt[acol]; k < d_brpt[acol + 1]; k++)
        {
            bcol = d_bcol[k];
            uint64_t bitmap[4] = {0};
            b64_multiply(bitmapA, d_bval[k].bitmap, bitmap);
            if (bitmap[0] | bitmap[1] | bitmap[2] | bitmap[3])
            {
                key = bcol;
                hash = (bcol * HASH_SCAL) & ((B_PWMIN)-1);
                adr = soffset + hash;
                while (1)
                {
                    if (shared_check[adr] == key)
                    {
                        int *dst_ptr = (int *)&shared_value[adr * 4];
                        int *src_ptr = (int *)bitmap;
#pragma unroll
                        for (int k = 0; k < 8; ++k)
                        {
                            atomicOr(dst_ptr + k, src_ptr[k]);
                        }
                        break;
                    }
                    else if (shared_check[adr] == -1)
                    {
                        old = atomicCAS(shared_check + adr, -1, key);
                        if (old == -1)
                        {
                            int *dst_ptr = (int *)&shared_value[adr * 4];
                            int *src_ptr = (int *)bitmap;
#pragma unroll
                            for (int k = 0; k < 8; ++k)
                            {
                                atomicOr(dst_ptr + k, src_ptr[k]);
                            }
                            break;
                        }
                    }
                    else
                    {
                        hash = (hash + 1) & ((B_PWMIN)-1);
                        adr = soffset + hash;
                    }
                }
            }
        }
    }

    for (j = tid; j < (B_PWMIN); j += PWARP)
    {
        if (shared_check[soffset + j] != -1)
        {
            index = atomicAdd(d_nz + rid, 1);
            int dst_adr = soffset + index;
            int src_adr = soffset + j;
            shared_check[dst_adr] = shared_check[src_adr];

#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                shared_value[dst_adr * 4 + k] = shared_value[src_adr * 4 + k];
            }
        }
    }
    int nz = d_nz[rid];
    // Sorting for shared data
    int count, target;
    for (j = tid; j < nz; j += PWARP)
    {
        target = shared_check[soffset + j];
        count = 0;
        for (k = 0; k < nz; k++)
        {
            count += (unsigned int)(shared_check[soffset + k] - target) >> 31;
        }
        int dst_adr = offset + count;
        int src_adr = soffset + j;
        d_ccol[dst_adr] = shared_check[src_adr];

#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            d_cval[dst_adr].bitmap[k] = shared_value[src_adr * 4 + k];
        }
    }
}

template <int SH_ROW>
__global__ void calculate_value_col_bin_each_tb(const int *d_arpt,
                                                const int *d_acol,
                                                const Tile *d_aval,
                                                const int *__restrict__ d_brpt,
                                                const int *__restrict__ d_bcol,
                                                const Tile *__restrict__ d_bval,
                                                int *d_crpt,
                                                int *d_ccol,
                                                Tile *d_cval,
                                                const int *d_row_perm,
                                                int *d_nz,
                                                int bin_offset,
                                                int bin_size)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & 31;
    int wid = threadIdx.x / 31;
    int wnum = blockDim.x / 31;
    int j;
    __shared__ int shared_check[SH_ROW];
    __shared__ uint64_t shared_value[SH_ROW * 4];

    for (j = threadIdx.x; j < SH_ROW; j += blockDim.x)
    {
        shared_check[j] = -1;
    }

    for (j = threadIdx.x; j < SH_ROW * 4; j += blockDim.x)
    {
        shared_value[j] = 0;
    }

    if (rid >= bin_size)
    {
        return;
    }

    rid = d_row_perm[rid + bin_offset];

    if (threadIdx.x == 0)
    {
        d_nz[rid] = 0;
    }
    __syncthreads();

    int acol;
    int k;
    int bcol, hash, key;
    int offset = d_crpt[rid];
    int old, index;
    //    real aval, bval;

    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum)
    {
        acol = ld_gbl_int32(d_acol + j);
        uint64_t bitmapA[4];
        bitmapA[0] = d_aval[j].bitmap[0];
        bitmapA[1] = d_aval[j].bitmap[1];
        bitmapA[2] = d_aval[j].bitmap[2];
        bitmapA[3] = d_aval[j].bitmap[3];
        //        aval = ld_gbl_real(d_aval + j);
        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += 32)
        {
            bcol = d_bcol[k];
            uint64_t bitmap[4] = {0};
            b64_multiply(bitmapA, d_bval[k].bitmap, bitmap);
            if (bitmap[0] | bitmap[1] | bitmap[2] | bitmap[3])
            {
                key = bcol;
                hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
                while (1)
                {
                    if (shared_check[hash] == key)
                    {
                        int *dst_ptr = (int *)&shared_value[hash * 4];
                        int *src_ptr = (int *)bitmap;
#pragma unroll
                        for (int k = 0; k < 8; ++k)
                        {
                            atomicOr(dst_ptr + k, src_ptr[k]);
                        }
                        break;
                    }
                    else if (shared_check[hash] == -1)
                    {
                        old = atomicCAS(shared_check + hash, -1, key);
                        if (old == -1)
                        {
                            int *dst_ptr = (int *)&shared_value[hash * 4];
                            int *src_ptr = (int *)bitmap;
#pragma unroll
                            for (int k = 0; k < 8; ++k)
                            {
                                atomicOr(dst_ptr + k, src_ptr[k]);
                            }
                            break;
                        }
                    }
                    else
                    {
                        hash = (hash + 1) & (SH_ROW - 1);
                    }
                }
            }

            // key = bcol;
            // hash = (bcol * HASH_SCAL) & (SH_ROW - 1);
            // while (1)
            // {
            //     if (shared_check[hash] == key)
            //     {
            //         break;
            //     }
            //     else if (shared_check[hash] == -1)
            //     {
            //         old = atomicCAS(shared_check + hash, -1, key);
            //         if (old == -1)
            //         {
            //             break;
            //         }
            //     }
            //     else
            //     {
            //         hash = (hash + 1) & (SH_ROW - 1);
            //     }
            // }
        }
    }

    __syncthreads();
    if (threadIdx.x < 32)
    {
        for (j = tid; j < SH_ROW; j += 32)
        {
            if (shared_check[j] != -1)
            {
                index = atomicAdd(d_nz + rid, 1);
                shared_check[index] = shared_check[j];
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    shared_value[index * 4 + k] = shared_value[j * 4 + k];
                }
                // if (rid == 0 && j == 0)
                // {
                //     printf("index: %d, shared_check: %d, shared_value: %016llx %016llx %016llx %016llx\n", index, shared_check[index], shared_value[index], shared_value[index + 1], shared_value[index + 2], shared_value[index + 3]);
                // }
            }
        }
    }
    __syncthreads();
    int nz = d_nz[rid];
    /* Sorting for shared data */
    int count, target;
    for (j = threadIdx.x; j < nz; j += blockDim.x)
    {
        target = shared_check[j];
        count = 0;
        for (k = 0; k < nz; k++)
        {
            count += (unsigned int)(shared_check[k] - target) >> 31;
        }
        d_ccol[offset + count] = shared_check[j];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            d_cval[offset + count].bitmap[k] = shared_value[j * 4 + k];
        }
    }
}

__global__ void calculate_value_col_bin_each_gl(const int *d_arpt,
                                                const int *d_acol,
                                                const Tile *d_aval,
                                                const int *__restrict__ d_brpt,
                                                const int *__restrict__ d_bcol,
                                                const Tile *__restrict__ d_bval,
                                                int *d_crpt,
                                                int *d_ccol,
                                                Tile *d_cval,
                                                const int *d_row_perm,
                                                int *d_nz,
                                                int *d_check,
                                                uint64_t *d_value,
                                                int max_row_nz,
                                                int bin_offset,
                                                int M)
{
    int rid = blockIdx.x;
    int tid = threadIdx.x & 31;
    int wid = threadIdx.x / 32;
    int wnum = blockDim.x / 32;
    int j;

    if (rid >= M)
    {
        return;
    }

    int doffset = rid * max_row_nz;

    rid = d_row_perm[rid + bin_offset];

    if (threadIdx.x == 0)
    {
        d_nz[rid] = 0;
    }
    __syncthreads();

    int acol;
    int k;
    int bcol, hash, key, adr;
    int offset = d_crpt[rid];
    int old, index;
    //    real aval, bval;

    for (j = d_arpt[rid] + wid; j < d_arpt[rid + 1]; j += wnum)
    {
        acol = ld_gbl_int32(d_acol + j);
        uint64_t bitmapA[4];
        bitmapA[0] = d_aval[j].bitmap[0];
        bitmapA[1] = d_aval[j].bitmap[1];
        bitmapA[2] = d_aval[j].bitmap[2];
        bitmapA[3] = d_aval[j].bitmap[3];

        for (k = d_brpt[acol] + tid; k < d_brpt[acol + 1]; k += 32)
        {
            bcol = d_bcol[k];
            uint64_t bitmap[4] = {0};
            b64_multiply(bitmapA, d_bval[k].bitmap, bitmap);
            if (bitmap[0] | bitmap[1] | bitmap[2] | bitmap[3])
            {
                key = bcol;
                hash = (bcol * HASH_SCAL) % max_row_nz;
                adr = doffset + hash;
                // int cnt = 0;
                while (1)
                {
                    if (d_check[adr] == key)
                    {
                        int *dst_ptr = (int *)&d_value[adr * 4];
                        int *src_ptr = (int *)bitmap;
#pragma unroll
                        for (int k = 0; k < 8; ++k)
                        {
                            atomicOr(dst_ptr + k, src_ptr[k]);
                        }
                        break;
                    }
                    else if (d_check[adr] == -1)
                    {
                        old = atomicCAS(d_check + adr, -1, key);
                        if (old == -1)
                        {
                            int *dst_ptr = (int *)&d_value[adr * 4];
                            int *src_ptr = (int *)bitmap;
#pragma unroll
                            for (int k = 0; k < 8; ++k)
                            {
                                atomicOr(dst_ptr + k, src_ptr[k]);
                            }
                            break;
                        }
                    }
                    else
                    {
                        hash = (hash + 1) % max_row_nz;
                        adr = doffset + hash;
                    }
                }
            }
        }
    }

    __syncthreads();
    if (threadIdx.x < 32)
    {
        for (j = tid; j < max_row_nz; j += 32)
        {
            if (d_check[doffset + j] != -1)
            {
                index = atomicAdd(d_nz + rid, 1);
                d_check[doffset + index] = d_check[doffset + j];
#pragma unroll
                for (int k = 0; k < 4; ++k)
                {
                    d_value[(doffset + index) * 4 + k] = d_value[(doffset + j) * 4 + k];
                }
            }
        }
    }
    __syncthreads();
    int nz = d_nz[rid];

    /* Sorting for shared data */
    int count, target;
    for (j = threadIdx.x; j < nz; j += blockDim.x)
    {
        target = d_check[doffset + j];
        count = 0;
        for (k = 0; k < nz; k++)
        {
            count += (unsigned int)(d_check[doffset + k] - target) >> 31;
        }
        d_ccol[offset + count] = d_check[doffset + j];
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            d_cval[offset + count].bitmap[k] = d_value[(doffset + j) * 4 + k];
        }
    }
}

double calculate_value_col_bin(int *d_arpt, int *d_acol, Tile *d_tileA,
                             int *d_brpt, int *d_bcol, Tile *d_tileB,
                             int *d_crpt, int *d_ccol, Tile *d_tileC,
                             sfBIN *bin,
                             int M, int N)
{

    // struct timeval tv;
    int i;
    int GS, BS;
    Timer t;
    timer_start(t);
    for (i = BIN_NUM - 1; i >= 0; i--)
    {
        if (bin->bin_size[i] > 0)
        {
            switch (i)
            {
            case 0:
                // BS = 256;
                // GS = div_round_up(bin->bin_size[i] * PWARP, BS);

                // calculate_value_col_bin_pwarp<<<GS, BS, 0, bin->stream[i]>>>(d_arpt, d_acol, d_tileA,
                //                                                              d_brpt, d_bcol, d_tileB,
                //                                                              d_crpt, d_ccol, d_tileC,
                //                                                              bin->d_row_perm, bin->d_row_nz,
                //                                                              bin->bin_offset[i], bin->bin_size[i]);
                // break;
            case 1:
                BS = 64;
                GS = bin->bin_size[i];

                calculate_value_col_bin_each_tb<256><<<GS, BS, 0, bin->stream[i]>>>(d_arpt, d_acol, d_tileA,
                                                                                    d_brpt, d_bcol, d_tileB,
                                                                                    d_crpt, d_ccol, d_tileC,
                                                                                    bin->d_row_perm, bin->d_row_nz,
                                                                                    bin->bin_offset[i], bin->bin_size[i]);
                break;
            case 2:
                BS = 128;
                GS = bin->bin_size[i];

                calculate_value_col_bin_each_tb<512><<<GS, BS, 0, bin->stream[i]>>>(d_arpt, d_acol, d_tileA,
                                                                                    d_brpt, d_bcol, d_tileB,
                                                                                    d_crpt, d_ccol, d_tileC,
                                                                                    bin->d_row_perm, bin->d_row_nz,
                                                                                    bin->bin_offset[i], bin->bin_size[i]);
                break;
            case 3:
                BS = 256;
                GS = bin->bin_size[i];

                calculate_value_col_bin_each_tb<1024><<<GS, BS, 0, bin->stream[i]>>>(d_arpt, d_acol, d_tileA,
                                                                                     d_brpt, d_bcol, d_tileB,
                                                                                     d_crpt, d_ccol, d_tileC,
                                                                                     bin->d_row_perm, bin->d_row_nz,
                                                                                     bin->bin_offset[i], bin->bin_size[i]);
                break;
            case 4:
                // BS = 512;
                // GS = bin->bin_size[i];

                // calculate_value_col_bin_each_tb<2048><<<GS, BS, 0, bin->stream[i]>>>(d_arpt, d_acol, d_tileA,
                //                                                                      d_brpt, d_bcol, d_tileB,
                //                                                                      d_crpt, d_ccol, d_tileC,
                //                                                                      bin->d_row_perm, bin->d_row_nz,
                //                                                                      bin->bin_offset[i], bin->bin_size[i]);
                // break;
            case 5:
                // BS = 1024;
                // GS = bin->bin_size[i];

                // calculate_value_col_bin_each_tb<4096><<<GS, BS, 0, bin->stream[i]>>>(d_arpt, d_acol,
                //                                                                      d_brpt, d_bcol,
                //                                                                      d_crpt, d_ccol,
                //                                                                      bin->d_row_perm, bin->d_row_nz,
                //                                                                      bin->bin_offset[i], bin->bin_size[i]);
                // break;
            case 6:
            {
                int max_row_nz = N * 2;
                int table_size = max_row_nz * bin->bin_size[i];
                int *d_check;
                uint64_t *d_bitmap;
                cudaMalloc((void **)&(d_check), sizeof(int) * table_size);
                cudaMalloc((void **)&(d_bitmap), sizeof(uint64_t) * 4 * table_size);
                cudaMemset(d_bitmap, 0, sizeof(uint64_t) * 4 * table_size);
                cudaMemset(d_check, -1, sizeof(int) * table_size);
                BS = 512;
                // GS = div_round_up(table_size, BS);
                // init_check<<<GS, BS, 0, bin->stream[i]>>>(d_check, table_size);
                GS = bin->bin_size[i];
                calculate_value_col_bin_each_gl<<<GS, BS, 0, bin->stream[i]>>>(d_arpt, d_acol, d_tileA,
                                                                               d_brpt, d_bcol, d_tileB,
                                                                               d_crpt, d_ccol, d_tileC,
                                                                               bin->d_row_perm, bin->d_row_nz,
                                                                               d_check, d_bitmap, max_row_nz,
                                                                               bin->bin_offset[i], bin->bin_size[i]);
                cudaDeviceSynchronize();
                cudaFree(d_check);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess)
                {
                    printf("Error: %s\n", cudaGetErrorString(err));
                }
            }
            break;
            default:
                exit(0);
            }
        }
    }
    cudaDeviceSynchronize();
    timer_end(t);
    return timer_duration(t);
}

__global__ void tile_spgemm_step1_cuda_spa_kernel(int *d_blkrowptrA, int *d_blkcolidxA, Tile *d_tileA, int blkmA,
                                                  int *d_blkrowptrB, int *d_blkcolidxB, Tile *d_tileB, int blknB,
                                                  int *d_blkrowptrC)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp_id = global_id >> 5; // global_id / WARP_SIZE;
    __shared__ unsigned int bitmask[2048];

    if (global_warp_id >= blkmA)
        return;

    const int nmasks = ceil((float)blknB / (float)32);
    const int local_warp_id = threadIdx.x >> 5; // global_id / WARP_SIZE;
    const int lane_id = threadIdx.x & 31;
    unsigned int *bitmask_local = &bitmask[local_warp_id * 512];

    for (int i = lane_id; i < nmasks; i += 32)
        bitmask_local[i] = 0;

    int astart = d_blkrowptrA[global_warp_id];
    int astop = d_blkrowptrA[global_warp_id + 1];
    for (int i = astart; i < astop; i++)
    {
        int rowidx = d_blkcolidxA[i];
        uint64_t bitmapA[4];
        int bstart = d_blkrowptrB[rowidx];
        int bstop = d_blkrowptrB[rowidx + 1];
        bitmapA[0] = d_tileA[i].bitmap[0];
        bitmapA[1] = d_tileA[i].bitmap[1];
        bitmapA[2] = d_tileA[i].bitmap[2];
        bitmapA[3] = d_tileA[i].bitmap[3];
        for (int j = bstart + lane_id; j < bstop; j += 32)
        {
            int colidx = d_blkcolidxB[j];
            uint64_t bitmap[4] = {0};
            b64_multiply(bitmapA, d_tileB[j].bitmap, bitmap);
            if (bitmap[0] | bitmap[1] | bitmap[2] | bitmap[3])
            {
                unsigned int mask = 1 << (31 - colidx % 32);
                atomicOr(&bitmask_local[colidx / 32], mask);
            }
        }
    }
    //__syncthreads();

    int cnt = 0;
    for (int i = lane_id; i < nmasks; i += 32)
        cnt += __popc(bitmask_local[i]);
    cnt = sum_32_shfl(cnt);

    if (!lane_id)
        d_blkrowptrC[global_warp_id] = cnt;
}

__global__ void tile_spgemm_step1_numeric_cuda_spa_kernel(int *d_blkrowptrA, int *d_blkcolidxA, Tile *d_tileA, int blkmA,
                                                          int *d_blkrowptrB, int *d_blkcolidxB, Tile *d_tileB, int blknB,
                                                          int *d_blkrowptrC, int *d_blkcolidxC, Tile *d_tileC, uint64_t *bitmap_spa_buffer)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp_id = global_id >> 5; // global_id / WARP_SIZE;
    __shared__ unsigned int bitmask[2048];

    if (global_warp_id >= blkmA)
        return;

    uint64_t *spa_buffer = bitmap_spa_buffer + global_warp_id * 4 * blknB;

    const int nmasks = ceil((float)blknB / (float)32);
    const int nmasks_warpwise = ceil((float)nmasks / (float)32) * 32; // make sure shfl func works
    const int local_warp_id = threadIdx.x >> 5;                       // global_id / WARP_SIZE;
    const int lane_id = threadIdx.x % 32;
    unsigned int *bitmask_local = &bitmask[local_warp_id * 512];
    uint64_t bitmap_A_local[4];

    for (int i = lane_id; i < nmasks_warpwise; i += 32)
        bitmask_local[i] = 0;

    int cbase = d_blkrowptrC[global_warp_id];

    int astart = d_blkrowptrA[global_warp_id];
    int astop = d_blkrowptrA[global_warp_id + 1];
    for (int i = astart; i < astop; i++)
    {
        int rowidx = d_blkcolidxA[i];
        int bstart = d_blkrowptrB[rowidx];
        int bstop = d_blkrowptrB[rowidx + 1];
        bitmap_A_local[0] = d_tileA[i].bitmap[0];
        bitmap_A_local[1] = d_tileA[i].bitmap[1];
        bitmap_A_local[2] = d_tileA[i].bitmap[2];
        bitmap_A_local[3] = d_tileA[i].bitmap[3];
        for (int j = bstart + lane_id; j < bstop; j += 32)
        {
            int colidx = d_blkcolidxB[j];
            uint64_t bitmap[4] = {0};
            b64_multiply(bitmap_A_local, d_tileB[j].bitmap, bitmap);
            // if (global_warp_id == 0 && rowidx == 0 && colidx == 0)
            // {
            //     printf("bitmap A: %016llx %016llx %016llx %016llx\n", bitmap_A_local[0], bitmap_A_local[1], bitmap_A_local[2], bitmap_A_local[3]);
            //     printf("bitmap B: %016llx %016llx %016llx %016llx\n", d_tileB[j].bitmap[0], d_tileB[j].bitmap[1], d_tileB[j].bitmap[2], d_tileB[j].bitmap[3]);
            //     printf("bitmap C: %016llx %016llx %016llx %016llx\n", bitmap[0], bitmap[1], bitmap[2], bitmap[3]);
            // }

            if (bitmap[0] | bitmap[1] | bitmap[2] | bitmap[3])
            {
                unsigned int mask = 1 << (31 - colidx % 32);
                atomicOr(&bitmask_local[colidx / 32], mask);
                int *dst_bitmap_addr = (int *)(spa_buffer + colidx * 4);
                int *src_bitmap_addr = (int *)bitmap;
#pragma unroll
                for (int k = 0; k < 8; ++k)
                {
                    atomicOr(dst_bitmap_addr + k, src_bitmap_addr[k]);
                    // if (global_warp_id == 0 && rowidx == 0 && colidx == 0)
                    //     printf("check %d: %08x %08x\n", k, dst_bitmap_addr[k], src_bitmap_addr[k]);
                }
            }
        }
    }

    int offset = 0;
    for (int i = lane_id; i < nmasks_warpwise; i += 32)
    {
        unsigned int maski = bitmask_local[i];
        int cnt = __popc(maski);

        // inclusive scan
        int cnt_scan = scan_32_shfl(cnt, lane_id);
        cnt_scan += offset;

        // sum
        offset = __shfl_sync(0xffffffff, cnt_scan, 31);

        // to exclusive scan
        cnt_scan -= cnt;

        // write to gmem
        int localoff = 0;
#pragma unroll
        for (int biti = 0; biti < 32; biti++)
        {
            if ((maski >> (31 - biti)) & 0x1)
            {
                // d_blkrowidxC[cbase + cnt_scan + localoff] = global_warp_id;
                d_blkcolidxC[cbase + cnt_scan + localoff] = i * 32 + biti;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    d_tileC[cbase + cnt_scan + localoff].bitmap[j] = spa_buffer[(i * 32 + biti) * 4 + j];
                // if (global_warp_id == 0 && i == 0 && biti == 0)
                //     printf("bitmap C: %016llx %016llx %016llx %016llx\n", d_tileC[cbase + cnt_scan + localoff].bitmap[0], d_tileC[cbase + cnt_scan + localoff].bitmap[1], d_tileC[cbase + cnt_scan + localoff].bitmap[2], d_tileC[cbase + cnt_scan + localoff].bitmap[3]);
                localoff++;
            }
        }
    }
}
