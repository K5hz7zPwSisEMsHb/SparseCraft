#include <map>
#include <unordered_map>

#include <tmatrix/DataStructure/TileMatrix.h>
#include <tmatrix/Utils/bitmap_utils.cuh>
#include <tmatrix/Utils/MemoryPool.h>
#include <tmatrix/Utils/omp_utils.h>
#include <tmatrix/Utils/timer.h>
#include <tmatrix/Utils/msg.h>
#include <tmatrix/DataStructure/CSR2Tile.cuh>
#include <tmatrix/Utils/rbtree.h>
#include <tmatrix/DataStructure/predict.cuh>

#include <omp.h>

int manual_select = -1;


__device__ __forceinline__ TileIndex device_b64_count_row(const uint64_t*bitmap, TileIndex row)
{
    int row8 = row >> 3;
    row &= 7;
    TileIndex row_left = (bitmap[row8] >> (row << 3)) & 0xff;
    TileIndex row_right= (bitmap[row8 + 1] >> (row << 3)) & 0xff;
    return __popc(row_left) + __popc(row_right);
}

__device__ __forceinline__ TileIndex device_b64_count_col(const uint64_t*bitmap, TileIndex col)
{
    int col8 = col >> 3;
    col &= 7;
    uint64_t col_upper = (bitmap[col8] >> col) & 0x0101010101010101;
    uint64_t col_lower = (bitmap[col8 + 2] >> col) & 0x0101010101010101;
    return __popcll(col_upper) + __popcll(col_lower);
}

__device__ __forceinline__ uint16_t device_codelen_bitmap2coo(const uint64_t*bitmap, MatIndex*nnz)
{
    MatIndex count = __popcll(bitmap[0]) + __popcll(bitmap[1]) + __popcll(bitmap[2]) + __popcll(bitmap[3]); 
    MatIndex bits = count * 8 + 8;
    *nnz = count;
    return (bits + 7) >> 3;
}

__device__ __forceinline__ uint16_t device_codelen_bitmap2csr(const uint64_t*bitmap, MatIndex*nnz)
{
    MatIndex count = __popcll(bitmap[0]) + __popcll(bitmap[1]) + __popcll(bitmap[2]) + __popcll(bitmap[3]); 
    MatIndex bits = count * 4 + 17 * 8;
    *nnz = count;
    return (bits + 7) >> 3;
}

__device__ __forceinline__ uint16_t device_codelen_bitmap2ell(const uint64_t*bitmap, MatIndex*nnz)
{
    MatIndex max_row_len = 0;
    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        MatIndex row_len = __popc(bitmap_get_row(bitmap, i));
        max_row_len = max(max_row_len, row_len);
    }
    *nnz = max_row_len * 16;
    MatIndex bits = max_row_len * 4 * 16 + 4;
    return (bits + 7) >> 3;
}

__device__ __forceinline__ uint16_t device_codelen_bitmap2hyb(const uint64_t*bitmap, MatIndex*nnz)
{
    MatIndex min_row_len = 17, row_len[16] = {0}, cnt = 0;

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        uint64_t row_map = bitmap_get_row(bitmap, i);
        row_len[i] = __popc(row_map);
        min_row_len = min(min_row_len, max(row_len[i], 1));
    }

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        cnt += max(row_len[i] - min_row_len, 0);
    }
    
    *nnz = min_row_len * 16 + cnt;
    MatIndex bits = ((min_row_len * 16 + 1) / 2 + 2 + cnt) * 8;
    return (bits + 7) >> 3;
}

__device__ __forceinline__ uint16_t device_codelen_bitmap2drw(const uint64_t*bitmap, MatIndex*nnz)
{
    MatIndex bits = 0;

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        uint16_t row_len = bitmap_get_row(bitmap, i);
        bits += row_len? 4: 0;
    }
    *nnz = bits * 4;
    bits += 4;
    
    return (bits + 7) >> 3;
}

__device__ __forceinline__ uint16_t device_codelen_bitmap2dcl(const uint64_t*bitmap, MatIndex*nnz)
{
    MatIndex bits = 0;

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        uint16_t col_len = bitmap_get_col(bitmap, i);
        bits += col_len? 4: 0;
    }
    *nnz = bits * 4;
    bits += 4;

    return (bits + 7) >> 3;
}

__device__ __forceinline__ uint16_t device_codelen_bitmap2dns(const uint64_t*bitmap, MatIndex*nnz)
{
    *nnz = 256;
    return 0;
}

__device__ uint16_t device_bitmap2codelen(const uint64_t*bitmap, MatIndex*nnz, TileFormat format)
{
    uint16_t (*codelen_func[])(const uint64_t*, MatIndex*) = {
        device_codelen_bitmap2coo,
        device_codelen_bitmap2csr,
        device_codelen_bitmap2ell,
        device_codelen_bitmap2hyb,
        device_codelen_bitmap2drw,
        device_codelen_bitmap2dcl,
        device_codelen_bitmap2dns
    };
    return codelen_func[format](bitmap, nnz);
}

void dense_tile_2_coo(const uint64_t* bitmap, MatValue*dense, TileIndex*bits, MatValue* val)
{
    bits[0] = bitmap_count(bitmap) - 1;
    #pragma unroll
    for (int i = 0, idx = 0; i < 256; ++i)
    {
        if (bitmap_check(bitmap, i)) 
        {
            bits[idx + 1] = i;
            val[idx++] = dense[i];
        }
    }
}

void dense_tile_2_csr(const uint64_t* bitmap, MatValue*dense, TileIndex*bits, MatValue* val)
{
    TileIndex nnz = 0;

    #pragma unroll
    for (MatIndex i = 0; i < TILE_N; ++i)
    {
        uint16_t bitrow = bitmap_get_row(bitmap, i), nzr = __builtin_popcount(bitrow);
        for (MatIndex j = 0; j < nzr; ++j)
        {
            unsigned short lowbit = bitrow & -bitrow, col = __builtin_ctz(lowbit);
            TileIndex_CSR_Set(bits + 17, nnz + j, col);
            val[nnz + j] = dense[i * TILE_N + col];
            bitrow ^= lowbit;
        }
        nnz += nzr;
        bits[i + 1] = nnz;
    }
    bits[0] = 0;
}

void dense_tile_2_ell(const uint64_t* bitmap, MatValue*dense, TileIndex*bits, MatValue* val)
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
            unsigned short lowbit = row_map[i] & -row_map[i], col = __builtin_ctz(lowbit);
            TileIndex_CSR_Set(bits, 1 + i * max_row_len + j, col);
            val[i * max_row_len + j] = dense[i * TILE_N + col];
            row_map[i] ^= lowbit;
        }
        memset(val + i * max_row_len + row_len, 0, (max_row_len - row_len) * sizeof(MatValue));
    }
}

void dense_tile_2_hyb(const uint64_t* bitmap, MatValue*dense, TileIndex*bits, MatValue* val)
{
    MatIndex min_row_len = 17, row_len[16] = {0}, cnt = 0;
    uint16_t row_map[16] = {0};

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
        for (MatIndex j = 0; j < min_row_len && row_map[i]; ++j)
        {
            uint16_t lowbit = row_map[i] & -row_map[i];
            TileIndex col = __builtin_ctz(lowbit);
            TileIndex_CSR_Set(bits, 4 + i * min_row_len + j, col);
            val[i * min_row_len + j] = dense[i * TILE_N + col];
            row_map[i] ^= lowbit;
        }

        for (MatIndex j = 0; j < rest; ++j)
        {
            uint16_t lowbit = row_map[i] & -row_map[i];
            TileIndex col = __builtin_ctz(lowbit);
            bits[ell_using_size + cnt] = i << 4 | col;
            val[min_row_len * 16 + cnt] = dense[i * TILE_N + col];
            cnt++;
            row_map[i] ^= lowbit;
        }
    }
    bits[1] = cnt;
}

void dense_tile_2_drw(const uint64_t* bitmap, MatValue*dense, TileIndex*bits, MatValue* val)
{
    MatIndex cnt = 0;

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        if (bitmap_get_row(bitmap, i))
        {
            TileIndex_CSR_Set(bits, cnt + 1, i);
            memcpy(val + cnt * TILE_N, dense + i * TILE_N, TILE_N * sizeof(MatValue));
            cnt++;
        }
    }
    TileIndex_CSR_Set(bits, 0, cnt - 1);
}

void dense_tile_2_dcl(const uint64_t* bitmap, MatValue*dense,TileIndex*bits, MatValue* val)
{
    TileIndex cnt = 0;
    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        if (bitmap_get_col(bitmap, i))
        {
            TileIndex_CSR_Set(bits, cnt + 1, i);
            #pragma unroll
            for (MatIndex j = 0; j < 16; ++j)
            {
                val[cnt * TILE_N + j] = dense[j * TILE_N + i];
            }
            cnt++;
        }
    }
    TileIndex_CSR_Set(bits, 0, cnt - 1);
}

void dense_tile_2_dns(const uint64_t* bitmap, MatValue*dense, TileIndex*bits, MatValue* val)
{
    memcpy(val, dense, 256 * sizeof(MatValue));
}

void dense_tile_2_fmt(dense_tile* t, TileIndex*bits, MatValue* val, TileFormat fmt)
{
    void (*bitmap2fmt[7])(const uint64_t* bitmap, MatValue*dense, TileIndex*bits, MatValue* val) = {
        dense_tile_2_coo,
        dense_tile_2_csr,
        dense_tile_2_ell,
        dense_tile_2_hyb,
        dense_tile_2_drw,
        dense_tile_2_dcl,
        dense_tile_2_dns
    };
    bitmap2fmt[fmt](t->bitmap, t->val, bits, val);
}

__device__ uint16_t device_bitmap_count(const uint64_t *bitmap)
{
    return __popcll(bitmap[0]) + __popcll(bitmap[1]) + __popcll(bitmap[2]) + __popcll(bitmap[3]); 
}

__device__ __forceinline__ bool device_bitmap_check(const uint64_t *bitmap, TileIndex row, TileIndex col)
{
    int row8 = row >> 3;
    row &= 7;
    int col8 = col >> 3;
    col &= 7;
    return bitmap[(row8 << 1) | col8] >> ((row << 3) | col) & 1;
}

__device__ bool device_bitmap_check(const uint64_t *bitmap, TileIndex idx)
{
    int row = idx >> 4;
    int col = idx & 15;
    return device_bitmap_check(bitmap, row, col);
}

__device__ void device_bitmap_2_coo(const uint64_t* bitmap, TileIndex *bits)
{
    bits[0] = device_bitmap_count(bitmap) - 1;
    
    #pragma unroll
    for (int i = 0, idx = 1; i < 256; ++i)
    {
        if (device_bitmap_check(bitmap, i))
        {
            bits[idx] = i;
            idx++;
        }
    }
}

__device__ void device_bitmap_2_csr(const uint64_t* bitmap, TileIndex *bits)
{
    TileIndex nnz = 0;

    #pragma unroll
    for (MatIndex i = 0; i < TILE_N; ++i)
    {
        uint16_t bitrow = bitmap_get_row(bitmap, i), nzr = __popc(bitrow);
        for (MatIndex j = 0; j < nzr; ++j)
        {
            uint16_t lowbit = bitrow & -bitrow, col = __ffs(lowbit) - 1;
            TileIndex_CSR_Set(bits + 17, nnz + j, col);
            bitrow ^= lowbit;
        }
        nnz += nzr;
        bits[i + 1] = nnz;
    }
    bits[0] = 0;
}

__device__ void device_bitmap_2_ell(const uint64_t* bitmap, TileIndex *bits)
{
    MatIndex max_row_len = 0;
    ushort row_map[16] = {0};

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        row_map[i] = bitmap_get_row(bitmap, i);
        MatIndex row_len = __popc(row_map[i]);
        max_row_len = max(max_row_len, row_len);
    }

    TileIndex_CSR_Set(bits, 0, max_row_len - 1);

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        MatIndex row_len = __popc(row_map[i]);
        for (MatIndex j = 0; j < row_len; ++j)
        {
            uint16_t lowbit = row_map[i] & -row_map[i], col = __ffs(lowbit) - 1;
            TileIndex_CSR_Set(bits, 1 + i * max_row_len + j, col);
            row_map[i] ^= lowbit;
        }
    }
}

__device__ void device_bitmap_2_hyb(const uint64_t* bitmap, TileIndex *bits)
{
    MatIndex min_row_len = 17, row_len[16] = {0}, cnt = 0;
    uint16_t row_map[16] = {0};

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        row_map[i] = bitmap_get_row(bitmap, i);
        row_len[i] = __popc(row_map[i]);
        min_row_len = min(min_row_len, max(row_len[i], 1));
    }
    bits[0] = min_row_len;
    MatIndex ell_using_size = (min_row_len * 16 + 1) / 2 + 2;

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        MatIndex rest = max(row_len[i] - min_row_len, 0);
        for (MatIndex j = 0; j < min_row_len && row_map[i]; ++j)
        {
            uint16_t lowbit = row_map[i] & -row_map[i];
            TileIndex col = __ffs(lowbit) - 1;
            TileIndex_CSR_Set(bits, 4 + i * min_row_len + j, col);
            row_map[i] ^= lowbit;
        }

        for (MatIndex j = 0; j < rest; ++j)
        {
            uint16_t lowbit = row_map[i] & -row_map[i];
            TileIndex col = __ffs(lowbit) - 1;
            bits[ell_using_size + cnt] = i << 4 | col;
            cnt++;
            row_map[i] ^= lowbit;
        }
    }
    bits[1] = cnt;
}

__device__ void device_bitmap_2_drw(const uint64_t* bitmap, TileIndex *bits)
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

__device__ void device_bitmap_2_dcl(const uint64_t* bitmap, TileIndex *bits)
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

__device__ void device_bitmap_2_dns(const uint64_t* bitmap, TileIndex *bits)
{
}

__device__ void device_bitmap_2_fmt(const uint64_t* bitmap, char *bits, TileFormat fmt)
{
    void (*bitmap2fmt[7])(const uint64_t* bitmap, TileIndex *bits) = {
        device_bitmap_2_coo,
        device_bitmap_2_csr,
        device_bitmap_2_ell,
        device_bitmap_2_hyb,
        device_bitmap_2_drw,
        device_bitmap_2_dcl,
        device_bitmap_2_dns
    };
    bitmap2fmt[fmt](bitmap, (TileIndex*)bits);
}

void update_bit64(uint64_t*b64, int row, int col)
{
    int row8 = row / 8, col8 = col / 8;
    row = row % 8, col = col % 8;
    b64[row8 << 1 | col8] |= (1ULL << (row << 3 | col));
}

__device__ bool bitmap_check_drw(const uint64_t* bitmap)
{
    for (int i = 0; i < 8; ++i)
    {
        int sum = __popc((bitmap[0] >> (i << 3)) & 0xFF) +
                  __popc((bitmap[1] >> (i << 3)) & 0xFF);
        if (sum && sum < 16) return false;
    }
    for (int i = 0; i < 8; ++i)
    {
        int sum = __popc((bitmap[2] >> (i << 3)) & 0xFF) +
                  __popc((bitmap[3] >> (i << 3)) & 0xFF);
        if (sum && sum < 16) return false;
    }
    return true;
}

__device__ bool bitmap_check_dcl(const uint64_t* bitmap)
{
    for (int i = 0; i < 8; ++i)
    {
        int sum = __popcll((bitmap[0] >> i) & 0x0101010101010101) +
                  __popcll((bitmap[2] >> i)  & 0x0101010101010101);
        if (sum && sum < 16) return false;
    }
    for (int i = 0; i < 8; ++i)
    {
        int sum = __popcll((bitmap[1] >> i)  & 0x0101010101010101) +
                  __popcll((bitmap[3] >> i)  & 0x0101010101010101);
        if (sum && sum < 16) return false;
    }
    return true;
}

__host__ __device__ uint16_t bitmap_get_row(const uint64_t *bitmap, TileIndex row)
{
    int row8 = row >> 3;
    int _row = row & 7;
    uint64_t row_upper = (bitmap[row8 << 1] >> (_row << 3)) & 0xff;
    uint64_t row_lower = (bitmap[(row8 << 1) + 1] >> (_row << 3)) & 0xff;
    return (row_lower << 8) | row_upper;
}

__host__ __device__ uint16_t bitmap_get_col(const uint64_t *bitmap, TileIndex col)
{
    int col8 = col >> 3;
    col &= 7;
    uint64_t col_upper = (bitmap[col8] >> col) & 0x0101010101010101;
    uint64_t col_lower = (bitmap[col8 + 2] >> col) & 0x0101010101010101;
    uint16_t res = 0;
    for (int i = 0; i < 8; i++)
    {
        uint16_t bits = (col_upper & 1) | ((col_lower & 1) << 8);
        res |= bits << i;
        col_upper >>= 8;
        col_lower >>= 8;
    }
    return res;
}

__device__ TileFormat pixel_select(const uint64_t*bitmap, int lane_id)
{
    return CSR;
}

__device__ TileFormat tile_select(const uint64_t*bitmap)
{
    int nnz = __popcll(bitmap[0]) + __popcll(bitmap[1]) +
              __popcll(bitmap[2]) + __popcll(bitmap[3]);
    if (nnz < 12)
    {
        return COO;
    }
    if (nnz >= 192)
    {
        return DNS;
    }
    if (bitmap_check_drw(bitmap))
    {
        return DRW;
    }
    if (bitmap_check_dcl(bitmap))
    {
        return DCL;
    }
    double row_len_mean = nnz * 1.0 / 16;
    double variance = 0.0;
    MatIndex max_row_len = 0;

    #pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        MatIndex row_len = __popcll(bitmap_get_row(bitmap, i));
        max_row_len = max(max_row_len, row_len);
        double delta = row_len - row_len_mean;
        variance += delta * delta;
    }
    variance /= 16;
    double variation = sqrt(variance) / row_len_mean;

    if (variance < 0.2) {
        return ELL;
    } else {
        return CSR;
    }
}

__device__ TileFormat storage_select(const uint64_t*bitmap)
{
    TileFormat fmt = COO;
    int min_bytes = 3000;
    for (TileFormat i = COO; i <= DNS; ++i)
    {
        int bitslen, valslen;
        bitslen = device_bitmap2codelen(bitmap, &valslen, i);
        int bytes = bitslen + valslen * sizeof(MatValue);
        if (bytes < min_bytes)
        {
            min_bytes = bytes;
            fmt = i;
        }
    }
    return fmt;
}

__global__ void tile_format_prediction_single(Tile*tiles, const int n)
{
    int gthread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (gthread_id >= n) return;
    tiles[gthread_id].fmt = tile_select(tiles[gthread_id].bitmap);
}

__global__ void storage_format_prediction_single(Tile*tiles, const int n)
{
    int gthread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (gthread_id >= n) return;

    TileFormat fmt = COO;
    int min_bytes = 3000;
    for (TileFormat i = COO; i <= DNS; ++i)
    {
        int bitslen, valslen;
        bitslen = device_bitmap2codelen(tiles[gthread_id].bitmap, &valslen, i);
        int bytes = bitslen + valslen * sizeof(MatValue);
        if (bytes < min_bytes)
        {
            min_bytes = bytes;
            fmt = i;
        }
    }
    tiles[gthread_id].fmt = fmt;
}

__global__ void tile_bytes_calculate(Tile*tiles, uint64_t*tmp, const int n)
{
    int gthread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (gthread_id >= n) return;

    TileFormat fmt = tiles[gthread_id].fmt;
    int bitslen, valslen;
    bitslen = device_bitmap2codelen(tiles[gthread_id].bitmap, &valslen, fmt);
    tiles[gthread_id].valslen = valslen - 1;
    bitslen = (bitslen + 15) / 16 * 16;
    valslen = (valslen + 1)  / 2  * 2;
    tiles[gthread_id].bitslen = bitslen / 16;
    uint64_t bytes = bitslen + valslen * sizeof(MatValue); // bytes must be 16x
    tmp[gthread_id] = bytes / 16;
}

__global__ void tile_bytes_apply(Tile*tiles, char*data, uint64_t*tmp, const int n)
{
    int gthread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (gthread_id >= n) return;
    tiles[gthread_id].bits_off = tmp[gthread_id] * 16;

    TileFormat fmt = tiles[gthread_id].fmt;
    char*bits = data + tiles[gthread_id].bits_off;
    device_bitmap_2_fmt(tiles[gthread_id].bitmap, bits, fmt);
}

__global__ void tile_data_apply(const Tile*tiles, char*data, const int n)
{
    int gthread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (gthread_id >= n) return;

    TileFormat fmt = tiles[gthread_id].fmt;
    char*bits = data + tiles[gthread_id].bits_off;
    device_bitmap_2_fmt(tiles[gthread_id].bitmap, bits, fmt);
}

double set_memory_pool(BaseMatrix*dC)
{
    uint64_t *tmp;
    double used_time = 0;

    cudaMalloc(&tmp, (dC->_nnz + 1) * sizeof(uint64_t));
    Timer t;
    timer_start(t);
    tile_bytes_calculate<<<(dC->_nnz + 255) / 256, 256>>>(dC->tiles, tmp, dC->_nnz);
    cudaDeviceSynchronize();
    timer_end(t);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        echo(error, "CUDA error set_memory_pool 0: %s", cudaGetErrorString(e));
        exit(1);
    }
    used_time += timer_duration(t);
    // thrust::exclusive_scan(thrust::device, tmp, tmp + dC->_nnz + 1, tmp, 0);
    {
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        timer_start(t);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, tmp, dC->_nnz + 1);
        timer_end(t);
        used_time += timer_duration(t);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        timer_start(t);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, tmp, dC->_nnz + 1);
        timer_end(t);
        used_time += timer_duration(t);
        cudaFree(d_temp_storage);
    }
    cudaMemcpy(&dC->_data_len, tmp + dC->_nnz, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    dC->_data_len *= 16;
    e = cudaMalloc(&dC->data, dC->_data_len);
    if (e != cudaSuccess)
    {
        echo(error, "Memory pool allocation failed: %s", cudaGetErrorString(e));
        echo(error, "Data length: %llu, memory: %.3lf MB", dC->_data_len, dC->_data_len / 1024.0 / 1024.0);
        exit(1);
    }
    cudaMemset(dC->data, 0, dC->_data_len);
    timer_start(t);
    tile_bytes_apply<<<(dC->_nnz + 255) / 256, 256>>>(dC->tiles, dC->data, tmp, dC->_nnz);
    cudaDeviceSynchronize();
    timer_end(t);
    e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        echo(error, "CUDA error set_memory_pool 1: %s", cudaGetErrorString(e));
        exit(1);
    }
    used_time += timer_duration(t);
    cudaFree(tmp);
    return used_time;
}

double set_memory_pool(BaseMatrixCSC*dC)
{
    uint64_t *tmp;
    double used_time = 0;

    cudaCheckError(cudaMalloc(&tmp, (dC->_nnz + 1) * sizeof(uint64_t)));
    Timer t;
    timer_start(t);
    tile_bytes_calculate<<<(dC->_nnz + 255) / 256, 256>>>(dC->tiles, tmp, dC->_nnz);
    cudaDeviceSynchronize();
    timer_end(t);
    used_time += timer_duration(t);
    // thrust::exclusive_scan(thrust::device, tmp, tmp + dC->_nnz + 1, tmp, 0);
    {
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        timer_start(t);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, tmp, dC->_nnz + 1);
        timer_end(t);
        used_time += timer_duration(t);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        timer_start(t);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, tmp, dC->_nnz + 1);
        timer_end(t);
        used_time += timer_duration(t);
        cudaFree(d_temp_storage);
    }
    cudaMemcpy(&dC->_data_len, tmp + dC->_nnz, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    dC->_data_len *= 16;
    cudaError_t e = cudaMalloc(&dC->data, dC->_data_len);
    if (e != cudaSuccess)
    {
        echo(error, "Memory pool allocation failed: %s", cudaGetErrorString(e));
        echo(error, "Data length: %llu, memory: %.3lf MB", dC->_data_len, dC->_data_len / 1024.0 / 1024.0);
        exit(1);
    }
    cudaMemset(dC->data, 0, dC->_data_len);
    timer_start(t);
    tile_bytes_apply<<<(dC->_nnz + 255) / 256, 256>>>(dC->tiles, dC->data, tmp, dC->_nnz);
    cudaDeviceSynchronize();
    timer_end(t);
    used_time += timer_duration(t);
    cudaFree(tmp);
    return used_time;
}

__global__ void build_A_row_map(
    const MatIndex *A_row_ptr, const MatIndex *A_col_idx, uint32_t *A_row_map, const int m, const int row_map_len
)
{
    int gthread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int gwarp_id = gthread_id >> 5, lane_id = threadIdx.x & 31;
    if (gwarp_id >= m) return;

    for (int i = A_row_ptr[gwarp_id] + lane_id; i < A_row_ptr[gwarp_id + 1]; i += 32)
    {
        int col = A_col_idx[i];
        int row_map_idx = col >> 5;
        int row_map_bit = col & 31;
        atomicOr(A_row_map + gwarp_id * row_map_len + row_map_idx, 1 << row_map_bit);
    }
}

__global__ void build_B_col_map(
    const MatIndex *B_col_ptr, const MatIndex *B_row_idx, uint32_t *B_col_map, const int n, const int col_map_len
)
{
    int gthread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int gwarp_id = gthread_id >> 5, lane_id = threadIdx.x & 31;
    if (gwarp_id >= n) return;

    for (int i = B_col_ptr[gwarp_id] + lane_id; i < B_col_ptr[gwarp_id + 1]; i += 32)
    {
        int row = B_row_idx[i];
        int col_map_idx = row >> 5;
        int col_map_bit = row & 31;
        atomicOr(B_col_map + gwarp_id * col_map_len + col_map_idx, 1 << col_map_bit);
    }
}

BaseMatrixCSC* load_mtx_2_csc_tile(
    const char* filename, MatIndex*report_nnz, std::string predict,
    MatIndex&m, MatIndex&n, MatIndex&nnz, MatIndex*&row_ptr, MatIndex*&col_idx, MatValue*&csr_val)
{
    const char* tile_format_name[] = {
        "COO", "CSR", "ELL", "HYB", "DRW", "DCL", "DNS"
    };
    
    *report_nnz = nnz;

    BaseMatrixCSC *t = (BaseMatrixCSC *)malloc(sizeof(BaseMatrixCSC));
    t->meta_m = m;
    t->meta_n = n;
    t->meta_nnz = nnz;
    t->_m = (m + TILE_N - 1) / TILE_N;
    t->_n = (n + TILE_N - 1) / TILE_N;

    RBTree **tile_distributions = (RBTree **)malloc(t->_n * sizeof(RBTree *));
    #pragma omp parallel for
    for (int i = 0; i < t->_n; ++i)
    {
        tile_distributions[i] = create_rbtree();
    }
    t->tile_col_ptr = (MatIndex *)calloc((t->_n + 1), sizeof(MatIndex));

    // create omp locks
    omp_lock_t *omp_locks = (omp_lock_t *)malloc(t->_n * sizeof(omp_lock_t));
    #pragma omp parallel for
    for (int i = 0; i < t->_n; ++i) omp_init_lock(&omp_locks[i]);

    #pragma omp parallel for
    for (int i = 0; i < m; i += TILE_N)
    {
        int row16 = i / TILE_N;
        int end_i = std::min(TILE_N, m - i);
        for (int ii = 0; ii < end_i; ++ii)
        {
            for (int jj = row_ptr[i + ii]; jj < row_ptr[i + ii + 1]; ++jj)
            {
                MatIndex col = col_idx[jj];
                MatIndex col16 = col / TILE_N;
                // lock the col16
                omp_set_lock(&omp_locks[col16]);
                RBTreeNode *node = rb_search(tile_distributions[col16], row16);
                if (node == NULL)
                {
                    dense_tile t;
                    update_bit64(t.bitmap, ii, col % TILE_N);
                    t.val[ii * TILE_N + col % TILE_N] = csr_val[jj];
                    rb_insert(tile_distributions[col16], row16, t);
                }
                else
                {
                    update_bit64(node->val.bitmap, ii, col % TILE_N);
                    node->val.val[ii * TILE_N + col % TILE_N] = csr_val[jj];
                }
                // unlock the col16
                omp_unset_lock(&omp_locks[col16]);
            }
        }
        // t->tile_row_ptr[row16 + 1] = tile_distributions[row16]->size;
    }

    #pragma omp parallel for
    for (int i = 0; i < t->_n; ++i)
    {
        t->tile_col_ptr[i + 1] = tile_distributions[i]->size;
        omp_destroy_lock(&omp_locks[i]);
    }

    omp_inclusive_scan(t->tile_col_ptr + 1, t->_m);
    t->_nnz = t->tile_col_ptr[t->_m];
    echo(info, "[LOAD MTX] m: %d, n: %d, report nnz: %d, nz16: %d | Use \"%s\" to predict tile format", m, n, nnz, t->_nnz, predict.c_str());

    t->tile_row_idx = (MatIndex *)malloc(t->_nnz * sizeof(MatIndex));
    t->tiles = (Tile *)calloc(t->_nnz, sizeof(Tile));

    MatIndex cnt_format[7] = {0};
    
    #pragma omp parallel for
    for (int i = 0; i < t->_n; ++i)
    {
        int idx = t->tile_col_ptr[i];
        RBTreeIterator*it = rb_iterator_init(tile_distributions[i]);
        while (rb_iterator_has_next(it))
        {
            RBTreeNode *node = rb_iterator_next(it);
            t->tile_row_idx[idx] = node->key;
            memcpy(t->tiles[idx].bitmap, node->val.bitmap, sizeof(uint64_t) * 4);
            idx++;
        }
        free(it);
    }

    // start format prediction using cuda
    Tile*d_tile;
    cudaMalloc(&d_tile, t->_nnz * sizeof(Tile));
    cudaMemcpy(d_tile, t->tiles, t->_nnz * sizeof(Tile), cudaMemcpyHostToDevice);

    double predict_time = 0;
    if (predict == "pixel") {
        sequence_slime_net*model = load_slimenet("model/slime_net.bin");
        sequence_slime_net*d_modle = slimenet_to_device(model);
        Timer tt;
        timer_start(tt);
        pixel_format_prediction_single<<<(t->_nnz + 1) / 2, 64>>>(d_modle, d_tile, t->_nnz);
        cudaDeviceSynchronize();
        timer_end(tt);
        predict_time = timer_duration(tt);
        cudaFree(d_modle);
        free(model);
    }
    else if (predict == "tile") {
        Timer tt;
        timer_start(tt);
        tile_format_prediction_single<<<(t->_nnz + 127) / 128, 128>>>(d_tile, t->_nnz);
        cudaDeviceSynchronize();
        timer_end(tt);
        predict_time = timer_duration(tt);
    }
    else if (predict == "storage") {
        Timer tt;
        timer_start(tt);
        storage_format_prediction_single<<<(t->_nnz + 127) / 128, 128>>>(d_tile, t->_nnz);
        cudaDeviceSynchronize();
        timer_end(tt);
        predict_time = timer_duration(tt);
    }
    else echo(error, "Unknown prediction method: %s", predict.c_str());
    if (predict_time != 0) echo(info, "Prediction Time: %.3lf ms, throughput: %.3lf K/s", predict_time, t->_nnz / predict_time);
    Tile*tmp = t->tiles;
    t->tiles = d_tile;
    Timer timer;
    timer_start(timer);
    echo(start_status, "Setting Memory Pool");
    set_memory_pool(t);
    cudaDeviceSynchronize();
    timer_end(timer);
    echo(stop_status, "");

    t->tiles = tmp;
    cudaMemcpy(t->tiles, d_tile, t->_nnz * sizeof(Tile), cudaMemcpyDeviceToHost);
    cudaFree(d_tile);
    
    char* data = (char*)malloc(t->_data_len * sizeof(char));
    cudaMemcpy(data, t->data, t->_data_len, cudaMemcpyDeviceToHost);
    cudaFree(t->data);
    t->data = data;
    echo(info, "Set Memory Pool Time: %.3lf ms; Used Memory: %.3lf MB", timer_duration(timer), t->_data_len / 1024.0 / 1024.0);
    // for (int i = 0; i < 10; ++i) echo(debug, "Tile %d: %s", i, tile_format_name[t->tiles[i].fmt]);
    #pragma omp parallel for reduction(+:cnt_format)
    for (int i = 0; i < t->_nnz; ++i)
    {
        cnt_format[t->tiles[i].fmt]++;
    }

    echo(custom, "python3 -c \"import plotext as plt; label = ['COO', 'CSR', 'ELL', 'HYB', 'DRW', 'DCL', 'DNS']; percentages = [%.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf]; plt.simple_bar(label, percentages, width = 100, title = 'Tile Format Breakdown'); plt.show()\"", 
        cnt_format[COO] * 100.0 / t->_nnz, cnt_format[CSR] * 100.0 / t->_nnz, cnt_format[ELL] * 100.0 / t->_nnz, 
        cnt_format[HYB] * 100.0 / t->_nnz, cnt_format[DRW] * 100.0 / t->_nnz, cnt_format[DCL] * 100.0 / t->_nnz, cnt_format[DNS] * 100.0 / t->_nnz);

    #pragma omp parallel for
    for (int i = 0; i < t->_n; ++i)
    {
        double sum = 0;
        int j = t->tile_col_ptr[i];
        RBTreeIterator*it = rb_iterator_init(tile_distributions[i]);
        while (rb_iterator_has_next(it))
        {
            RBTreeNode *node = rb_iterator_next(it);
            dense_tile_2_fmt(
                &(node->val),
                (TileIndex*) (t->data + t->tiles[j].bits_off),
                (MatValue*) (t->data + t->tiles[j].bits_off + t->tiles[j].bitslen * 16),
                t->tiles[j].fmt
            );
            ++j;
        }
        free(it);
        rb_free(tile_distributions[i]);
    }
    free(tile_distributions);
    return t;
}

BaseMatrix* load_mtx_2_tile(
    const char* filename, MatIndex*report_nnz, std::string predict,
    MatIndex&m, MatIndex&n, MatIndex&nnz, MatIndex*&row_ptr, MatIndex*&col_idx, MatValue*&csr_val, 
    double&predict_time)
{
    const char* tile_format_name[] = {
        "COO", "CSR", "ELL", "HYB", "DRW", "DCL", "DNS"
    };
    
    *report_nnz = nnz;

    BaseMatrix *t = (BaseMatrix *)malloc(sizeof(BaseMatrix));
    t->meta_m = m;
    t->meta_n = n;
    t->meta_nnz = nnz;
    t->_m = (m + TILE_N - 1) / TILE_N;
    t->_n = (n + TILE_N - 1) / TILE_N;

    RBTree **tile_distributions = (RBTree **)malloc(t->_m * sizeof(RBTree *));
    #pragma omp parallel for
    for (int i = 0; i < t->_m; ++i)
    {
        tile_distributions[i] = create_rbtree();
    }
    t->tile_row_ptr = (MatIndex *)calloc((t->_m + 1), sizeof(MatIndex));

    #pragma omp parallel for
    for (int i = 0; i < m; i += TILE_N)
    {
        int row16 = i / TILE_N;
        int end_i = std::min(TILE_N, m - i);
        for (int ii = 0; ii < end_i; ++ii)
        {
            for (int jj = row_ptr[i + ii]; jj < row_ptr[i + ii + 1]; ++jj)
            {
                MatIndex col = col_idx[jj];
                MatIndex col16 = col / TILE_N;
                RBTreeNode *node = rb_search(tile_distributions[row16], col16);
                if (node == NULL)
                {
                    dense_tile t;
                    update_bit64(t.bitmap, ii, col % TILE_N);
                    t.val[ii * TILE_N + col % TILE_N] = csr_val[jj];
                    rb_insert(tile_distributions[row16], col16, t);
                }
                else
                {
                    update_bit64(node->val.bitmap, ii, col % TILE_N);
                    node->val.val[ii * TILE_N + col % TILE_N] = csr_val[jj];
                }
            }
        }
        t->tile_row_ptr[row16 + 1] = tile_distributions[row16]->size;
    }

    omp_inclusive_scan(t->tile_row_ptr + 1, t->_m);
    t->_nnz = t->tile_row_ptr[t->_m];
    echo(info, "[LOAD MTX] m: %d, n: %d, report nnz: %d, nz16: %d | Use \"%s\" to predict tile format", m, n, nnz, t->_nnz, predict.c_str());

    t->tile_col_idx = (MatIndex *)malloc(t->_nnz * sizeof(MatIndex));
    t->tiles = (Tile *)calloc(t->_nnz, sizeof(Tile));

    MatIndex cnt_format[7] = {0};
    
    #pragma omp parallel for
    for (int i = 0; i < t->_m; ++i)
    {
        int idx = t->tile_row_ptr[i];
        RBTreeIterator*it = rb_iterator_init(tile_distributions[i]);
        while (rb_iterator_has_next(it))
        {
            RBTreeNode *node = rb_iterator_next(it);
            t->tile_col_idx[idx] = node->key;
            memcpy(t->tiles[idx].bitmap, node->val.bitmap, sizeof(uint64_t) * 4);
            idx++;
        }
        free(it);
    }

    // start format prediction using cuda
    Tile*d_tile;
    cudaMalloc(&d_tile, t->_nnz * sizeof(Tile));
    cudaMemcpy(d_tile, t->tiles, t->_nnz * sizeof(Tile), cudaMemcpyHostToDevice);

    predict_time = 0;
    if (predict == "pixel") {
        sequence_slime_net*model = load_slimenet("model/slime_net.bin");
        if (model == nullptr) return nullptr;
        sequence_slime_net*d_modle = slimenet_to_device(model);
        Timer tt;
        timer_start(tt);
        pixel_format_prediction_single<<<(t->_nnz + 1) / 2, 64>>>(d_modle, d_tile, t->_nnz);
        cudaDeviceSynchronize();
        timer_end(tt);
        predict_time = timer_duration(tt);
        cudaFree(d_modle);
        free(model);
    }
    else if (predict == "tile") {
        Timer tt;
        timer_start(tt);
        tile_format_prediction_single<<<(t->_nnz + 127) / 128, 128>>>(d_tile, t->_nnz);
        cudaDeviceSynchronize();
        timer_end(tt);
        predict_time = timer_duration(tt);
    }
    else if (predict == "storage") {
        Timer tt;
        timer_start(tt);
        storage_format_prediction_single<<<(t->_nnz + 127) / 128, 128>>>(d_tile, t->_nnz);
        cudaDeviceSynchronize();
        timer_end(tt);
        predict_time = timer_duration(tt);
    }
    else echo(error, "Unknown prediction method: %s", predict.c_str());
    if (predict_time != 0) echo(info, "Prediction Time: %.3lf ms, throughput: %.3lf K/s", predict_time, t->_nnz / predict_time);
    Tile*tmp = t->tiles;
    t->tiles = d_tile;
    Timer timer;
    timer_start(timer);
    echo(start_status, "Setting Memory Pool");
    set_memory_pool(t);
    cudaDeviceSynchronize();
    timer_end(timer);
    echo(stop_status, "");

    t->tiles = tmp;
    cudaMemcpy(t->tiles, d_tile, t->_nnz * sizeof(Tile), cudaMemcpyDeviceToHost);
    cudaFree(d_tile);
    
    char* data = (char*)malloc(t->_data_len * sizeof(char));
    cudaMemcpy(data, t->data, t->_data_len, cudaMemcpyDeviceToHost);
    cudaFree(t->data);
    t->data = data;
    echo(info, "Set Memory Pool Time: %.3lf ms; Used Memory: %.3lf MB", timer_duration(timer), t->_data_len / 1024.0 / 1024.0);
    // for (int i = 0; i < 10; ++i) echo(debug, "Tile %d: %s", i, tile_format_name[t->tiles[i].fmt]);
    #pragma omp parallel for reduction(+:cnt_format)
    for (int i = 0; i < t->_nnz; ++i)
    {
        cnt_format[t->tiles[i].fmt]++;
    }

    echo(custom, "python3 -c \"import plotext as plt; label = ['COO', 'CSR', 'ELL', 'HYB', 'DRW', 'DCL', 'DNS']; percentages = [%.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf, %.2lf]; plt.simple_bar(label, percentages, width = 100, title = 'Tile Format Breakdown'); plt.show()\"", 
        cnt_format[COO] * 100.0 / t->_nnz, cnt_format[CSR] * 100.0 / t->_nnz, cnt_format[ELL] * 100.0 / t->_nnz, 
        cnt_format[HYB] * 100.0 / t->_nnz, cnt_format[DRW] * 100.0 / t->_nnz, cnt_format[DCL] * 100.0 / t->_nnz, cnt_format[DNS] * 100.0 / t->_nnz);

    #pragma omp parallel for
    for (int i = 0; i < t->_m; ++i)
    {
        int j = t->tile_row_ptr[i];
        RBTreeIterator*it = rb_iterator_init(tile_distributions[i]);
        while (rb_iterator_has_next(it))
        {
            RBTreeNode *node = rb_iterator_next(it);
            dense_tile_2_fmt(
                &(node->val),
                (TileIndex*) (t->data + t->tiles[j].bits_off),
                (MatValue*) (t->data + t->tiles[j].bits_off + t->tiles[j].bitslen * 16),
                t->tiles[j].fmt
            );
            ++j;
        }
        free(it);
        rb_free(tile_distributions[i]);
    }
    free(tile_distributions);
    return t;
}

__global__ void storage_format_count(uint64_t* bitmaps, int* count, const int n)
{
    int gthread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (gthread_id >= n) return;

    TileFormat fmt = COO;
    int min_bytes = 3000;
    for (TileFormat i = COO; i <= DNS; ++i)
    {
        int bitslen, valslen;
        bitslen = device_bitmap2codelen(bitmaps + gthread_id * 4, &valslen, i);
        int bytes = bitslen + valslen * sizeof(MatValue);
        if (bytes < min_bytes)
        {
            min_bytes = bytes;
            fmt = i;
        }
    }
    atomicAdd(count + fmt, 1);
}

void count_storage_block(
    MatIndex &m, MatIndex &n, MatIndex &nnz, MatIndex *&row_ptr, MatIndex *&col_idx, MatIndex *count)
{
    int blkm = (m + TILE_N - 1) / TILE_N;
    int blkn = (n + TILE_N - 1) / TILE_N;
    int*row_ptr16 = (int *)calloc(blkm + 1, sizeof(int));
    std::unordered_map<MatIndex, uint64_t[4]> *row_bitmaps = new std::unordered_map<MatIndex, uint64_t[4]>[blkm];

    #pragma omp parallel for
    for (int i = 0; i < m; i += TILE_N)
    {
        int row16 = i / TILE_N;
        int end_i = std::min(TILE_N, m - i);
        for (int ii = 0; ii < end_i; ++ii)
        {
            for (int jj = row_ptr[i + ii]; jj < row_ptr[i + ii + 1]; ++jj)
            {
                MatIndex col = col_idx[jj];
                MatIndex col16 = col / TILE_N;
                update_bit64(row_bitmaps[row16][col16], ii, col % TILE_N);
            }
        }
        row_ptr16[row16 + 1] = row_bitmaps[row16].size();
    }

    omp_inclusive_scan(row_ptr16 + 1, blkm);
    int nnz16 = row_ptr16[blkm];
    uint64_t *bitmaps = (uint64_t *)malloc(nnz16 * sizeof(uint64_t) * 4);

    #pragma omp parallel for
    for (int i = 0; i < blkm; ++i)
    {
        int idx = row_ptr16[i];
        for (auto it = row_bitmaps[i].begin(); it != row_bitmaps[i].end(); ++it, ++idx)
        {
            memcpy(bitmaps + idx * 4, it->second, sizeof(uint64_t) * 4);
        }
    }
    delete[] row_bitmaps;
    free(row_ptr16);

    // start format prediction using cuda
    uint64_t *d_bitmaps;
    cudaMalloc(&d_bitmaps, nnz16 * sizeof(uint64_t) * 4);
    cudaMemcpy(d_bitmaps, bitmaps, nnz16 * sizeof(uint64_t) * 4, cudaMemcpyHostToDevice);
    int*d_count;
    cudaMalloc(&d_count, sizeof(int) * 7);
    cudaMemset(d_count, 0, sizeof(int) * 7);
    storage_format_count<<<(nnz16 + 255) / 256, 256>>>(d_bitmaps, d_count, nnz16);
    cudaDeviceSynchronize();
    cudaMemcpy(count, d_count, sizeof(int) * 7, cudaMemcpyDeviceToHost);
    cudaFree(d_bitmaps);
    cudaFree(d_count);
}

bool check_in_bitmap8(uint64_t *bitmap, int idx)
{
    TileIndex row = idx >> 4, col = idx & 15;
    TileIndex row8 = row >> 3, col8 = col >> 3;
    row &= 7, col &= 7;
    return bitmap[row8 << 1 | col8] & (1ULL << (row << 3 | col));
}

void BaseMatrix_And_CSR_Compare(BaseMatrix* tile_matrix, MatIndex m, MatIndex n, MatIndex nnz, MatIndex*row_ptr, MatIndex*col_idx, MatValue*csr_val)
{
    if (tile_matrix->meta_nnz != nnz) echo(warning, "nnz not equal: %d %d", tile_matrix->meta_nnz, nnz);

    // #pragma omp parallel for
    for (MatIndex i = 0; i < tile_matrix->_m; ++i)
    {
        // if (not check) continue;
        std::unordered_map<MatIndex, MatValue> row_map[TILE_N];
        MatIndex row16 = i * TILE_N;
        for (int j = tile_matrix->tile_row_ptr[i]; j < tile_matrix->tile_row_ptr[i + 1]; ++j)
        {
            MatIndex col16 = tile_matrix->tile_col_idx[j] * TILE_N;
            TileFormat fmt = tile_matrix->tiles[j].fmt;
            TileIndex *bits = (TileIndex *)(tile_matrix->data + tile_matrix->tiles[j].bits_off);
            MatValue *vals = (MatValue *)(tile_matrix->data + tile_matrix->tiles[j].bits_off + tile_matrix->tiles[j].bitslen * 16);
            switch (fmt)
            {
            case COO:
            {
                MatIndex nnz = bits[0] + 1;
                for (int k = 0; k < nnz; ++k)
                {
                    row_map[bits[1 + k] / TILE_N][col16 + bits[1 + k] % TILE_N] = vals[k];
                }
            }
            break;
            case CSR:
            {
                for (int ii = 0; ii < TILE_N; ++ii)
                {
                    for (int jj = bits[ii]; jj < bits[ii + 1]; ++jj)
                    {
                        row_map[ii][col16 + TileIndex_CSR_Get(bits + 17, jj)] = vals[jj];
                    }
                }
            }
            break;
            case ELL:
            {
                int max_row_len = TileIndex_CSR_Get(bits, 0) + 1;
                for (int ii = 0; ii < TILE_N; ++ii)
                {
                    for (int jj = 0; jj < max_row_len; ++jj)
                    {
                        TileIndex cur_col = TileIndex_CSR_Get(bits, 1 + ii * max_row_len + jj);
                        if (check_in_bitmap8(tile_matrix->tiles[j].bitmap, ii * TILE_N + cur_col)) {
                            row_map[ii][col16 + cur_col] = vals[ii * max_row_len + jj];
                        }
                    }
                }
            }
            break;
            case HYB:
            {
                int min_row_len = bits[0], coo_cnt = bits[1], elllen = min_row_len * TILE_N, coo_start = (min_row_len * TILE_N + 1) / 2 + 2;
                for (int ii = 0; ii < TILE_N; ++ii)
                {
                    for (int jj = 0; jj < min_row_len; ++jj)
                    {
                        TileIndex cur_col = TileIndex_CSR_Get(bits, 4 + ii * min_row_len + jj);
                        if (check_in_bitmap8(tile_matrix->tiles[j].bitmap, ii * TILE_N + cur_col))
                            row_map[ii][col16 + cur_col] = vals[ii * min_row_len + jj];
                    }
                }
                for (int ii = 0; ii < coo_cnt; ++ii)
                {
                    TileIndex row = bits[coo_start + ii] / TILE_N, col = bits[coo_start + ii] % TILE_N;
                    row_map[row][col16 + col] = vals[elllen + ii];
                }
            }
            break;
            case DRW:
            {
                int cnt = TileIndex_CSR_Get(bits, 0) + 1;
                for (int ii = 0; ii < cnt; ++ii)
                {
                    TileIndex row = TileIndex_CSR_Get(bits, ii + 1);
                    #pragma unroll
                    for (int jj = 0; jj < TILE_N; ++jj)
                    {
                        if (check_in_bitmap8(tile_matrix->tiles[j].bitmap, row * TILE_N + jj))
                            row_map[row][col16 + jj] = vals[ii * TILE_N + jj];
                    }
                }
            }
            break;
            case DCL:
            {
                int cnt = TileIndex_CSR_Get(bits, 0) + 1;
                for (int ii = 0; ii < cnt; ++ii)
                {
                    TileIndex col = TileIndex_CSR_Get(bits, ii + 1);
                    #pragma unroll
                    for (int jj = 0; jj < TILE_N; ++jj)
                    {
                        if (check_in_bitmap8(tile_matrix->tiles[j].bitmap, jj * TILE_N + col))
                            row_map[jj][col16 + col] = vals[ii * TILE_N + jj];
                    }
                }
            }
            break;
            case DNS:
            {
                for (int ii = 0; ii < TILE_N; ++ii)
                for (int jj = 0; jj < TILE_N; ++jj)
                {
                    if (check_in_bitmap8(tile_matrix->tiles[j].bitmap, ii * TILE_N + jj))
                        row_map[ii][col16 + jj] = vals[ii * TILE_N + jj];
                }
            }
            break;
            default:
                break;
            }
        }

        for (int ii = 0; ii < TILE_N; ++ii)
        {
            if (row16 + ii < m)
            for (int j = row_ptr[row16 + ii]; j < row_ptr[row16 + ii + 1]; ++j)
            {
                if (row_map[ii].find(col_idx[j]) == row_map[ii].end())
                {
                    echo(error, "Missing element: (%d, %d)", row16 + ii, col_idx[j]);
                    return;
                }
                if (fabs(row_map[ii][col_idx[j]] - csr_val[j]) > 1e-6)
                {
                    echo(error, "Different element: (%d, %d), %lf, %lf", row16 + ii, col_idx[j], row_map[ii][col_idx[j]], csr_val[j]);
                    return;
                }
            }
        }
    }
    echo(success, "Check passed");
}

void csr_2_mtx(const char*filename, MatIndex m, MatIndex n, MatIndex nnz, MatIndex*row_ptr, MatIndex*col_idx, MatValue*csr_val)
{
    FILE*fp = fopen(filename, "w");
    fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp, "%d %d %d\n", m, n, nnz);
    for (MatIndex i = 0; i < m; ++i)
    {
        for (MatIndex j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
        {
            fprintf(fp, "%d %d %lf\n", i + 1, col_idx[j] + 1, csr_val[j]);
        }
    }
    fclose(fp);
}
