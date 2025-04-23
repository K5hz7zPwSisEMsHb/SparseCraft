#pragma once

#include <tmatrix/DataStructure/common.h>
#include <tmatrix/DataStructure/TileMatrix.cuh>
#include <tmatrix/Utils/omp_utils.h>

#define SeqMatrixDim 256


BaseMatrix*empty_matrix_header()
{
    BaseMatrix *m = (BaseMatrix *)malloc(sizeof(BaseMatrix));
    m->meta_m = 16;
    m->meta_n = 16;
    m->_m = m->meta_m >> 4;
    m->_n = 1;
    m->_nnz = m->meta_m >> 4;
    return m;
}

void dense_tile_2_coo(bit256 bitmap, TileIndex *bits, MatValue* val, MatValue template_val)
{
    bits[0] = bitmap.count() - 1;
    
    #pragma unroll
    for (int i = 0, idx = 1; i < 256; ++i)
    {
        if (bitmap.test(i))
        {
            bits[idx] = i;
            val[idx - 1] = template_val;
            idx++;
        }
    }
}

void dense_tile_2_csr(bit256 bitmap, TileIndex *bits, MatValue* val, MatValue template_val)
{
    TileIndex nnz = 0;

    #pragma unroll
    for (MatIndex i = 0; i < TILE_N; ++i)
    {
        unsigned short bitrow = (bitmap & mask_row).to_ulong(), nzr = __builtin_popcount(bitrow);
        for (MatIndex j = 0; j < nzr; ++j)
        {
            unsigned short lowbit = bitrow & -bitrow, col = __builtin_ctz(lowbit);
            TileIndex_CSR_Set(bits, TILE_N * 2 + 2 + nnz + j, col);
            val[nnz + j] = template_val;
            bitrow ^= lowbit;
        }
        nnz += nzr;
        bits[i + 1] = nnz;
        bitmap >>= TILE_N;
    }
    bits[0] = 0;
}

void dense_tile_2_ell(bit256 bitmap, TileIndex *bits, MatValue* val, MatValue template_val)
{
    MatIndex max_row_len = 0;
    ushort row_map[16] = {0};

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        row_map[i] = (bitmap & mask_row).to_ulong();
        MatIndex row_len = __builtin_popcount(row_map[i]);
        max_row_len = std::max(max_row_len, row_len);
        bitmap >>= 16;
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
            val[i * max_row_len + j] = template_val;
            row_map[i] ^= lowbit;
        }
        memset(val + i * max_row_len + row_len, 0, (max_row_len - row_len) * sizeof(MatValue));
    }
}

void dense_tile_2_hyb(bit256 bitmap, TileIndex *bits, MatValue* val, MatValue template_val)
{
    MatIndex min_row_len = 17, row_len[16] = {0}, cnt = 0;
    uint16_t row_map[16] = {0};

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        row_map[i] = (bitmap & mask_row).to_ulong();
        row_len[i] = __builtin_popcount(row_map[i]);
        min_row_len = std::min(min_row_len, std::max(row_len[i], 1));
        bitmap >>= 16;
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
            val[i * min_row_len + j] = template_val;
            row_map[i] ^= lowbit;
        }

        for (MatIndex j = 0; j < rest; ++j)
        {
            uint16_t lowbit = row_map[i] & -row_map[i];
            TileIndex col = __builtin_ctz(lowbit);
            bits[ell_using_size + cnt] = i << 4 | col;
            val[min_row_len * 16 + cnt] = template_val;
            cnt++;
            row_map[i] ^= lowbit;
        }
    }
    bits[1] = cnt;
}

void dense_tile_2_drw(bit256 bitmap, TileIndex *bits, MatValue* val, MatValue template_val)
{
    TileIndex cnt = 0;

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        MatIndex row_len = (bitmap & mask_row).count();
        if (row_len)
        {
            TileIndex_CSR_Set(bits, cnt + 1, i);
            // memcpy(val + cnt * TILE_N, dense + i * TILE_N, TILE_N * sizeof(MatValue));
            #pragma unroll
            for (MatIndex j = 0; j < 16; ++j)
            {
                val[cnt * TILE_N + j] = template_val;
            }
            cnt++;
        }
        bitmap >>= 16;
    }
    TileIndex_CSR_Set(bits, 0, cnt - 1);
}

void dense_tile_2_dcl(bit256 bitmap, TileIndex *bits, MatValue* val, MatValue template_val)
{
    TileIndex cnt = 0;
    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        MatIndex col_len = (bitmap & mask_col).count();
        if (col_len)
        {
            TileIndex_CSR_Set(bits, cnt + 1, i);
            #pragma unroll
            for (MatIndex j = 0; j < 16; ++j)
            {
                val[cnt * TILE_N + j] = template_val;
            }
            cnt++;
        }
        bitmap >>= 1;
    }
    TileIndex_CSR_Set(bits, 0, cnt - 1);
}

void dense_tile_2_dns(bit256 bitmap, TileIndex *bits, MatValue* val, MatValue template_val)
{
    #pragma unroll
    for (MatIndex i = 0; i < 256; ++i)
    {
        val[i] = template_val;
    }
}

void dense_tile_2_fmt(bit256& t, char *bits, char* val, TileFormat fmt, MatValue template_val)
{
    void (*bitmap2fmt[7])(bit256 bitmap, TileIndex *bits, MatValue* val, MatValue template_val) = {
        dense_tile_2_coo,
        dense_tile_2_csr,
        dense_tile_2_ell,
        dense_tile_2_hyb,
        dense_tile_2_drw,
        dense_tile_2_dcl,
        dense_tile_2_dns
    };
    bitmap2fmt[fmt](t, (TileIndex*)bits, (MatValue*)val, template_val);
}

BaseMatrix* empty_coo_matrix()
{
    BaseMatrix *m = empty_matrix_header();

    m->tiles        = (Tile *) malloc(m->_nnz * sizeof(Tile));
    m->data         = (char *) calloc(m->_nnz * 2320, sizeof(char));
    return m;
}

BaseMatrix*DeviceEmptyMatrix(BaseMatrix*hm)
{
    BaseMatrix *m = (BaseMatrix *)malloc(sizeof(BaseMatrix));
    m->meta_m = hm->meta_m;
    m->meta_n = hm->meta_n;
    m->_m = hm->_m;
    m->_n = hm->_n;
    m->_nnz = hm->_nnz;
    cudaMalloc(&m->tiles, m->_nnz * sizeof(Tile));
    cudaMalloc(&m->data, m->_nnz * 2320 * sizeof(char));
    // cudaMemcpy(m->tiles, hm->tiles, m->_nnz * sizeof(Tile), cudaMemcpyHostToDevice);
    // cudaMemcpy(m->data, hm->data, m->_nnz * 2320llu * sizeof(char), cudaMemcpyHostToDevice);
    return m;
}

void BaseMatrix_Host_to_Device(BaseMatrix*hm, BaseMatrix*dm)
{
    cudaMemcpy(dm->tiles, hm->tiles, dm->_nnz * sizeof(Tile),           cudaMemcpyHostToDevice);
    cudaMemcpy(dm->data,  hm->data,  dm->_nnz * 2320llu * sizeof(char), cudaMemcpyHostToDevice);
}

void MatrixGenSame(bit256 distribution, BaseMatrix*m, TileFormat fmt)
{
    MatIndex valslen, _raw_valslen;
    MatIndex bitslen = bitmap2codelen(distribution, &valslen, fmt);
    _raw_valslen = valslen - 1;
    bitslen = (bitslen + 15) / 16 * 16;
    valslen = (valslen + 1) / 2 * 2;
    uint64_t bytes = bitslen + valslen * 8;

    #pragma omp parallel for
    for (int i = 0; i < m->_nnz; ++i)
    {
        m->tiles->bits_off = i * bytes;
        m->tiles->bitslen = bitslen / 16;
        m->tiles->valslen = _raw_valslen;

        dense_tile_2_fmt(distribution, m->data + m->tiles->bits_off, m->data + m->tiles->bits_off + bitslen, fmt, 1);
    }
}

