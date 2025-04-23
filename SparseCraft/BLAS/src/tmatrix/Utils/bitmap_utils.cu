#include <tmatrix/Utils/bitmap_utils.cuh>

void print_uint64(const uint64_t *arr)
{
    for (int i = 0; i < 2; i++)
    {
        for (int ii = 0; ii < 8; ii++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int jj = 0; jj < 8; jj++)
                {
                    printf("%ld", (arr[i * 2 + j] >> (ii * 8 + jj)) & 1);
                }
                printf(j == 1? "\n": " ");
            }
        }
        printf("\n");
    }
}

bool bitmap_check(const uint64_t *bitmap, TileIndex row, TileIndex col)
{
    int row8 = row >> 3;
    row &= 7;
    int col8 = col >> 3;
    col &= 7;
    return bitmap[(row8 << 1) | col8] >> ((row << 3) | col) & 1;
}

bool bitmap_check(const uint64_t *bitmap, TileIndex idx)
{
    int row = idx >> 4;
    int col = idx & 15;
    return bitmap_check(bitmap, row, col);
}

uint16_t bitmap_count(const uint64_t *bitmap)
{
    return __builtin_popcountll(bitmap[0]) + __builtin_popcountll(bitmap[1]) + __builtin_popcountll(bitmap[2]) + __builtin_popcountll(bitmap[3]); 
}

TileIndex host_b64_count_row(const uint64_t*bitmap, TileIndex row)
{
    int row8 = row >> 3;
    row &= 7;
    TileIndex row_left = (bitmap[row8] >> (row << 3)) & 0xff;
    TileIndex row_right= (bitmap[row8 + 1] >> (row << 3)) & 0xff;
    return __builtin_popcountll(row_left) + __builtin_popcountll(row_right);
}

TileIndex host_b64_count_col(const uint64_t*bitmap, TileIndex col)
{
    int col8 = col >> 3;
    col &= 7;
    uint64_t col_upper = (bitmap[col8] >> col) & 0x0101010101010101;
    uint64_t col_lower = (bitmap[col8 + 2] >> col) & 0x0101010101010101;
    return __builtin_popcountll(col_upper) + __builtin_popcountll(col_lower);
}

TileIndex host_codelen_bitmap2coo(const uint64_t*bitmap, MatIndex*nnz)
{
    uint16_t count = (__builtin_popcountll(bitmap[0]) + __builtin_popcountll(bitmap[1]) + __builtin_popcountll(bitmap[2]) + __builtin_popcountll(bitmap[3])); 
    MatIndex bits = count * 8 + 8;
    *nnz = count;
    return (bits + 7) >> 3;
}

TileIndex host_codelen_bitmap2csr(const uint64_t*bitmap, MatIndex*nnz)
{
    uint16_t count = (__builtin_popcountll(bitmap[0]) + __builtin_popcountll(bitmap[1]) + __builtin_popcountll(bitmap[2]) + __builtin_popcountll(bitmap[3])); 
    MatIndex bits = count * 4 + 17 * 8;
    *nnz = count;
    return (bits + 7) >> 3;
}

TileIndex host_codelen_bitmap2ell(const uint64_t*bitmap, MatIndex*nnz)
{
    MatIndex max_row_len = 0;
    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        MatIndex row_len = host_b64_count_row(bitmap, i);
        max_row_len = std::max(max_row_len, row_len);
    }
    *nnz = max_row_len * 16;
    MatIndex bits = max_row_len * 4 * 16 + 4;
    return (bits + 7) >> 3;
}

TileIndex host_codelen_bitmap2hyb(const uint64_t*bitmap, MatIndex*nnz)
{
    MatIndex min_row_len = 17, row_len[16] = {0}, cnt = 0;

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        row_len[i] = host_b64_count_row(bitmap, i);
        min_row_len = std::min(min_row_len, std::max(row_len[i], 1));
    }

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        cnt += std::max(row_len[i] - min_row_len, 0);
    }
    
    *nnz = min_row_len * 16 + cnt;
    MatIndex bits = ((min_row_len * 16 + 1) / 2 + 2 + cnt) * 8;
    return (bits + 7) >> 3;
}

TileIndex host_codelen_bitmap2drw(const uint64_t*bitmap, MatIndex*nnz)
{
    MatIndex bits = 0;

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        MatIndex row_len = host_b64_count_row(bitmap, i);
        bits += row_len? 4: 0;
    }
    *nnz = bits * 4;
    bits += 4;
    
    return (bits + 7) >> 3;
}

TileIndex host_codelen_bitmap2dcl(const uint64_t*bitmap, MatIndex*nnz)
{
    MatIndex bits = 0;

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        MatIndex col_len = host_b64_count_col(bitmap, i);
        bits += col_len? 4: 0;
    }
    *nnz = bits * 4;
    bits += 4;

    return (bits + 7) >> 3;
}

TileIndex host_codelen_bitmap2dns(const uint64_t*bitmap, MatIndex*nnz)
{
    *nnz = 256;
    return 0;
}

TileIndex host_bitmap2codelen(const uint64_t*bitmap, MatIndex*nnz, TileFormat format)
{
    TileIndex (*codelen_func[])(const uint64_t*, MatIndex*) = {
        host_codelen_bitmap2coo,
        host_codelen_bitmap2csr,
        host_codelen_bitmap2ell,
        host_codelen_bitmap2hyb,
        host_codelen_bitmap2drw,
        host_codelen_bitmap2dcl,
        host_codelen_bitmap2dns
    };
    return codelen_func[format](bitmap, nnz);
}