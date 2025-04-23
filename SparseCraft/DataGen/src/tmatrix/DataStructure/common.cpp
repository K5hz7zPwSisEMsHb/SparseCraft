#include <tmatrix/DataStructure/common.h>

TileIndex codelen_bitmap2coo(bit256 bitmap, MatIndex*nnz)
{
    MatIndex bits = bitmap.count() * 8 + 8;
    *nnz = bitmap.count();
    return (bits + 7) >> 3;
}

TileIndex codelen_bitmap2csr(bit256 bitmap, MatIndex*nnz)
{
    MatIndex bits = bitmap.count() * 4 + 17 * 8;
    *nnz = bitmap.count();
    return (bits + 7) >> 3;
}

TileIndex codelen_bitmap2ell(bit256 bitmap, MatIndex*nnz)
{
    MatIndex max_row_len = 0;
    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        MatIndex row_len = (bitmap & mask_row).count();
        max_row_len = std::max(max_row_len, row_len);
        bitmap >>= 16;
    }
    *nnz = max_row_len * 16;
    MatIndex bits = max_row_len * 4 * 16 + 4;
    return (bits + 7) >> 3;
}

TileIndex codelen_bitmap2hyb(bit256 bitmap, MatIndex*nnz)
{
    MatIndex min_row_len = 17, row_len[16] = {0}, cnt = 0;

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        row_len[i] = (bitmap & mask_row).count();
        min_row_len = std::min(min_row_len, std::max(row_len[i], 1));
        bitmap >>= 16;
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

TileIndex codelen_bitmap2drw(bit256 bitmap, MatIndex*nnz)
{
    MatIndex bits = 0;

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        MatIndex row_len = (bitmap & mask_row).count();
        bitmap >>= 16;
        bits += row_len? 4: 0;
    }
    *nnz = bits * 4;
    bits += 4;
    
    return (bits + 7) >> 3;
}

TileIndex codelen_bitmap2dcl(bit256 bitmap, MatIndex*nnz)
{
    MatIndex bits = 0;

    #pragma unroll
    for (MatIndex i = 0; i < 16; ++i)
    {
        MatIndex row_len = (bitmap & mask_col).count();
        bitmap >>= 1;
        bits += row_len? 4: 0;
    }
    *nnz = bits * 4;
    bits += 4;

    return (bits + 7) >> 3;
}

TileIndex codelen_bitmap2dns(bit256 bitmap, MatIndex*nnz)
{
    *nnz = 256;
    return 0;
}

TileIndex bitmap2codelen(bit256 bitmap, MatIndex*nnz, TileFormat format)
{
    TileIndex (*codelen_func[])(bit256, MatIndex*) = {
        codelen_bitmap2coo,
        codelen_bitmap2csr,
        codelen_bitmap2ell,
        codelen_bitmap2hyb,
        codelen_bitmap2drw,
        codelen_bitmap2dcl,
        codelen_bitmap2dns
    };
    return codelen_func[format](bitmap, nnz);
}