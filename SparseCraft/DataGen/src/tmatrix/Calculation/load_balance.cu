#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
#include <tmatrix/Calculation/load_balance.h>

int lb_spmv_coo_style(BaseMatrix*m, MatIndex **blkcoostylerowidx, MatIndex**blkcoostylerowidx_colstart, MatIndex**blkcoostylerowidx_colstop)
{
    int rowblkblock_tmp = 0;
    for (int blki = 0; blki < m->_m; blki++)
    {
        int balancenumblk = m->tile_row_ptr[blki + 1] - m->tile_row_ptr[blki];
        if (balancenumblk <= PREFETCH_SMEM_TH_SPMV)
            rowblkblock_tmp++;
        else
        {
            rowblkblock_tmp += ceil((double)balancenumblk / (double)PREFETCH_SMEM_TH_SPMV);
        }
    }

    MatIndex *blkcoostylerowidx_tmp = (MatIndex *)malloc(sizeof(MatIndex) * rowblkblock_tmp);;
    memset(blkcoostylerowidx_tmp, 0, sizeof(MatIndex) * rowblkblock_tmp);
    MatIndex *blkcoostylerowidx_colstart_tmp = (MatIndex *)malloc(sizeof(MatIndex) * rowblkblock_tmp);;
    memset(blkcoostylerowidx_colstart_tmp, 0, sizeof(MatIndex) * rowblkblock_tmp);
    MatIndex *blkcoostylerowidx_colstop_tmp = (MatIndex *)malloc(sizeof(MatIndex) * rowblkblock_tmp);
    memset(blkcoostylerowidx_colstop_tmp, 0, sizeof(MatIndex) * rowblkblock_tmp);

    int rowblkblockcnt = 0;
    for (int blki = 0; blki < m->_m; blki++)
    {
        int balancenumblk = m->tile_row_ptr[blki + 1] - m->tile_row_ptr[blki];
        if (balancenumblk <= PREFETCH_SMEM_TH_SPMV)
        {
            blkcoostylerowidx_tmp[rowblkblockcnt] = blki;
            rowblkblockcnt++;
        }
        else
        {
            int numblklocal = ceil((double)balancenumblk / (double)PREFETCH_SMEM_TH_SPMV);
            int lenblklocal = ceil((double)balancenumblk / (double)numblklocal);
            for (int iii = 0; iii < numblklocal; iii++)
            {
                blkcoostylerowidx_tmp[rowblkblockcnt] = blki | 0x80000000; // can generate -0
                blkcoostylerowidx_colstart_tmp[rowblkblockcnt] = m->tile_row_ptr[blki] + iii * lenblklocal;
                if (iii == numblklocal - 1)
                    blkcoostylerowidx_colstop_tmp[rowblkblockcnt] = m->tile_row_ptr[blki] + balancenumblk;
                else
                    blkcoostylerowidx_colstop_tmp[rowblkblockcnt] = m->tile_row_ptr[blki] + (iii + 1) * lenblklocal;

                rowblkblockcnt++;
            }
        }
    }

    cudaMalloc(blkcoostylerowidx, sizeof(MatIndex) * rowblkblockcnt);
    cudaMemcpy(*blkcoostylerowidx, blkcoostylerowidx_tmp, sizeof(MatIndex) * rowblkblockcnt, cudaMemcpyHostToDevice);
    cudaMalloc(blkcoostylerowidx_colstart, sizeof(MatIndex) * rowblkblockcnt);
    cudaMemcpy(*blkcoostylerowidx_colstart, blkcoostylerowidx_colstart_tmp, sizeof(MatIndex) * rowblkblockcnt, cudaMemcpyHostToDevice);
    cudaMalloc(blkcoostylerowidx_colstop, sizeof(MatIndex) * rowblkblockcnt);
    cudaMemcpy(*blkcoostylerowidx_colstop, blkcoostylerowidx_colstop_tmp, sizeof(MatIndex) * rowblkblockcnt, cudaMemcpyHostToDevice);

    free(blkcoostylerowidx_tmp);
    free(blkcoostylerowidx_colstart_tmp);
    free(blkcoostylerowidx_colstop_tmp);

    return rowblkblockcnt;
}

int lb_spmm_coo_style(BaseMatrix*m, MatIndex **blkcoostylerowidx, MatIndex**blkcoostylerowidx_colstart, MatIndex**blkcoostylerowidx_colstop)
{
    int rowblkblock_tmp = 0;
    for (int blki = 0; blki < m->_m; blki++)
    {
        int balancenumblk = m->tile_row_ptr[blki + 1] - m->tile_row_ptr[blki];
        if (balancenumblk <= PREFETCH_SMEM_TH_SPMM)
            rowblkblock_tmp++;
        else
        {
            rowblkblock_tmp += ceil((double)balancenumblk / (double)PREFETCH_SMEM_TH_SPMM);
        }
    }

    MatIndex *blkcoostylerowidx_tmp = (MatIndex *)malloc(sizeof(MatIndex) * rowblkblock_tmp);;
    memset(blkcoostylerowidx_tmp, 0, sizeof(MatIndex) * rowblkblock_tmp);
    MatIndex *blkcoostylerowidx_colstart_tmp = (MatIndex *)malloc(sizeof(MatIndex) * rowblkblock_tmp);
    memset(blkcoostylerowidx_colstart_tmp, 0, sizeof(MatIndex) * rowblkblock_tmp);
    MatIndex *blkcoostylerowidx_colstop_tmp = (MatIndex *)malloc(sizeof(MatIndex) * rowblkblock_tmp);
    memset(blkcoostylerowidx_colstop_tmp, 0, sizeof(MatIndex) * rowblkblock_tmp);

    int rowblkblockcnt = 0;
    for (int blki = 0; blki < m->_m; blki++)
    {
        int balancenumblk = m->tile_row_ptr[blki + 1] - m->tile_row_ptr[blki];
        if (balancenumblk <= PREFETCH_SMEM_TH_SPMM)
        {
            blkcoostylerowidx_tmp[rowblkblockcnt] = blki;
            rowblkblockcnt++;
        }
        else
        {
            int numblklocal = ceil((double)balancenumblk / (double)PREFETCH_SMEM_TH_SPMM);
            int lenblklocal = ceil((double)balancenumblk / (double)numblklocal);
            for (int iii = 0; iii < numblklocal; iii++)
            {
                blkcoostylerowidx_tmp[rowblkblockcnt] = blki | 0x80000000; // can generate -0
                blkcoostylerowidx_colstart_tmp[rowblkblockcnt] = m->tile_row_ptr[blki] + iii * lenblklocal;
                if (iii == numblklocal - 1)
                    blkcoostylerowidx_colstop_tmp[rowblkblockcnt] = m->tile_row_ptr[blki] + balancenumblk;
                else
                    blkcoostylerowidx_colstop_tmp[rowblkblockcnt] = m->tile_row_ptr[blki] + (iii + 1) * lenblklocal;

                rowblkblockcnt++;
            }
        }
    }

    cudaMalloc(blkcoostylerowidx, sizeof(MatIndex) * rowblkblockcnt);
    cudaMemcpy(*blkcoostylerowidx, blkcoostylerowidx_tmp, sizeof(MatIndex) * rowblkblockcnt, cudaMemcpyHostToDevice);
    cudaMalloc(blkcoostylerowidx_colstart, sizeof(MatIndex) * rowblkblockcnt);
    cudaMemcpy(*blkcoostylerowidx_colstart, blkcoostylerowidx_colstart_tmp, sizeof(MatIndex) * rowblkblockcnt, cudaMemcpyHostToDevice);
    cudaMalloc(blkcoostylerowidx_colstop, sizeof(MatIndex) * rowblkblockcnt);
    cudaMemcpy(*blkcoostylerowidx_colstop, blkcoostylerowidx_colstop_tmp, sizeof(MatIndex) * rowblkblockcnt, cudaMemcpyHostToDevice);

    free(blkcoostylerowidx_tmp);
    free(blkcoostylerowidx_colstart_tmp);
    free(blkcoostylerowidx_colstop_tmp);

    return rowblkblockcnt;
}