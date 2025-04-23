#include <tmatrix/DataStructure/TileMatrix.cuh>

#ifndef PREFETCH_SMEM_TH_SPMV
#define PREFETCH_SMEM_TH_SPMV 4
#endif

#ifndef PREFETCH_SMEM_TH_SPMM
#define PREFETCH_SMEM_TH_SPMM 8
#endif

int lb_spmv_coo_style(BaseMatrix*m, MatIndex **blkcoostylerowidx, MatIndex**blkcoostylerowidx_colstart, MatIndex**blkcoostylerowidx_colstop);
int lb_spmm_coo_style(BaseMatrix*m, MatIndex **blkcoostylerowidx, MatIndex**blkcoostylerowidx_colstart, MatIndex**blkcoostylerowidx_colstop);
