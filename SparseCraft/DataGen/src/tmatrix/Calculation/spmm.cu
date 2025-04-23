#include <tmatrix/Calculation/spmm.cuh>

__global__ void SpMM_CalculateOnly(
    Tile*At, MatValue*Bd, MatValue*Cd, char*Ad, MatIndex right_n, MatIndex ith
)
{
    At = At + ith;
    Ad = Ad + ith * 2320;
    
    MatIndex gwarp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int TN = 8;
    TileFormat fmt = At->fmt;
    MatValue rc[TN] = {0};
    const int lwarp_id = threadIdx.x >> 5, lane_id = threadIdx.x & 31;

    __shared__ MatValue s_dense_tile_array[16 * TN * 2];
    __shared__ MatValue s_out_tile_array[16 * TN * 2];
    MatValue * s_dense_tile = s_dense_tile_array + lwarp_id * 16 * TN;
    MatValue * s_out_tile = s_out_tile_array + lwarp_id * 16 * TN;

    TileIndex*bits = (TileIndex*) Ad + At[0].bits_off;
    MatValue*A_val = (MatValue*) (Ad + At[0].bits_off + At[0].bitslen * 16);

    if (lane_id < TN)
    {
        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            s_out_tile[i * TN + lane_id] = 0;
        }
    }

    if (lane_id < TN) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) s_dense_tile[i * TN + lane_id] = Bd[i * 8 + lane_id];
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
    #pragma unroll
    for (int l = 0; l < TN; l++)
    {
        rc[l] += __shfl_down_sync(0xFFFFFFFF, rc[l], 16);
        if (lane_id < 16)
            s_out_tile[lane_id * TN + l] += rc[l];
    }
    __syncthreads();

    if (gwarp_id == 0 && lane_id == 0) Cd[0] = s_out_tile[lane_id];
}
