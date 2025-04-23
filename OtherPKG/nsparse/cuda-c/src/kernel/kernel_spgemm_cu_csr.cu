#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <cuda.h>
// #include <helper_cuda.h>
// #include <cusparse_v2.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <nsparse.h>

__global__ void set_intprod_per_row(int *d_arpt, int *d_acol,
                                    const int* d_brpt,
                                    long long int *d_max_row_nz,
                                    int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) {
        return;
    }
    int nz_per_row = 0;
    int j;
    for (j = d_arpt[i]; j < d_arpt[i + 1]; j++) {
        nz_per_row += d_brpt[d_acol[j] + 1] - d_brpt[d_acol[j]];
    }
    d_max_row_nz[i] = nz_per_row;
}

void get_spgemm_flop(sfCSR *a, sfCSR *b,
                     int M, long long int *flop)
{
    int GS, BS;
    long long int *d_max_row_nz;

    BS = MAX_LOCAL_THREAD_NUM;
    cudaMalloc((void **)&(d_max_row_nz), sizeof(long long int) * M);
  
    GS = div_round_up(M, BS);
    set_intprod_per_row<<<GS, BS>>>(a->d_rpt, a->d_col,
                                    b->d_rpt,
                                    d_max_row_nz,
                                    M);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in kernel: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
  
    long long int *tmp = (long long int *)malloc(sizeof(long long int) * M);
    cudaMemcpy(tmp, d_max_row_nz, sizeof(long long int) * M, cudaMemcpyDeviceToHost);
    *flop = thrust::reduce(thrust::device, d_max_row_nz, d_max_row_nz + M);

    (*flop) *= 2;
    cudaFree(d_max_row_nz);
}

