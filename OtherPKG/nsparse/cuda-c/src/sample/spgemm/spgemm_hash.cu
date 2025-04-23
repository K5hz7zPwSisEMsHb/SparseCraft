#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#include <math.h>
#include <algorithm>

#include <cuda.h>
// #include <helper_cuda.h>

#include <nsparse.h>

double spgemm_csr(sfCSR *a, sfCSR *b, sfCSR *c)
{

    int i;
  
    long long int flop_count;
    cudaEvent_t event[2];
    float msec, ave_msec, flops;
    
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
    printf("Device compute capability: %d.%d\n", prop.major, prop.minor);

    for (i = 0; i < 2; i++) {
        cudaEventCreate(&(event[i]));
    }
  
    /* Memcpy A and B from Host to Device */
    csr_memcpy(a);
    csr_memcpy(b);
  
    /* Count flop of SpGEMM computation */
    get_spgemm_flop(a, b, a->M, &flop_count);
    printf("flop_count: %lld\n", flop_count);
    /* Execution of SpGEMM on Device */
    ave_msec = 0;
    cudaEventRecord(event[0], 0);
    spgemm_kernel_hash(a, b, c);
    cudaEventRecord(event[1], 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&msec, event[0], event[1]);
    ave_msec += msec;
  
    flops = (float)(flop_count) / 1000 / 1000 / ave_msec;
  
    printf("SpGEMM using CSR format (Hash-based): %s, %f[GFLOPS], %f[ms]\n", a->matrix_name, flops, ave_msec);

    csr_memcpyDtH(c);
    release_csr(*c);
  
    release_csr(*a);
    release_csr(*b);
    for (i = 0; i < 2; i++) {
        cudaEventDestroy(event[i]);
    }
    return flops;
}

/* Main Function */
int main(int argc, char **argv)
{
    sfCSR mat_a, mat_b, mat_c;
    
    printf("sizeof real: %d\n", sizeof(real));
    /* Set CSR reading from MM file */
    init_csr_matrix_from_file(&mat_a, argv[1]);
    init_csr_matrix_from_file(&mat_b, argv[1]);
  
    double flops = spgemm_csr(&mat_a, &mat_b, &mat_c);

    release_cpu_csr(mat_a);
    release_cpu_csr(mat_b);
    release_cpu_csr(mat_c);

    printf("%lf\n", flops);
    return 0;
}
