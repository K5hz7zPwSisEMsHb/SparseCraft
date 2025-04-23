#include <tmatrix/Calculation/spmv.cuh>

#include <tmatrix/DataStructure/CSR2Tile.cuh>
#include <tmatrix/DataStructure/hd.cuh>
#include <tmatrix/Calculation/load_balance.h>
#include <tmatrix/Calculation/ShulkerBox/spmv.cuh>
#include <tmatrix/Utils/timer.h>
#include <tmatrix/Utils/msg.h>
#include <tmatrix/MMIO/mmio_highlevel.h>
#include <cusparse.h>

__global__ void BaseMatrixSpMV_Cuda_Core_Load_Balance(
    MatIndex m, MatIndex n, MatIndex rowblks,
    MatIndex *block_row_ptr, MatIndex *block_col_start, MatIndex *block_col_end, MatIndex *A_rowptr, MatIndex *A_colind, 
    Tile *A_tiles, char* A_data,
    MatValue *x, MatValue *y)
{
    MatIndex gwarp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (gwarp_id >= rowblks) return;

    MatIndex row16 = block_row_ptr[gwarp_id];
    MatIndex signbit = row16 & 0x80000000;
    row16 ^= signbit;
    
    MatIndex start_Aj = signbit? block_col_start[gwarp_id] : A_rowptr[row16], end_Aj = signbit? block_col_end[gwarp_id] : A_rowptr[row16 + 1];
    int lane_id = threadIdx.x & 31, warp_id = threadIdx.x >> 5;

    __shared__ MatValue sx[64], sy[64];

    MatValue *warp_x = sx + warp_id * TILE_N;
    MatValue *warp_y = sy + warp_id * TILE_N;
    
    if (lane_id < TILE_N) warp_y[lane_id] = 0;
    __syncwarp();

    MatValue sumsum = spm_dv(A_tiles, A_data, A_colind, x, y, lane_id, start_Aj, end_Aj, warp_x, warp_y);

    if (lane_id < TILE_N && sumsum != 0) 
    {
        MatIndex y_off = row16 << 4 | lane_id;
        if (signbit) atomicAdd(y + y_off, sumsum);
        else y[y_off] = sumsum;
    }
}

double*spmv_result(MatIndex m, MatIndex n, MatIndex nnz, MatIndex *row_ptr, MatIndex *col_idx, MatValue *csr_val)
{
    MatValue *y = (MatValue *)calloc(m, sizeof(MatValue));

    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        MatValue sum = 0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            sum += csr_val[j];
        }
        y[i] = sum;
    }

    return y;
}

int test_spmv(const char* filename, int repeat, int device)
{
    echo(info, "Test SpMV with matrix file: %s, repeat: %d", filename, repeat);
    MatIndex report_nnz;
    MatIndex m, n, nnz, *row_ptr, *col_idx;
    MatValue *csr_val;
    MatIndex isSymmetric;
    mmio_allinone(filename, &m, &n, &nnz, &isSymmetric, &row_ptr, &col_idx, &csr_val);
    double predict_time = 0;
    BaseMatrix* mat = load_mtx_2_tile(filename, &report_nnz, "pixel", m, n, nnz, row_ptr, col_idx, csr_val, predict_time);

    if (mat == nullptr)
    {
        echo(error, "Failed to load matrix from file: %s", filename);
        return 1;
    }

    echo(success, "Matrix loaded from file: \"%s\", m: %d, n: %d, nnz: %d (report: %d)", filename, mat->meta_m, mat->meta_n, mat->meta_nnz, report_nnz);
    MatValue* x = (MatValue*)malloc(sizeof(MatValue) * mat->meta_n), *y = (MatValue*)malloc(sizeof(MatValue) * mat->meta_m);

    #pragma omp parallel for
    for (MatIndex i = 0; i < mat->meta_n; ++i) x[i] = 1.0;

    cudaInit(device);
    MatValue* d_x, *d_y;
    // 负载均衡，一个block算4个块
    MatIndex warpsPerBlock = 2;
    Timer timer_pre;
    timer_start(timer_pre);

    MatIndex *block_row, *block_col_start, *block_col_end;
    MatIndex row_blocks = lb_spmv_coo_style(mat, &block_row, &block_col_start, &block_col_end);
    MatIndex using_blocks = ceil((double)row_blocks / (double)warpsPerBlock);
    
    timer_end(timer_pre);
    echo(success, "TileMatrix SpMV block partition time: %f ms", timer_duration(timer_pre));

    cudaMalloc(&d_x, sizeof(MatValue) * mat->meta_n);
    cudaMalloc(&d_y, sizeof(MatValue) * mat->meta_m);
    
    cudaMemcpy(d_x, x, sizeof(MatValue) * mat->meta_n, cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, sizeof(MatValue) * mat->meta_m);

    BaseMatrix*d_mat = BaseMatrix_Host_to_Device(mat);
    echo(success, "Matrix loaded to device");
    free(x);
    cudaMemset(d_y, 0, sizeof(MatValue) * mat->meta_m);

    BaseMatrixSpMV_Cuda_Core_Load_Balance<<<using_blocks, warpsPerBlock * 32>>>(
        mat->_m, mat->_n, row_blocks,
        block_row, block_col_start, block_col_end, d_mat->tile_row_ptr, d_mat->tile_col_idx, d_mat->tiles, d_mat->data,
        d_x, d_y
    );
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        echo(error, "Error: %s", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(y, d_y, sizeof(MatValue) * mat->meta_m, cudaMemcpyDeviceToHost);
    MatValue *y_cpu = spmv_result(m, n, nnz, row_ptr, col_idx, csr_val);
    bool check_flag = true;
    // #pragma omp parallel for
    for (MatIndex i = 0; i < mat->meta_m; ++i) {
        if (fabs(y_cpu[i] - y[i]) > 1e-6 && check_flag) {
            check_flag = false;
            echo(error, "Error: y[%d] = %lf, y_cpu[%d] = %lf", i, y[i], i, y_cpu[i]);
            break;
        }
    }
    
    if (check_flag) echo(success, "Check passed");
    else echo(error, "Check failed");

    double duration = 1e9;

    for (int i = 0; i < repeat; ++i) {
        cudaMemset(d_y, 0, sizeof(MatValue) * mat->meta_m);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        BaseMatrixSpMV_Cuda_Core_Load_Balance<<<using_blocks, warpsPerBlock * 32>>>(
            mat->_m, mat->_n, row_blocks,
            block_row, block_col_start, block_col_end, d_mat->tile_row_ptr, d_mat->tile_col_idx, d_mat->tiles, d_mat->data,
            d_x, d_y
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaDeviceSynchronize();
        duration = min(milliseconds, duration);
    }
    
    // duration = timer_duration(timer_spmv) / repeat;
    
    echo(success, "TileMatrix SpMV: %.2lf ms / per time, GFLOPS: %.2lf", duration, 2.0 * report_nnz / duration / 1e6);
    printf("%d,%d,%d,%.3lf,%.3lf,%.3lf\n", mat->meta_m, mat->meta_n, mat->meta_nnz, predict_time, /*cusparse_gflops,*/ duration, 2.0 * report_nnz / duration / 1e6);

    free(row_ptr);
    free(col_idx);
    free(csr_val);
    free(y);
    free(y_cpu);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(block_row);
    cudaFree(block_col_start);
    cudaFree(block_col_end);
    DestroyBaseMatrixHost(mat);
    DestroyBaseMatrixDevice(d_mat);

    return 0;
}