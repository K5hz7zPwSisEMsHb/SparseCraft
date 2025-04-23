#include <tmatrix/Calculation/spgemm.cuh>

#include <tmatrix/MMIO/mmio_highlevel.h>
#include <tmatrix/DataStructure/predict.cuh>
#include <tmatrix/DataStructure/CSR2Tile.cuh>
#include <tmatrix/DataStructure/hd.cuh>
#include <tmatrix/Utils/bitmap_utils.cuh>
#include <tmatrix/Utils/omp_utils.h>
#include <tmatrix/Utils/timer.h>
#include <tmatrix/Utils/msg.h>
#include <map>
#include <vector>

#include <omp.h>
#include <cusparse.h>
#include <tmatrix/Calculation/nsparse_asm.cuh>
#include <tmatrix/Calculation/spgemm_subkernels.cuh>
#include <tmatrix/Calculation/bin_spgemm.cuh>
#include <tmatrix/Utils/rbtree.h>

#define MAGIC_NUM1 0.1
#define MAGIC_NUM2 0.001

BaseMatrix *BaseMatrix_SpGEMM_Symbol_only_select_C(BaseMatrix *A, BaseMatrix *B, sfBIN*bin, oaktree*model, double &used_time, double&predict_time, const double ratio)
{
    BaseMatrix* dC = new BaseMatrix();
    dC->meta_m = A->meta_m;
    dC->meta_n = B->meta_n;
    dC->_m = A->_m;
    dC->_n = B->_n;
    cudaMalloc(&dC->tile_row_ptr, (dC->_m + 1) * sizeof(MatIndex));

    Timer t;
    Tile first;
    used_time = 0;
    if (B->_n > 512 * 32)
    {
        init_bin(bin, A->_m);
        used_time += set_max_bin(bin, A->tile_row_ptr, A->tile_col_idx, B->tile_row_ptr, A->_m);
        used_time += set_row_nnz(bin, A->tile_row_ptr, A->tile_col_idx, A->tiles, B->tile_row_ptr, B->tile_col_idx, B->tiles, dC->tile_row_ptr, A->_m, &dC->_nnz);
        used_time += set_min_bin(bin, A->_m);
        cudaMalloc(&dC->tile_col_idx, dC->_nnz * sizeof(MatIndex));
        cudaMalloc(&dC->tiles, dC->_nnz * sizeof(Tile));
        timer_start(t);
        calculate_value_col_bin(A->tile_row_ptr, A->tile_col_idx, A->tiles, B->tile_row_ptr, B->tile_col_idx, B->tiles, dC->tile_row_ptr, dC->tile_col_idx, dC->tiles, bin, A->_m, B->_n);
        timer_end(t);
        used_time += timer_duration(t);
    }
    else
    {
        int num_threads = 128;
        int num_blocks = ceil((double)A->_m / (double)(4));
        timer_start(t);
        tile_spgemm_step1_cuda_spa_kernel<<<num_blocks, num_threads>>>(A->tile_row_ptr, A->tile_col_idx, A->tiles, A->_m,
                                                                          B->tile_row_ptr, B->tile_col_idx, B->tiles, B->_n,
                                                                          dC->tile_row_ptr);
        cudaDeviceSynchronize();
        timer_end(t);
        used_time += timer_duration(t);
        // thrust::exclusive_scan(thrust::device, dC->tile_row_ptr, dC->tile_row_ptr + dC->_m + 1, dC->tile_row_ptr, 0);
        {
            // cub
            void *d_temp_storage = NULL;
            size_t temp_storage_bytes = 0;
            timer_start(t);
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, dC->tile_row_ptr, dC->_m + 1);
            timer_end(t);
            used_time += timer_duration(t);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            timer_start(t);
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, dC->tile_row_ptr, dC->_m + 1);
            timer_end(t);
            used_time += timer_duration(t);
            cudaFree(d_temp_storage);
        }
        cudaMemcpy(&dC->_nnz, dC->tile_row_ptr + dC->_m, sizeof(int), cudaMemcpyDeviceToHost);
        uint64_t *buffer;
        cudaMalloc(&buffer, (4 * dC->_n * dC->_m) * sizeof(uint64_t));
        cudaMemset(buffer, 0, (4 * dC->_n * dC->_m) * sizeof(uint64_t));
        cudaMalloc(&dC->tile_col_idx, dC->_nnz * sizeof(MatIndex));
        cudaMalloc(&dC->tiles, dC->_nnz * sizeof(Tile));
        
        timer_start(t);
        tile_spgemm_step1_numeric_cuda_spa_kernel<<<(dC->_nnz + 3) / 4, 128>>>(A->tile_row_ptr, A->tile_col_idx, A->tiles, A->_m, 
                                                                            B->tile_row_ptr, B->tile_col_idx, B->tiles, B->_n,
                                                                            dC->tile_row_ptr, dC->tile_col_idx, dC->tiles, buffer);
        cudaDeviceSynchronize();
        timer_end(t);
        used_time += timer_duration(t);
        cudaFree(buffer);
    }
    // cudaMemcpy(&first, dC->tiles, sizeof(Tile), cudaMemcpyDeviceToHost);
    // int bytes = first.bitslen * 16 + (first.valslen + 1) * sizeof(MatValue);
    // int valslen = first.valslen + 1;
    // int bitslen = first.bitslen;
    // echo(rule, "");
    // echo(info, "First tile: fmt: \"%d\", bitslen: %d, valslen: %d, bitmap: %016llx, %016llx, %016llx, %016llx", first.fmt, bitslen, valslen, first.bitmap[0], first.bitmap[1], first.bitmap[2], first.bitmap[3]);
    timer_start(t);
    if (ratio > MAGIC_NUM1 || ratio < MAGIC_NUM2)
        tile_format_prediction_single<<<(dC->_nnz + 255) / 256, 256>>>(dC->tiles, dC->_nnz);
    else 
        oaktree_prediction_single<<<(dC->_nnz + 255) / 256, 256>>>(model, dC->tiles, dC->_nnz);
    // oaktree_prediction_single<<<(dC->_nnz + 255) / 256, 256>>>(model, dC->tiles, dC->_nnz);
    cudaDeviceSynchronize();
    timer_end(t);
    predict_time = timer_duration(t);
    used_time += predict_time;
    used_time += set_memory_pool(dC);
    
    return dC;
}

void print_first_tile(BaseMatrix*m)
{
    const char* tile_format_name[] = {"COO", "CSR", "ELL", "HYB", "DRW", "DCL", "DNS"};
    Tile first;
    char first_data[2308] = {0};
    cudaMemcpy(&first, m->tiles, sizeof(Tile), cudaMemcpyDeviceToHost);
    int bytes = first.bitslen * 16 + (first.valslen + 1) * sizeof(MatValue);
    int valslen = first.valslen + 1;
    int bitslen = first.bitslen;
    MatValue *vals = (MatValue *)(first_data + first.bits_off + bitslen * 16);
    echo(rule, "");
    echo(info, "First tile: fmt: \"%s\", bitslen: %d, valslen: %d, bitmap: %016llx, %016llx, %016llx, %016llx", tile_format_name[first.fmt], bitslen, valslen, first.bitmap[0], first.bitmap[1], first.bitmap[2], first.bitmap[3]);
    int first_row_len;
    cudaMemcpy(&first_row_len, m->tile_row_ptr + 1, sizeof(int), cudaMemcpyDeviceToHost);
    int *col = (int *)malloc(first_row_len * sizeof(int));
    cudaMemcpy(col, m->tile_col_idx, first_row_len * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Col (len: %d): ", first_row_len);
    for (int i = 0; i < first_row_len; ++i)
    {
        printf("%d ", col[i]);
    }
    printf("\n");
    cudaMemcpy(first_data, m->data, bytes, cudaMemcpyDeviceToHost);
    printf("Val: ");
    for (int i = 0; i < valslen; ++i)
    {
        // echo(debug, "C[%d]: %.2lf", i, vals[i]);
        printf("%.2lf ", vals[i]);
    }
    printf("\n");
}

int test_spgemm(const char *filename, int repeat, int device)
{
    echo(markdown, "# Test SpGEMM with matrix file: %s, repeat: %d", filename, repeat);
    MatIndex report_nnz;
    MatIndex Am, An, Annz, *Arow_ptr = nullptr, *Acol_idx = nullptr;
    MatValue *Acsr_val = nullptr;

    cudaInit(device);
    MatIndex isSymmetric;
    mmio_allinone(filename, &Am, &An, &Annz, &isSymmetric, &Arow_ptr, &Acol_idx, &Acsr_val);

    if (Am != An)
    {
        echo(error, "Matrix is not square: %s", filename);
        free(Arow_ptr);
        free(Acol_idx);
        free(Acsr_val);
        return -1;
    }

    uint64_t intermidiate_result = 0;
#pragma omp parallel for reduction(+ : intermidiate_result)
    for (MatIndex i = 0; i < Am; ++i)
    {
        uint64_t sub_intermidiate_result = 0;
        for (MatIndex Aj = Arow_ptr[i]; Aj < Arow_ptr[i + 1]; ++Aj)
        {
            MatIndex Acol = Acol_idx[Aj];
            sub_intermidiate_result += Arow_ptr[Acol + 1] - Arow_ptr[Acol];
        }
        intermidiate_result += sub_intermidiate_result;
    }

    double predict_time = 0;
    BaseMatrix *A;
    double ratio = intermidiate_result * 1.0 / Am / An;
    echo(info, "Matrix ratio: %lf", ratio);
    if (ratio > MAGIC_NUM1 || ratio < MAGIC_NUM2) {
        A = load_mtx_2_tile(filename, &report_nnz, "tile", Am, An, Annz, Arow_ptr, Acol_idx, Acsr_val, predict_time);
    } else {
        A = load_mtx_2_tile(filename, &report_nnz, "pixel", Am, An, Annz, Arow_ptr, Acol_idx, Acsr_val, predict_time);
    }
    BaseMatrix *B = load_mtx_2_tile(filename, &report_nnz, "tile", Am, An, Annz, Arow_ptr, Acol_idx, Acsr_val, predict_time);

    if (A == nullptr || B == nullptr)
    {
        echo(error, "Failed to load matrix file: %s", filename);
        return -1;
    }

    echo(success, "Matrix loaded from file: \"%s\", m: %d, n: %d, nnz: %d (report: %d)", filename, A->meta_m, A->meta_n, A->meta_nnz, report_nnz);

    BaseMatrix *d_A = BaseMatrix_Host_to_Device(A);
    BaseMatrix *d_B = BaseMatrix_Host_to_Device(B);

    oaktree*host_tree = load_oaktree("model/oaktree.bin");
    if (host_tree == nullptr)
    {
        echo(error, "Failed to load oaktree");
        return -1;
    }
    oaktree*device_tree = oaktree_to_device(host_tree);
    if (device_tree == nullptr)
    {
        echo(error, "Failed to load oaktree to device");
        return -1;
    }
    free(host_tree);

    sfBIN bin;
    bin.inited = 0;
    double symbolic_time = 0;
    BaseMatrix *d_C = BaseMatrix_SpGEMM_Symbol_only_select_C(d_A, d_B, &bin, device_tree, symbolic_time, predict_time, ratio);
    if (d_C == nullptr)
    {
        echo(error, "Failed to calculate SpGEMM Symbol");
        return -1;
    }
    echo(success, "[Sym] Matrix multiplied, m: %d, n: %d, nz blk: %d", d_C->meta_m, d_C->meta_n, d_C->_nnz);

    double min_symbolic_time = symbolic_time;
    double min_predict_time = predict_time;
    for (int i = 0; i < repeat; ++i)
    {
        BaseMatrix*tmp = BaseMatrix_SpGEMM_Symbol_only_select_C(d_A, d_B, &bin, device_tree, symbolic_time, predict_time, ratio);
        DestroyBaseMatrixDevice(tmp);
        if (symbolic_time < min_symbolic_time)
            min_symbolic_time = symbolic_time;
        if (predict_time < min_predict_time)
            min_predict_time = predict_time;
    }
    symbolic_time = min_symbolic_time;
    predict_time = min_predict_time;
    echo(info, "Symbolic time: %lf, Predict time: %lf", symbolic_time, predict_time);

    char*c_data = (char*)malloc(d_C->_data_len);
    cudaMemcpy(c_data, d_C->data, d_C->_data_len, cudaMemcpyDeviceToHost);

    double duration = 1e9;
    // if (bin.inited == 0) init_bin(&bin, A->_m);
    // cudaMemcpy(bin.d_row_nz, d_C->tile_row_ptr, sizeof(MatIndex) * (d_C->_m + 1), cudaMemcpyDeviceToDevice);
    // set_min_bin_for_numeric(&bin, A->_m);
    DestroyBaseMatrixHost(B);
    DestroyBaseMatrixDevice(d_B);
    
    BaseMatrixCSC* B_csc = load_mtx_2_csc_tile(filename, &report_nnz, "tile", Am, An, Annz, Arow_ptr, Acol_idx, Acsr_val);
    if (B_csc == nullptr)
    {
        echo(error, "Failed to load matrix file: %s", filename);
        return -1;
    }
    BaseMatrixCSC *d_B_csc = BaseMatrix_Host_to_Device(B_csc);

    // bin_spgemm_call(bin, d_A, d_B, d_C, NULL, NULL, NULL, bin.max_nz + 1);
    spgemm_csr_x_csc_tile(d_A, d_B_csc, d_C);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        echo(error, "Error: %s", cudaGetErrorString(err));
        return -1;
    }

    // print_first_tile(d_C);

#ifdef CHECK_RESULT
    echo(info, "Start Check");

    cudaMemcpy(C->data, d_C->data, C->_data_len, cudaMemcpyDeviceToHost);
    BaseMatrix_And_CSR_Compare(C, Cm, Cn, Cnnz, Crow_ptr, Ccol_idx, Ccsr_val);
#endif

    for (int i = 0; i < repeat; i++)
    {
        cudaMemcpy(d_C->data, c_data, d_C->_data_len, cudaMemcpyHostToDevice);
        // double used_time = bin_spgemm_call(bin, d_A, d_B, d_C, NULL, NULL, NULL, bin.max_nz + 1);
        double used_time = spgemm_csr_x_csc_tile(d_A, d_B_csc, d_C);
        duration = std::min(duration, used_time);
    }
    // release_bin(bin);
    duration += symbolic_time;
    echo(success, "Matrix multiplied on GPU, passed!");
    echo(info, "intermidiate result: %ld", intermidiate_result);

    printf("%lf ms\n", duration);
    printf("%lf\n", 2.0 * intermidiate_result / duration / 1e6);

    free(Arow_ptr);
    free(Acol_idx);
    free(Acsr_val);
    DestroyBaseMatrixHost(A);
    DestroyBaseMatrixHost(B_csc);
    DestroyBaseMatrixDevice(d_A);
    DestroyBaseMatrixDevice(d_B_csc);
    DestroyBaseMatrixDevice(d_C);
    return 0;
}
