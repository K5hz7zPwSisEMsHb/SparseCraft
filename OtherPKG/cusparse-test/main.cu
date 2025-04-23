#include <msg.h>
#include <common.h>
#include <mmio_highlevel.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <timer.h>

#include <map>

double test_spmv(int m, int n, int nnz, int*row_ptr, int*col_ptr, double*value, double*convert_time)
{
    int* d_row_ptr, *d_col_ptr;
    double* d_value, *d_x, *d_y;

    cudaMalloc(&d_row_ptr, (m + 1) * sizeof(int));
    cudaMalloc(&d_col_ptr, nnz * sizeof(int));
    cudaMalloc(&d_value, nnz * sizeof(double));

    cudaMemcpy(d_row_ptr, row_ptr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ptr, col_ptr, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, nnz * sizeof(double), cudaMemcpyHostToDevice);
    
    double*x = (double*)malloc(n * sizeof(double));
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        x[i] = 1.0;

    cudaMalloc(&d_x, n * sizeof(MatValue));
    cudaMalloc(&d_y, m * sizeof(MatValue));

    cudaMemcpy(d_x, x, n * sizeof(MatValue), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, m * sizeof(MatValue));
    cudaFree(x);

    cusparseSpMatDescr_t matA;
    Timer t;
    timer_start(t);
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseCreateCsr(&matA, m, n, nnz, d_row_ptr, d_col_ptr, d_value, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    timer_end(t);
    echo(debug, "cusparseCreateCsr: %lf ms", timer_duration(t));
    if (convert_time) {
        *convert_time = timer_duration(t);
    }
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, m, d_y, CUDA_R_64F);

    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0;
    MatValue alpha = 1.0, beta = 0.0;

    cusparseStatus_t status1 = cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, (cusparseSpMVAlg_t)0, &bufferSize1);

    if (status1 != CUSPARSE_STATUS_SUCCESS) {
        echo(error, "STEP1: %s", cusparseGetErrorString(status1));
        return -1;
    }

    cudaMalloc(&dBuffer1, bufferSize1);

    cusparseStatus_t status2 = cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, (cusparseSpMVAlg_t)0, &bufferSize1);

    if (status2 != CUSPARSE_STATUS_SUCCESS) {
        echo(error, "STEP2: %s", cusparseGetErrorString(status2));
        return -1;
    }
    
    cusparseStatus_t status3;
    status3 = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, (cusparseSpMVAlg_t)0, dBuffer1);
    if (status3 != CUSPARSE_STATUS_SUCCESS) {
        echo(error, "STEP3: %s", cusparseGetErrorString(status3));
        return -1;
    }
    
    double first = 0.0; 
    cudaMemcpy(&first, d_y, sizeof(double), cudaMemcpyDeviceToHost);
    echo(debug, "first: %lf", first);

    // time measurement
    // Timer t;
    timer_start(t);
    for (int i = 0; i < 1000; ++i)
        status3 = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, (cusparseSpMVAlg_t)0, dBuffer1);
    cudaDeviceSynchronize();
    timer_end(t);
    double usingTime = timer_duration(t) / 1000;

    cudaFree(d_row_ptr);
    cudaFree(d_col_ptr);
    cudaFree(d_value);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(dBuffer1);
    cudaFree(dBuffer2);
    cusparseDestroySpMat(matA);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroy(handle);

    return 2.0 * nnz / usingTime / 1e6;
}

double test_spmv_bsr(int m, int n, int nnz, int*row_ptr, int*col_ptr, double*value)
{
    int* d_row_ptr, *d_col_ptr;
    double* d_value, *d_x, *d_y;

    cudaMalloc(&d_row_ptr, (m + 1) * sizeof(int));
    cudaMalloc(&d_col_ptr, nnz * sizeof(int));
    cudaMalloc(&d_value, nnz * sizeof(double));

    cudaMemcpy(d_row_ptr, row_ptr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ptr, col_ptr, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, nnz * sizeof(double), cudaMemcpyHostToDevice);
    // convert csr to bsr use cusparseXcsr2bsr
    int blockDim = 16;
    int base;
    int nnzb;
    int *bsrRowPtrC, *bsrColIndC;
    double *bsrValC;
    // create handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    // create matrix descriptor
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    // create bsr matrix descriptor
    cusparseMatDescr_t descrC;
    cusparseCreateMatDescr(&descrC);
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
    cusparseDirection_t dirA = CUSPARSE_DIRECTION_COLUMN;
    int mb = (m + blockDim-1)/blockDim;
    int nb = (n + blockDim-1)/blockDim;
    cudaMalloc((void**)&bsrRowPtrC, sizeof(int) *(mb+1));
    cusparseXcsr2bsrNnz(handle, dirA, m, n,
            descrA, d_row_ptr, d_col_ptr, blockDim,
            descrC, bsrRowPtrC, &nnzb);
    cudaMalloc((void**)&bsrColIndC, sizeof(int)*nnzb);
    cudaMalloc((void**)&bsrValC, sizeof(double)*(blockDim*blockDim)*nnzb);
    cusparseDcsr2bsr(handle, dirA, m, n,
            descrA, d_value, d_row_ptr, d_col_ptr, blockDim,
            descrC, bsrValC, bsrRowPtrC, bsrColIndC);
    // step 2: allocate vector x and vector y large enough for bsrmv
    double *hx = (double*)malloc(n * sizeof(double));
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        hx[i] = 1.0;
    
    cudaMalloc(&d_x, sizeof(double)*n);
    cudaMalloc(&d_y, sizeof(double)*(mb*blockDim));
    cudaMemcpy(d_x, hx, sizeof(double)*n, cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, sizeof(double)*(mb*blockDim));
    free(hx);
    MatValue alpha = 1.0, beta = 0.0;
    // step 3: perform bsrmv
    cusparseDbsrmv(handle, dirA, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb, &alpha,
    descrC, bsrValC, bsrRowPtrC, bsrColIndC, blockDim, d_x, &beta, d_y);
    
    // perform time measurement
    Timer t;
    timer_start(t);
    for (int i = 0; i < 1000; ++i)
        cusparseDbsrmv(handle, dirA, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb, &alpha,
        descrC, bsrValC, bsrRowPtrC, bsrColIndC, blockDim, d_x, &beta, d_y);
    cudaDeviceSynchronize();
    timer_end(t);
    double usingTime = timer_duration(t) / 1000;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_row_ptr);
    cudaFree(d_col_ptr);
    cudaFree(d_value);
    cudaFree(bsrRowPtrC);
    cudaFree(bsrColIndC);
    cudaFree(bsrValC);
    cusparseDestroy(handle);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroyMatDescr(descrC);
    return 2.0 * nnz / usingTime / 1e6;
}

double test_spmm(int m, int n, int nnz, int*row_ptr, int*col_ptr, double*value, int right_n)
{
    // A is a m x n sparse matrix, B is a n x right_n matrix, C is a m x right_n matrix
    cusparseHandle_t handle = NULL;
    cusparseMatDescr_t descrA = NULL;
    cusparseSpMatDescr_t A = NULL;
    cusparseDnMatDescr_t B = NULL;
    cusparseDnMatDescr_t C = NULL;
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    
    MatIndex *d_csrRowPtr, *d_csrColIdx;
    MatValue *d_csrVal, *d_b, *d_c;
    cudaMalloc((void **)&d_csrRowPtr, (m + 1) * sizeof(MatIndex));
    cudaMalloc((void **)&d_csrColIdx, nnz * sizeof(MatIndex));
    cudaMalloc((void **)&d_csrVal, nnz * sizeof(MatValue));
    cudaMalloc((void **)&d_b, n * right_n * sizeof(MatValue));
    cudaMalloc((void **)&d_c, m * right_n * sizeof(MatValue));

    MatValue *b = (MatValue *)malloc(n * right_n * sizeof(MatValue));
    #pragma omp parallel for
    for (int i = 0; i < n * right_n; ++i)
        b[i] = 1.0;

    cudaMemcpy(d_csrRowPtr, row_ptr, (m + 1) * sizeof(MatIndex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdx, col_ptr, nnz * sizeof(MatIndex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal, value, nnz * sizeof(MatValue), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * right_n * sizeof(MatValue), cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, m * right_n * sizeof(MatValue));

    cusparseCreateCsr(&A, m, n, nnz, d_csrRowPtr, d_csrColIdx, d_csrVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnMat(&B, n, right_n, n, d_b, CUDA_R_64F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&C, m, right_n, m, d_c, CUDA_R_64F, CUSPARSE_ORDER_COL);

    MatValue alpha = 1.0, beta = 0.0;
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, B, &beta, C, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);

    cudaMalloc(&dBuffer, bufferSize);
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 1000; ++i)
    {
        cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, B, &beta, C, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float duration;
    cudaEventElapsedTime(&duration, start, stop);
    duration /= 1000;

    // destroy matrix/vector descriptors
    cusparseDestroySpMat(A);
    cusparseDestroyDnMat(B);
    cusparseDestroyDnMat(C);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroy(handle);
    cudaFree(dBuffer);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColIdx);
    cudaFree(d_csrVal);
    cudaFree(d_b);
    cudaFree(d_c);
    return 2.0 * nnz * right_n / duration / 1e6;
}

double test_spmm_coo(int m, int n, int nnz, int*row_ptr, int*col_ptr, double*value, int right_n)
{
    // A is a m x k sparse matrix, B is a k x n matrix, C is a m x n matrix

    int*row_idx = (int *)malloc(nnz * sizeof(int));
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) row_idx[j] = i;
    }

    cusparseHandle_t handle = NULL;
    cusparseMatDescr_t descrA = NULL;
    cusparseSpMatDescr_t A = NULL;
    cusparseDnMatDescr_t B = NULL;
    cusparseDnMatDescr_t C = NULL;
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    
    MatIndex *d_csrRowPtr, *d_csrColIdx;
    MatValue *d_csrVal, *d_b, *d_c;
    cudaMalloc((void **)&d_csrRowPtr, nnz * sizeof(MatIndex));
    cudaMalloc((void **)&d_csrColIdx, nnz * sizeof(MatIndex));
    cudaMalloc((void **)&d_csrVal, nnz * sizeof(MatValue));
    cudaMalloc((void **)&d_b, n * right_n * sizeof(MatValue));
    cudaMalloc((void **)&d_c, m * right_n * sizeof(MatValue));

    MatValue *b = (MatValue *)malloc(n * right_n * sizeof(MatValue));
    #pragma omp parallel for
    for (int i = 0; i < n * right_n; ++i)
        b[i] = 1.0;

    cudaMemcpy(d_csrRowPtr, row_idx, nnz * sizeof(MatIndex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdx, col_ptr, nnz * sizeof(MatIndex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal, value, nnz * sizeof(MatValue), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * right_n * sizeof(MatValue), cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, m * right_n * sizeof(MatValue));

    cusparseCreateCoo(&A, m, n, nnz, d_csrRowPtr, d_csrColIdx, d_csrVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnMat(&B, n, right_n, right_n, d_b, CUDA_R_64F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&C, m, right_n, right_n, d_c, CUDA_R_64F, CUSPARSE_ORDER_ROW);

    MatValue alpha = 1.0, beta = 0.0;
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, B, &beta, C, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);

    cudaMalloc(&dBuffer, bufferSize);
    Timer timer;
    timer_start(timer);
    for (int i = 0; i < 1000; ++i)
        cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, B, &beta, C, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
    cudaDeviceSynchronize();
    timer_end(timer);
    double duration = timer_duration(timer) / 1000;

    // destroy matrix/vector descriptors
    free(row_idx);
    cusparseDestroySpMat(A);
    cusparseDestroyDnMat(B);
    cusparseDestroyDnMat(C);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroy(handle);
    cudaFree(dBuffer);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColIdx);
    cudaFree(d_csrVal);
    cudaFree(d_b);
    cudaFree(d_c);
    return 2.0 * nnz * right_n / duration / 1e6;
}

double test_spgemm(int m, int n, int nnz, int*row_ptr, int*col_ptr, double*value)
{
    MatIndex *d_row_ptr, *d_col_ptr;
    MatValue *d_value, alpha = 1.0, beta = 0.0;

    uint64_t intermidiate = 0;
    #pragma omp parallel for reduction(+:intermidiate)
    for (int i = 0; i < m; ++i)
    {
        uint64_t sum = 0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
        {
            int col = col_ptr[j];
            sum += row_ptr[col + 1] - row_ptr[col];
        }
        intermidiate += sum;
    }

    cudaMalloc(&d_row_ptr, (m + 1) * sizeof(MatIndex));
    cudaMalloc(&d_col_ptr, nnz * sizeof(MatIndex));
    cudaMalloc(&d_value, nnz * sizeof(MatValue));

    cudaMemcpy(d_row_ptr, row_ptr, (m + 1) * sizeof(MatIndex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ptr, col_ptr, nnz * sizeof(MatIndex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, nnz * sizeof(MatValue), cudaMemcpyHostToDevice);

    double duration = 0;
    int repeat = 10;
    for (int i = 0; i < repeat; ++i) {
        cusparseHandle_t handle;
        cusparseCreate(&handle);

        cusparseSpMatDescr_t matA, matB, matC;
        cusparseCreateCsr(&matA, m, n, nnz, d_row_ptr, d_col_ptr, d_value, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        cusparseCreateCsr(&matB, m, n, nnz, d_row_ptr, d_col_ptr, d_value, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        cusparseCreateCsr(&matC, m, n, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        cusparseSpGEMMDescr_t spgemmDesc;
        cusparseSpGEMM_createDescr(&spgemmDesc);

        size_t buffer_size1 = 0, buffer_size2 = 0;
        void *buffer1 = NULL, *buffer2 = NULL;
        
        Timer t1, t2, t3, t4;
        timer_start(t1);
        auto status = cusparseSpGEMM_workEstimation(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &buffer_size1, NULL);
        timer_end(t1);
        
        if (status != CUSPARSE_STATUS_SUCCESS) {
            echo(error, "STEP1: %s", cusparseGetErrorString(status));
            return -1;
        }

        cudaMalloc(&buffer1, buffer_size1);
        timer_start(t2);
        status = cusparseSpGEMM_workEstimation(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &buffer_size1, buffer1);
        timer_end(t2);
        
        if (status != CUSPARSE_STATUS_SUCCESS) {
            echo(error, "STEP2: %s", cusparseGetErrorString(status));
            return -1;
        }

        timer_start(t3);
        status = cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &buffer_size2, NULL);
        timer_end(t3);
        
        if (status != CUSPARSE_STATUS_SUCCESS) {
            echo(error, "STEP3: %s", cusparseGetErrorString(status));
            return -1;
        } 

        cudaMalloc(&buffer2, buffer_size2);

        timer_start(t4);
        cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &buffer_size2, buffer2);
        cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);
        timer_end(t4);
        double used_time = timer_duration(t1) + timer_duration(t2) + timer_duration(t3) + timer_duration(t4);
        // echo(debug, "duration: %lf ms, step1: %lf ms, step2: %lf ms, step3: %lf ms, step4: %lf ms", used_time, timer_duration(t1), timer_duration(t2), timer_duration(t3), timer_duration(t4));
        duration += used_time;

        int64_t Cm, Cn, Cnnz;
        cusparseSpMatGetSize(matC, &Cm, &Cn, &Cnnz);

        if (Cnnz == 0) {
            echo(error, "Cnnz == 0");
            return -1;
        }

        cusparseDestroy(handle);
        cusparseDestroySpMat(matA);
        cusparseDestroySpMat(matB);
        cusparseDestroySpMat(matC);
        cusparseSpGEMM_destroyDescr(spgemmDesc);
        cudaFree(buffer1);
        cudaFree(buffer2);
    }
    duration /= repeat;
    cudaFree(d_row_ptr);
    cudaFree(d_col_ptr);
    cudaFree(d_value);
    echo(debug, "intermidiate: %lu, duration: %lf", intermidiate, duration);
    return intermidiate * 2.0 / duration / 1e6;
}

double test_spgemm_coo(int m, int n, int nnz, int*row_ptr, int*col_ptr, double*value)
{

    int*row_idx = (int *)malloc(nnz * sizeof(int));
    #pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) row_idx[j] = i;
    }

    MatIndex *d_row_ptr, *d_col_ptr;
    MatValue *d_value, alpha = 1.0, beta = 0.0;

    uint64_t intermidiate = 0;
    #pragma omp parallel for reduction(+:intermidiate)
    for (int i = 0; i < m; ++i)
    {
        uint64_t sum = 0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
        {
            int col = col_ptr[j];
            sum += row_ptr[col + 1] - row_ptr[col];
        }
        intermidiate += sum;
    }

    cudaMalloc(&d_row_ptr, nnz * sizeof(MatIndex));
    cudaMalloc(&d_col_ptr, nnz * sizeof(MatIndex));
    cudaMalloc(&d_value, nnz * sizeof(MatValue));

    cudaMemcpy(d_row_ptr, row_idx, nnz * sizeof(MatIndex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ptr, col_ptr, nnz * sizeof(MatIndex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, nnz * sizeof(MatValue), cudaMemcpyHostToDevice);

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseSpMatDescr_t matA, matB, matC;
    cusparseCreateCoo(&matA, m, n, nnz, d_row_ptr, d_col_ptr, d_value, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateCoo(&matB, m, n, nnz, d_row_ptr, d_col_ptr, d_value, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateCoo(&matC, m, n, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseSpGEMM_createDescr(&spgemmDesc);

    size_t buffer_size1 = 0, buffer_size2 = 0;
    void *buffer1 = NULL, *buffer2 = NULL;

    cusparseSpGEMM_workEstimation(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &buffer_size1, NULL);

    cudaMalloc(&buffer1, buffer_size1);
    cusparseSpGEMM_workEstimation(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &buffer_size1, buffer1);

    cusparseSpGEMM_compute(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &buffer_size2, NULL);
    cudaMalloc(&buffer2, buffer_size2);
    Timer t;
    timer_start(t);
    if (
        CUSPARSE_STATUS_INSUFFICIENT_RESOURCES == cusparseSpGEMM_compute(
                                                      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      &alpha, matA, matB, &beta, matC, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &buffer_size2, buffer2))
    {
        echo(error, "insufficient resources\n");
    }
    timer_end(t);

    cusparseDestroy(handle);
    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    cusparseSpGEMM_destroyDescr(spgemmDesc);

    cudaFree(d_row_ptr);
    cudaFree(d_col_ptr);
    cudaFree(d_value);
    cudaFree(buffer1);
    cudaFree(buffer2);
    return intermidiate * 2.0 / timer_duration(t) / 1e6;
}

int main(int argc, char*argv[])
{
    int m, n, nnz, is_symmetric;
    int *row_ptr, *col_ptr;
    double *value;
    int device_id = atoi(argv[argc-2]);
    cudaSetDevice(device_id);
    // print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    echo(info, "Device %d: %s, compute capability: %d.%d", device_id, prop.name, prop.major, prop.minor);

    mmio_allinone(argv[1], &m, &n, &nnz, &is_symmetric, &row_ptr, &col_ptr, &value);
    std::string op = argv[argc - 1];
    double gflops, convert_time = 0;

    if (op == "--spmv")
    {
        gflops = test_spmv(m, n, nnz, row_ptr, col_ptr, value, &convert_time);
    } else if (op == "--spmm")
    {
        int right_n = atoi(argv[argc-3]);
        gflops = test_spmm(m, n, nnz, row_ptr, col_ptr, value, right_n);
    } else if (op == "--spgemm")
    {
        gflops = test_spgemm(m, n, nnz, row_ptr, col_ptr, value);
    } else if (op == "--spmm-coo")
    {
        int right_n = atoi(argv[argc-3]);
        gflops = test_spmm_coo(m, n, nnz, row_ptr, col_ptr, value, right_n);
    } else if (op == "--spgemm-coo")
    {
        gflops = test_spgemm_coo(m, n, nnz, row_ptr, col_ptr, value);
    } else
    {
        echo(error, "Invalid operation: \"%s\"", op.c_str());
    }

    printf("%.3lf,%.3lf\n", convert_time, gflops);

    free(row_ptr);
    free(col_ptr);
    free(value);
    return 0;
}
