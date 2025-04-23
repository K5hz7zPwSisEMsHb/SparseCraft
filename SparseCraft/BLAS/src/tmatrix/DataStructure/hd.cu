#include <tmatrix/DataStructure/hd.cuh>

BaseMatrix* BaseMatrix_Host_to_Device(BaseMatrix*mm)
{
    BaseMatrix *dmm_host = (BaseMatrix *)malloc(sizeof(BaseMatrix));

    MatIndex* tile_row_ptr;
    MatIndex* tile_col_idx;
    Tile* tiles;
    char* data;

    cudaError_t e = cudaMalloc(&tile_row_ptr, (mm->_m + 1) * sizeof(MatIndex)); 
    if (e != cudaSuccess || tile_row_ptr == NULL) {
        echo(error, "cudaMalloc tile_row_ptr failed\n");
    }
    e = cudaMalloc(&tile_col_idx, mm->_nnz * sizeof(MatIndex));
    if (e != cudaSuccess || tile_col_idx == NULL) {
        echo(error, "cudaMalloc tile_col_idx failed\n");
    }
    e = cudaMalloc(&tiles, mm->_nnz * sizeof(Tile));
    if (e != cudaSuccess || tiles == NULL) {
        echo(error, "cudaMalloc tiles failed\n");
    }
    e = cudaMalloc(&data, mm->_data_len);
    if (e != cudaSuccess || data == NULL) {
        echo(error, "cudaMalloc data failed\n");
    }

    e = cudaMemcpy(tile_row_ptr, mm->tile_row_ptr, (mm->_m + 1) * sizeof(MatIndex), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        echo(error, "cudaMemcpy tile_row_ptr failed\n");
    }
    e = cudaMemcpy(tile_col_idx, mm->tile_col_idx, mm->_nnz * sizeof(MatIndex),     cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        echo(error, "cudaMemcpy tile_col_idx failed\n");
    }
    e = cudaMemcpy(tiles,        mm->tiles,        mm->_nnz * sizeof(Tile),         cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        echo(error, "cudaMemcpy tiles failed\n");
    }
    e = cudaMemcpy(data,         mm->data,         mm->_data_len,                   cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        echo(error, "cudaMemcpy data failed\n");
    }

    dmm_host->meta_m = mm->meta_m;
    dmm_host->meta_n = mm->meta_n;
    dmm_host->meta_nnz = mm->meta_nnz;
    dmm_host->_m = mm->_m;
    dmm_host->_n = mm->_n;
    dmm_host->_nnz = mm->_nnz;
    dmm_host->tile_row_ptr = tile_row_ptr;
    dmm_host->tile_col_idx = tile_col_idx;
    dmm_host->tiles        = tiles;
    dmm_host->data         = data;

    return dmm_host;
}

BaseMatrixCSC* BaseMatrix_Host_to_Device(BaseMatrixCSC*mm)
{
    BaseMatrixCSC *dmm_host = (BaseMatrixCSC *)malloc(sizeof(BaseMatrixCSC));

    MatIndex* tile_col_ptr;
    MatIndex* tile_row_idx;
    Tile* tiles;
    char* data;

    cudaError_t e = cudaMalloc(&tile_col_ptr, (mm->_m + 1) * sizeof(MatIndex)); 
    if (e != cudaSuccess || tile_col_ptr == NULL) {
        echo(error, "cudaMalloc tile_col_ptr failed\n");
    }
    e = cudaMalloc(&tile_row_idx, mm->_nnz * sizeof(MatIndex));
    if (e != cudaSuccess || tile_row_idx == NULL) {
        echo(error, "cudaMalloc tile_row_idx failed\n");
    }
    e = cudaMalloc(&tiles, mm->_nnz * sizeof(Tile));
    if (e != cudaSuccess || tiles == NULL) {
        echo(error, "cudaMalloc tiles failed\n");
    }
    e = cudaMalloc(&data, mm->_data_len);
    if (e != cudaSuccess || data == NULL) {
        echo(error, "cudaMalloc data failed\n");
    }

    e = cudaMemcpy(tile_col_ptr, mm->tile_col_ptr, (mm->_m + 1) * sizeof(MatIndex), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        echo(error, "cudaMemcpy tile_col_ptr failed\n");
    }
    e = cudaMemcpy(tile_row_idx, mm->tile_row_idx, mm->_nnz * sizeof(MatIndex),     cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        echo(error, "cudaMemcpy tile_row_idx failed\n");
    }
    e = cudaMemcpy(tiles,        mm->tiles,        mm->_nnz * sizeof(Tile),         cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        echo(error, "cudaMemcpy tiles failed\n");
    }
    e = cudaMemcpy(data,         mm->data,         mm->_data_len,                   cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        echo(error, "cudaMemcpy data failed\n");
    }

    dmm_host->meta_m = mm->meta_m;
    dmm_host->meta_n = mm->meta_n;
    dmm_host->meta_nnz = mm->meta_nnz;
    dmm_host->_m = mm->_m;
    dmm_host->_n = mm->_n;
    dmm_host->_nnz = mm->_nnz;
    dmm_host->tile_col_ptr = tile_col_ptr;
    dmm_host->tile_row_idx = tile_row_idx;
    dmm_host->tiles        = tiles;
    dmm_host->data         = data;

    return dmm_host;
}

BaseMatrix*BaseMatrix_Device_to_Host(BaseMatrix*mm)
{
    BaseMatrix *hmm_device = (BaseMatrix *)malloc(sizeof(BaseMatrix));

    MatIndex* tile_row_ptr = (MatIndex *)malloc((mm->_m + 1) * sizeof(MatIndex));
    MatIndex* tile_col_idx = (MatIndex *)malloc(mm->_nnz * sizeof(MatIndex));
    Tile* tiles = (Tile *)malloc(mm->_nnz * sizeof(Tile));
    char* data = (char *)malloc(mm->_data_len);

    cudaError_t e = cudaMemcpy(tile_row_ptr, mm->tile_row_ptr, (mm->_m + 1) * sizeof(MatIndex), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        echo(error, "cudaMemcpy tile_row_ptr failed\n");
    }
    e = cudaMemcpy(tile_col_idx, mm->tile_col_idx, mm->_nnz * sizeof(MatIndex),     cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        echo(error, "cudaMemcpy tile_col_idx failed\n");
    }
    e = cudaMemcpy(tiles,        mm->tiles,        mm->_nnz * sizeof(Tile),         cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        echo(error, "cudaMemcpy tiles failed\n");
    }
    e = cudaMemcpy(data,         mm->data,         mm->_data_len,                   cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        echo(error, "cudaMemcpy data failed\n");
    }

    hmm_device->meta_m = mm->meta_m;
    hmm_device->meta_n = mm->meta_n;
    hmm_device->meta_nnz = mm->meta_nnz;
    hmm_device->_m = mm->_m;
    hmm_device->_n = mm->_n;
    hmm_device->_nnz = mm->_nnz;
    hmm_device->tile_row_ptr = tile_row_ptr;
    hmm_device->tile_col_idx = tile_col_idx;
    hmm_device->tiles        = tiles;
    hmm_device->data         = data;

    return hmm_device;
}

void DestroyBaseMatrixHost(BaseMatrix *hmm)
{
    free(hmm->tile_row_ptr);
    free(hmm->tile_col_idx);
    free(hmm->tiles);
    // free(hmm->data);
    free(hmm);
}

void DestroyBaseMatrixHost(BaseMatrixCSC *hmm)
{
    free(hmm->tile_col_ptr);
    free(hmm->tile_row_idx);
    free(hmm->tiles);
    // free(hmm->data);
    free(hmm);
}

void DestroyBaseMatrixDevice(BaseMatrix *dmm)
{
    cudaFree(dmm->tile_row_ptr);
    cudaFree(dmm->tile_col_idx);
    cudaFree(dmm->tiles);
    cudaFree(dmm->data);
    free(dmm);
}

void DestroyBaseMatrixDevice(BaseMatrixCSC *dmm)
{
    cudaFree(dmm->tile_col_ptr);
    cudaFree(dmm->tile_row_idx);
    cudaFree(dmm->tiles);
    cudaFree(dmm->data);
    free(dmm);
}