#include <tmatrix/DataStructure/TileMatrix.h>
#include <tmatrix/Utils/MemoryPool.h>
#include <tmatrix/Utils/rbtree.h>
#include <cstring>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <cub/cub.cuh>

#include <map>

void load_model_spgemm_C();

__global__ void tile_format_prediction_single(Tile*tiles, const int n);
__global__ void storage_format_prediction_single(Tile*tiles, const int n);

double set_memory_pool(BaseMatrix*d);
BaseMatrix* load_mtx_2_tile(const char* filename, MatIndex*report_nnz, std::string op, MatIndex&m, MatIndex&n, MatIndex&nnz, MatIndex*&row_ptr, MatIndex*&col_idx, MatValue*&csr_val, double&predict_time);
BaseMatrixCSC* load_mtx_2_csc_tile(const char* filename, MatIndex*report_nnz, std::string predict, MatIndex&m, MatIndex&n, MatIndex&nnz, MatIndex*&row_ptr, MatIndex*&col_idx, MatValue*&csr_val);

void count_storage_block(MatIndex&m, MatIndex&n, MatIndex&nnz, MatIndex*&row_ptr, MatIndex*&col_idx, MatIndex*count);

void BaseMatrix_And_CSR_Compare(BaseMatrix* tile_matrix, MatIndex m, MatIndex n, MatIndex nnz, MatIndex*row_ptr, MatIndex*col_idx, MatValue*csr_val);
void csr_2_mtx(const char*filename, MatIndex m, MatIndex n, MatIndex nnz, MatIndex*row_ptr, MatIndex*col_idx, MatValue*csr_val);
uint16_t bitmap_get_row(const uint64_t *bitmap, TileIndex row);

__global__ void build_A_row_map(const MatIndex *A_row_ptr, const MatIndex *A_col_idx, uint32_t *A_row_map, const int m, const int row_map_len);
__global__ void build_B_col_map(const MatIndex *B_col_ptr, const MatIndex *B_row_idx, uint32_t *B_col_map, const int n, const int col_map_len);