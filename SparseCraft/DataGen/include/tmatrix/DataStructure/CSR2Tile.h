#include <tmatrix/DataStructure/TileMatrix.h>
#include <tmatrix/DataStructure/common.h>
#include <tmatrix/Utils/MemoryPool.h>
#include <cstring>

#include <map>

struct dense_tile
{
    bit256 bitmap;
    MatValue val[TILE_N * TILE_N];

    dense_tile() : bitmap(0) {
        memset(val, 0, TILE_N * TILE_N * sizeof(MatValue));
    }
};

TileFormat force_select(bit256 bitmap);
TileFormat tilespmv_select(bit256 bitmap);
TileFormat pixel_select(bit256 bitmap);

void load_model_spgemm_C();
void pixel_spgemm_select_AB(const bit256& left_bitmap, const bit256& right_bitmap, int&fmtA, int&fmtB);
TileFormat pixel_spgemm_select_C(const bit256& A, const bit256& B, const bit256& C, const TileFormat&fmtA, const TileFormat&fmtB);

extern TileFormat (*SELECT_METHODS[])(bit256);
extern const char *SELECT_METHOD_NAMES[];

void dense_tile_2_fmt(dense_tile* t, char *bits, char* val, TileFormat fmt);
BaseMatrix* load_mtx_2_tile(const char* filename, MatIndex*report_nnz, MemoryPool*pool, TileFormat (*select)(bit256), MatIndex&m, MatIndex&n, MatIndex&nnz, MatIndex*&row_ptr, MatIndex*&col_idx, MatValue*&csr_val, int force_fmt = -1);
BaseMatrix* spgemm_load_mtx(const char* filename, MatIndex*report_nnz, MatIndex&m, MatIndex&n, MatIndex&nnz, MatIndex*&row_ptr, MatIndex*&col_idx, MatValue*&csr_val, std::map<MatIndex, dense_tile> *&tile_distributions);

void BaseMatrix_And_CSR_Compare(BaseMatrix* tile_matrix, MatIndex m, MatIndex n, MatIndex nnz, MatIndex*row_ptr, MatIndex*col_idx, MatValue*csr_val);
void csr_2_mtx(const char*filename, MatIndex m, MatIndex n, MatIndex nnz, MatIndex*row_ptr, MatIndex*col_idx, MatValue*csr_val);
