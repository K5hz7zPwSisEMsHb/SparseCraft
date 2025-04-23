#pragma once
#include <tmatrix/DataStructure/TileMatrix.h>

double bin_spgemm_call(sfBIN&, BaseMatrix *A, BaseMatrix *B, BaseMatrix *C, int*d_check, int*d_index, double*d_value, int table_size);
double spgemm_csr_x_csc_tile(BaseMatrix*A, BaseMatrixCSC*B, BaseMatrix*C);