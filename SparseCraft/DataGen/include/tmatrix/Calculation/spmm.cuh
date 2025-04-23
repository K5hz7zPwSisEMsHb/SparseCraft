#pragma once
#include <tmatrix/DataStructure/TileMatrix.cuh>

__global__ void SpMM_CalculateOnly(
    Tile*At, MatValue*Bt, MatValue*Ct, char*Ad, MatIndex right_n, MatIndex ith
);