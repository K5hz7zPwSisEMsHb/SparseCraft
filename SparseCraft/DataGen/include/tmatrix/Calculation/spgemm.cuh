#pragma once
#include <tmatrix/DataStructure/TileMatrix.cuh>

// __global__ void SpGEMM_CalculateOnly(
//     Tile*At, Tile*Bt, Tile*Ct, char*Ad, char*Bd, char*Cd, MatValue*tmp_buffer
// );

__global__ void SpGEMM_B_CalculateOnly(
    Tile*At, char*Ad, Tile*Bt, char*Bd, MatValue*tmp_buffer
);