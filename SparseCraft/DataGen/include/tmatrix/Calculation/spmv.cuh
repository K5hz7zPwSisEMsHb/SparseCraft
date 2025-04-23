#pragma once
#include <tmatrix/DataStructure/TileMatrix.cuh>

__global__ void SpMV_CalculateOnly(
    Tile*At, char*Ad, MatValue*x, MatValue*y
);
