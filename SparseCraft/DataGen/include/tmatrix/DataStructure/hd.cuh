#pragma once

#include <tmatrix/DataStructure/TileMatrix.cuh>
#include <cuda_runtime.h>
#include <tmatrix/Utils/msg.h>

#define load_f64_cs(r,l) asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(r) : "l"(l));

BaseMatrix* BaseMatrix_Host_to_Device(BaseMatrix*mm);
void DestroyBaseMatrixHost(BaseMatrix *hmm);
void DestroyBaseMatrixDevice(BaseMatrix *dmm);
