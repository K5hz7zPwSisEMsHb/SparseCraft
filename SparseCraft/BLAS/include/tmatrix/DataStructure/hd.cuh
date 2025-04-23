#pragma once

#include <tmatrix/DataStructure/TileMatrix.h>
#include <cuda_runtime.h>
#include <tmatrix/Utils/msg.h>

#define load_f64_cs(r,l) asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(r) : "l"(l));

BaseMatrix* BaseMatrix_Host_to_Device(BaseMatrix*mm);
BaseMatrixCSC* BaseMatrix_Host_to_Device(BaseMatrixCSC*mm);
void DestroyBaseMatrixHost(BaseMatrix *hmm);
void DestroyBaseMatrixHost(BaseMatrixCSC *hmm);
void DestroyBaseMatrixDevice(BaseMatrix *dmm);
void DestroyBaseMatrixDevice(BaseMatrixCSC *dmm);
