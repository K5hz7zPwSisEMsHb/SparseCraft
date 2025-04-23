#pragma once
#include <common.h>

int mmio_allinone(const char *filename, int *m, int *n, MatIndex *nnz, int *isSymmetric,
                  MatIndex **csrRowPtr, MatIndex **csrColIdx, MatValue **csrVal);