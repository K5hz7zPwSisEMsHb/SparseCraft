#!/bin/bash
cd ASpT_SpMM_GPU
nvcc -std=c++11 -O3 -gencode=arch=compute_120,code=sm_120 dspmm_32.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o dspmm_32
cd ..
