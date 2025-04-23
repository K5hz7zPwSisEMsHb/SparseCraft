#pragma once

#include <tmatrix/common.h>

template<typename T>
void omp_inclusive_scan(T *input, const int length)
{
    if (length == 0 || length == 1)
        return;

    T partial_sum=0;
    #pragma omp simd reduction(inscan, +:partial_sum)
    for (int i = 0; i < length; i++)
    {
        partial_sum += input[i];
        #pragma omp scan inclusive(partial_sum)
        input[i] = partial_sum;
    }
}

template<typename T>
void omp_quicksort_key_val(MatIndex *key, T *val, int left, int right)
{
    if (left >= right)
        return;
    int i = left, j = right;
    int pivot = key[left];
    T pivot_val = val[left];
    while (i < j)
    {
        while (i < j && key[j] >= pivot)
            j--;
        key[i] = key[j];
        val[i] = val[j];
        while (i < j && key[i] <= pivot)
            i++;
        key[j] = key[i];
        val[j] = val[i];
    }
    key[i] = pivot;
    val[i] = pivot_val;
    #pragma omp task
    omp_quicksort_key_val(key, val, left, i - 1);
    #pragma omp task
    omp_quicksort_key_val(key, val, i + 1, right);
}

template<typename T>
void quicksort_key_val(MatIndex *key, T *val, int left, int right)
{
    if (left >= right)
        return;
    int i = left, j = right;
    int pivot = key[left];
    T pivot_val = val[left];
    while (i < j)
    {
        while (i < j && key[j] >= pivot)
            j--;
        key[i] = key[j];
        val[i] = val[j];
        while (i < j && key[i] <= pivot)
            i++;
        key[j] = key[i];
        val[j] = val[i];
    }
    key[i] = pivot;
    val[i] = pivot_val;
    quicksort_key_val(key, val, left, i - 1);
    quicksort_key_val(key, val, i + 1, right);
}