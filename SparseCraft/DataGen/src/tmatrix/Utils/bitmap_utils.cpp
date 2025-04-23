#include <tmatrix/Utils/bitmap_utils.h>

void bit256_2_uint64_arr(bit256 bitmap, uint64_t *arr)
{
    #pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            #pragma unroll
            for (int ii = 0; ii < 8; ++ii)
            {
                #pragma unroll
                for (int jj = 0; jj < 8; ++jj)
                {
                    uint64_t bit = bitmap.test(i * 128 + ii * 16 + j * 8 + jj)? 1: 0;
                    arr[i * 2 + j] |= bit << (ii * 8 + jj);
                }
            }
        }
    }
}

void convert_b64_to_b256(const uint64_t *src, bit256 &dst)
{
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
        {
            uint64_t val = src[i * 2 + j];
            int start_idx = i * 128 + j * 8;
            for (int ii = 0; ii < 8; ++ii)
                for (int jj = 0; jj < 8; ++jj)
                {
                    dst.set(start_idx + ii * 16 + jj, (val >> (ii * 8 + jj)) & 1);
                }
        }
}

void print_uint64(uint64_t *arr)
{
    for (int i = 0; i < 2; i++)
    {
        for (int ii = 0; ii < 8; ii++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int jj = 0; jj < 8; jj++)
                {
                    printf("%ld", (arr[i * 2 + j] >> (ii * 8 + jj)) & 1);
                }
                printf(j == 1? "\n": " ");
            }
        }
        printf("\n");
    }
}

void print_bit256(bit256 bitmap)
{
    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            printf("%d", bitmap.test(i * 16 + j)? 1: 0);
            if (j == 7) printf(" ");
        }
        printf("\n");
        if (i == 7) printf("\n");
    }
}

#define bitmap8_col_x_row(col, row) (((col) * 0xff) & ((row) * 0x0101010101010101))

inline uint64_t bitmap88_x(uint64_t A, uint64_t B, int&intermidiate_counter)
{
    uint64_t C = 0;

#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        uint64_t tmp = bitmap8_col_x_row(A & 0x0101010101010101, B & 0xff);
        intermidiate_counter += __builtin_popcountll(tmp);
        C |= tmp;
        A >>= 1;
        B >>= 8;
    }

    return C;
}

int b64_multiply(const uint64_t *A, const uint64_t *B, uint64_t *C)
{
    int intermidiate_counter = 0;
    C[0] = bitmap88_x(A[0], B[0], intermidiate_counter);
    C[1] = bitmap88_x(A[0], B[1], intermidiate_counter);
    C[2] = bitmap88_x(A[2], B[0], intermidiate_counter);
    C[3] = bitmap88_x(A[2], B[1], intermidiate_counter);
    
    C[0] |= bitmap88_x(A[1], B[2], intermidiate_counter);
    C[1] |= bitmap88_x(A[1], B[3], intermidiate_counter);
    C[2] |= bitmap88_x(A[3], B[2], intermidiate_counter);    
    C[3] |= bitmap88_x(A[3], B[3], intermidiate_counter);
    return intermidiate_counter;
}