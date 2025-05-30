#ifndef _MMIO_HIGHLEVEL_
#define _MMIO_HIGHLEVEL_

// #ifndef MAT_VAL_TYPE
// #define MAT_VAL_TYPE double
// #endif

#include "mmio.h"

// void exclusive_scan(MAT_PTR_TYPE *input, int length)
// {
//     if (length == 0 || length == 1)
//         return;

//     MAT_PTR_TYPE old_val, new_val;

//     old_val = input[0];
//     input[0] = 0;
//     for (int i = 1; i < length; i++)
//     {
//         new_val = input[i];
//         input[i] = old_val + input[i - 1];
//         old_val = new_val;
//     }
// }


// read matrix infomation from mtx file
int mmio_info(int *m, int *n, int *nnz, int *isSymmetric, char *filename)
{
    int m_tmp, n_tmp, nnz_tmp;
    
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    
    int nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;
    
    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;
    
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }
    
    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_complex( matcode ) )  { isComplex = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }
    
    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;
    
    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric_tmp = 1;
        //printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }
    
    int *csrRowPtr_counter = (int *)malloc((m_tmp+1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));
    
    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    MAT_VAL_TYPE *csrVal_tmp    = (MAT_VAL_TYPE *)malloc(nnz_mtx_report * sizeof(MAT_VAL_TYPE));
    
    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    
    for (int i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        MAT_VAL_TYPE fval, fval_im;
        int ival;
        int returnvalue;
        
        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }
        
        // adjust from 1-based to 0-based
        idxi--;
        idxj--;
        
        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }
    
    if (f != stdin)
        fclose(f);
    
    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }
    
    // exclusive scan for csrRowPtr_counter
    int old_val, new_val;
    
    old_val = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;
    for (int i = 1; i <= m_tmp; i++)
    {
        new_val = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i-1];
        old_val = new_val;
    }
    
    nnz_tmp = csrRowPtr_counter[m_tmp];
    
    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;
    
    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);
    
    return 0;
}

// read matrix infomation from mtx file
int mmio_data(int *csrRowPtr, int *csrColIdx, MAT_VAL_TYPE *csrVal, char *filename)
{
    int m_tmp, n_tmp, nnz_tmp;
    
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    
    int nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;
    
    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;
    
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }
    
    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_complex( matcode ) )  { isComplex = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }
    
    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;
    
    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric_tmp = 1;
        //printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }
    
    int *csrRowPtr_counter = (int *)malloc((m_tmp+1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));
    
    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    MAT_VAL_TYPE *csrVal_tmp    = (MAT_VAL_TYPE *)malloc(nnz_mtx_report * sizeof(MAT_VAL_TYPE));
    
    /* NOTE: when reading in VALUE_TYPEs, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    
    for (int i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        MAT_VAL_TYPE fval, fval_im;
        int ival;
        int returnvalue;
        
        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }
        
        // adjust from 1-based to 0-based
        idxi--;
        idxj--;
        
        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }
    
    if (f != stdin)
        fclose(f);
    
    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }
    
    // exclusive scan for csrRowPtr_counter
    int old_val, new_val;
    
    old_val = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;
    for (int i = 1; i <= m_tmp; i++)
    {
        new_val = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i-1];
        old_val = new_val;
    }
    
    nnz_tmp = csrRowPtr_counter[m_tmp];
    memcpy(csrRowPtr, csrRowPtr_counter, (m_tmp+1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));
    
    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx[offset] = csrColIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
                
                offset = csrRowPtr[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx[offset] = csrRowIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx[offset] = csrColIdx_tmp[i];
                csrVal[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx[offset] = csrColIdx_tmp[i];
            csrVal[offset] = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }
    
    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);
    
    return 0;
}

#endif
