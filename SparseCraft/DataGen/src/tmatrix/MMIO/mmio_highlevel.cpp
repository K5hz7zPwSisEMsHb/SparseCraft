#include <tmatrix/MMIO/mmio.h>
#include <tmatrix/MMIO/mmio_highlevel.h>
#include <stdio.h>
#include <tmatrix/Utils/omp_utils.h>
#include <tmatrix/Utils/msg.h>

void pack_csr_bin(const char *filename, int m, int n, MatIndex nnz, int isSymmetric, MatIndex *csrRowPtr, MatIndex*csrColIdx, MatValue*csrVal)
{
    FILE *f = fopen(filename, "wb");
    fwrite(&m, sizeof(int), 1, f);
    fwrite(&n, sizeof(int), 1, f);
    fwrite(&nnz, sizeof(MatIndex), 1, f);
    fwrite(&isSymmetric, sizeof(int), 1, f);
    fwrite(csrRowPtr, sizeof(MatIndex), m + 1, f);
    fwrite(csrColIdx, sizeof(MatIndex), nnz, f);
    fwrite(csrVal, sizeof(MatValue), nnz, f);
    fclose(f);
}

void unpack_csr_bin(const char *filename, int *m, int *n, MatIndex *nnz, int *isSymmetric, MatIndex **csrRowPtr, MatIndex **csrColIdx, MatValue **csrVal)
{
    FILE *f = fopen(filename, "rb");
    fread(m, sizeof(int), 1, f);
    fread(n, sizeof(int), 1, f);
    fread(nnz, sizeof(MatIndex), 1, f);
    fread(isSymmetric, sizeof(int), 1, f);
    *csrRowPtr = (MatIndex *)malloc((*m + 1) * sizeof(MatIndex));
    *csrColIdx = (MatIndex *)malloc(*nnz * sizeof(MatIndex));
    *csrVal = (MatValue *)malloc(*nnz * sizeof(MatValue));
    fread(*csrRowPtr, sizeof(MatIndex), *m + 1, f);
    fread(*csrColIdx, sizeof(MatIndex), *nnz, f);
    fread(*csrVal, sizeof(MatValue), *nnz, f);
    fclose(f);
}

int mmio_allinone(const char *filename, int *m, int *n, MatIndex *nnz, int *isSymmetric,
                  MatIndex **csrRowPtr, MatIndex **csrColIdx, MatValue **csrVal)
{
    int m_tmp, n_tmp;
    MatIndex nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    MatIndex nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    char new_file_path[128];
    sprintf(new_file_path, "%s.csr", filename);
    if (file_exists(new_file_path))
    {
        unpack_csr_bin(new_file_path, m, n, nnz, isSymmetric, csrRowPtr, csrColIdx, csrVal);
        return 0;
    }

    f = fopen(filename, "r");

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode))
    {
        isPattern = 1; /*printf("type = Pattern\n");*/
    }
    if (mm_is_real(matcode))
    {
        isReal = 1; /*printf("type = real\n");*/
    }
    if (mm_is_complex(matcode))
    {
        isComplex = 1; /*printf("type = real\n");*/
    }
    if (mm_is_integer(matcode))
    {
        isInteger = 1; /*printf("type = integer\n");*/
    }

    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode))
    {
        isSymmetric_tmp = 1;
    }

    MatIndex *csrRowPtr_counter = (MatIndex *)calloc((m_tmp + 1), sizeof(MatIndex));

    MatIndex *csrRowIdx_tmp = (MatIndex *)malloc(nnz_mtx_report * sizeof(MatIndex));
    MatIndex *csrColIdx_tmp = (MatIndex *)malloc(nnz_mtx_report * sizeof(MatIndex));
    MatValue *csrVal_tmp = (MatValue *)malloc(nnz_mtx_report * sizeof(MatValue));

    for (MatIndex i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;

        if (isReal)
        {
            fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        csrRowPtr_counter[idxi]++;

        idxi--;
        idxj--;

        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (MatIndex i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i] + 1]++;
        }
    }

    omp_inclusive_scan(csrRowPtr_counter + 1, m_tmp);

    MatIndex *csrRowPtr_alias = (MatIndex *)malloc((m_tmp + 1) * sizeof(MatIndex));
    nnz_tmp = csrRowPtr_counter[m_tmp];
    MatIndex *csrColIdx_alias = (MatIndex *)malloc(nnz_tmp * sizeof(MatIndex));
    MatValue *csrVal_alias = (MatValue *)malloc(nnz_tmp * sizeof(MatValue));

#pragma omp parallel for
    for (MatIndex i = 0; i < m_tmp + 1; i++)
    {
        csrRowPtr_alias[i] = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = 0;
    }

    if (isSymmetric_tmp)
    {
        for (MatIndex i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                MatIndex offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]]++;
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];

                offset = csrRowPtr_alias[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]]++;
                csrColIdx_alias[offset] = csrRowIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
            }
            else
            {
                MatIndex offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]]++;
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
            }
        }
    }
    else
    {
        for (MatIndex i = 0; i < nnz_mtx_report; i++)
        {
            MatIndex offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            csrColIdx_alias[offset] = csrColIdx_tmp[i];
            csrVal_alias[offset] = csrVal_tmp[i];
        }
    }

    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    *csrRowPtr = csrRowPtr_alias;
    *csrColIdx = csrColIdx_alias;
    *csrVal = csrVal_alias;

    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    pack_csr_bin(new_file_path, *m, *n, *nnz, *isSymmetric, *csrRowPtr, *csrColIdx, *csrVal);

    return 0;
}
