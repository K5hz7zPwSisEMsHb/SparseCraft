#include <tmatrix/Utils/msg.h>
#include <tmatrix/DataStructure/matrixGen.h>
#include <tmatrix/DataStructure/hd.cuh>
#include <tmatrix/Utils/timer.h>
#include <tmatrix/Utils/bitmap_utils.h>
#include <tmatrix/Calculation/spmv.cuh>
#include <tmatrix/Calculation/spmm.cuh>
#include <tmatrix/Calculation/spgemm.cuh>
#include <cstring>

using namespace std;

int spmv(int argc, char **argv)
{
    const char *tile_format_name[] = {"COO", "CSR", "ELL", "HYB", "DRW", "DCL", "DNS"};
    MatIndex repeat = 500;
    FILE *distributions_file = fopen(argv[1], "r");
    if (distributions_file == NULL)
    {
        echo(error, "Error: %s", strerror(errno));
        return -1;
    }
    int items;
    fscanf(distributions_file, "%d", &items);
    while (fgetc(distributions_file) != '\n');
    // items = min(items, 350000);
    echo(info, "Total %d distributions", items);
    bit256 *distributions = new bit256[items];
    char line[257];

    for (int i = 0; i < items; i++)
    {
        fscanf(distributions_file, "%s", line);
        bit256 tb(line);
        TileIndex offset = 256 - strlen(line);
        if (strncmp(tb.to_string().c_str() + offset, line, strlen(line) - 1) != 0)
        {
            echo(error, "Error: %s\n       %s", line, tb.to_string().c_str());
        }
        distributions[i] = tb;
    }
    fclose(distributions_file);
    echo(info, "Read distributions finished");

    Timer timer;

    Tile *dA_tile, *hA_tile = new Tile[min(items, 100000)];
    char *dA_data, *hA_data = new char[2320 * min(items, 100000)];
    MatValue *dB_data, hB_data[16];
    MatValue *dC_data;

    #pragma omp parallel for
    for (int i = 0; i < 16; ++i)
        hB_data[i] = 1;

    cudaMalloc(&dA_tile, sizeof(Tile) * min(items, 100000));
    cudaMalloc(&dA_data, 2320 * min(items, 100000));
    cudaMalloc(&dB_data, 16 * sizeof(MatValue));
    cudaMalloc(&dC_data, 25600 * sizeof(MatValue));
    cudaMemcpy(dB_data, hB_data, 16 * sizeof(MatValue), cudaMemcpyHostToDevice);
    char filename[256];

    double *best_gflops = new double[min(items, 100000)];
    double *best_duration = new double[min(items, 100000)];
    TileFormat *best_fmt = new TileFormat[min(items, 100000)];

    int use_m = 800;

    for (int i = 0; i < items; i += 100000)
    {
        int end_j = std::min(i + 100000, items);
        #pragma omp parallel for
        for (int j = 0; j < end_j - i; ++j)
        {
            best_fmt[j] = COO;
            best_gflops[j] = 0;
            best_duration[j] = 0;
        }
        
        for (TileFormat tfA = COO; tfA <= DNS; tfA++)
        {
            #pragma omp parallel for
            for (int j = 0; j < end_j - i; ++j)
            {
                memset(&hA_tile[j], 0, sizeof(Tile));
                memset(hA_data + j * 2320, 0, sizeof(hA_data));
                MatIndex A_bitslen, A_valslen;
                A_bitslen = bitmap2codelen(distributions[i + j], &A_valslen, tfA);
                hA_tile[j].valslen = A_valslen - 1;
                A_valslen = (A_valslen + 1) / 2 * 2;
                A_bitslen = (A_bitslen + 15) / 16 * 16;
                hA_tile[j].bits_off = 0;
                hA_tile[j].bitslen = A_bitslen / 16;
                dense_tile_2_fmt(distributions[i + j], hA_data + j * 2320, hA_data + j * 2320 + A_bitslen, tfA, 1);
                hA_tile[j].fmt = tfA;
            }

            cudaMemcpy(dA_tile, hA_tile, sizeof(Tile) * (end_j - i), cudaMemcpyHostToDevice);
            cudaMemcpy(dA_data, hA_data, 2320 * (end_j - i), cudaMemcpyHostToDevice);

            for (int j = 0; j < end_j - i; ++j)
            {
                echo(start_status, "Format %s: Distribution %d / %d", tile_format_name[tfA], i + j, items);
                int used_bytes = (hA_tile[j].bitslen) * 16 + (hA_tile[j].valslen + 1) * 8;

                SpMV_CalculateOnly<<<use_m, 64>>>(dA_tile + j, dA_data + j * 2320, dB_data, dC_data);
                cudaDeviceSynchronize();
                cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess)
                {
                    echo(error, "Error: %s", cudaGetErrorString(e));
                    return -1;
                }
                timer_start(timer);
                for (int _ = 0; _ < repeat; ++_)
                {
                    SpMV_CalculateOnly<<<use_m, 64>>>(dA_tile + j, dA_data + j * 2320, dB_data, dC_data);
                }
                cudaDeviceSynchronize();
                timer_end(timer);
                double duration = timer_duration(timer) / repeat + (use_m * 2 * used_bytes) / 1e9;
                double gflops = use_m * 4 * distributions[i + j].count() / duration / 1e6;

                if (gflops >= best_gflops[j])
                {
                    best_duration[j] = duration;
                    best_gflops[j] = gflops;
                    best_fmt[j] = tfA;
                }
            }
        }

        for (int j = 0; j < end_j - i; ++j)
        {
            sprintf(filename, "gen/spmv/%s.txt", tile_format_name[best_fmt[j]]);
            FILE *output_file = fopen(filename, "a");
            if (output_file == NULL)
            {
                echo(error, "Failed to open file %s!", filename);
                return -1;
            }
            fprintf(output_file, "%s %lf %lf\n", distributions[i + j].to_string().c_str(), best_duration[j], best_gflops[j]);
            fclose(output_file);
        }
    }
    echo(stop_status, "");
    return 0;
}

int spmm(int argc, char **argv)
{
    const char *tile_format_name[] = {"COO", "CSR", "ELL", "HYB", "DRW", "DCL", "DNS"};
    MatIndex repeat = 100;
    FILE *distributions_file = fopen(argv[1], "r");
    if (distributions_file == NULL)
    {
        echo(error, "Error: %s", strerror(errno));
        return -1;
    }
    int items;
    fscanf(distributions_file, "%d", &items);
    while (fgetc(distributions_file) != '\n');
    // items = min(items, 40000);
    echo(info, "Total %d distributions", items);
    bit256 *distributions = new bit256[items];
    char line[257];

    for (int i = 0; i < items; i++)
    {
        fscanf(distributions_file, "%s", line);
        bit256 tb(line);
        TileIndex offset = 256 - strlen(line);
        if (strncmp(tb.to_string().c_str() + offset, line, strlen(line) - 1) != 0)
        {
            echo(error, "Error: %s\n       %s", line, tb.to_string().c_str());
        }
        distributions[i] = tb;
    }
    fclose(distributions_file);
    echo(info, "Read distributions finished");

    Timer timer;

    Tile *dA_tile, *hA_tile = new Tile[min(items, 100000)];
    char *dA_data, *hA_data = new char[2320 * min(items, 100000)];
    MatValue *dB_data, hB_data[128];
    MatValue *dC_data;

    for (int i = 0; i < 128; ++i)
        hB_data[i] = 1;

    cudaMalloc(&dA_tile, sizeof(Tile) * min(items, 100000));
    cudaMalloc(&dA_data, 2320 * min(items, 100000));
    cudaMalloc(&dB_data, 128 * sizeof(MatValue));
    cudaMalloc(&dC_data, 25600 * sizeof(MatValue));
    cudaMemcpy(dB_data, hB_data, 128 * sizeof(MatValue), cudaMemcpyHostToDevice);
    char filename[256];

    double *best_gflops = new double[min(items, 100000)];
    double *best_duration = new double[min(items, 100000)];
    TileFormat *best_fmt = new TileFormat[min(items, 100000)];

    for (int i = 0; i < items; i += 100000)
    {
        int end_j = std::min(i + 100000, items);

        #pragma omp parallel for
        for (int j = 0; j < end_j - i; ++j)
        {
            best_fmt[j] = COO;
            best_gflops[j] = 0;
            best_duration[j] = 0;
        }

        for (TileFormat tfA = COO; tfA <= DNS; tfA++)
        {
            #pragma omp parallel for
            for (int j = 0; j < end_j - i; ++j)
            {
                memset(&hA_tile[j], 0, sizeof(Tile));
                memset(hA_data + j * 2320, 0, sizeof(hA_data));
                MatIndex A_bitslen, A_valslen;
                A_bitslen = bitmap2codelen(distributions[i + j], &A_valslen, tfA);
                hA_tile[j].valslen = A_valslen - 1;
                A_valslen = (A_valslen + 1) / 2 * 2;
                A_bitslen = (A_bitslen + 15) / 16 * 16;
                hA_tile[j].bits_off = 0;
                hA_tile[j].bitslen = A_bitslen / 16;
                dense_tile_2_fmt(distributions[i + j], hA_data + j * 2320, hA_data + j * 2320 + A_bitslen, tfA, 1);
                hA_tile[j].fmt = tfA;
            }

            cudaMemcpy(dA_tile, hA_tile, sizeof(Tile) * (end_j - i), cudaMemcpyHostToDevice);
            cudaMemcpy(dA_data, hA_data, 2320 * (end_j - i), cudaMemcpyHostToDevice);

            for (int j = 0; j < end_j - i; ++j)
            {
                echo(start_status, "Format %s: Distribution %d / %d", tile_format_name[tfA], i + j, items);
                SpMM_CalculateOnly<<<400, 64>>>(dA_tile, dB_data, dC_data, dA_data, 8, j);
                cudaDeviceSynchronize();
                cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess)
                {
                    echo(error, "Error: %s", cudaGetErrorString(e));
                    return -1;
                }

                timer_start(timer);
                for (int _ = 0; _ < repeat; ++_)
                {
                    SpMM_CalculateOnly<<<400, 64>>>(dA_tile, dB_data, dC_data, dA_data, 8, j);
                }
                cudaDeviceSynchronize();
                timer_end(timer);
                double duration = timer_duration(timer) / repeat;
                double gflops = 400 * 4 * distributions[i + j].count() * 8 / duration / 1e6;
                
                if (gflops > best_gflops[j])
                {
                    best_duration[j] = duration;
                    best_gflops[j] = gflops;
                    best_fmt[j] = tfA;
                }
            }
        }

        for (int j = 0; j < end_j - i; ++j)
        {
            sprintf(filename, "gen/spmm/%s.txt", tile_format_name[best_fmt[j]]);
            FILE *output_file = fopen(filename, "a");
            if (output_file == NULL)
            {
                echo(error, "Failed to open file %s!", filename);
                return -1;
            }
            fprintf(output_file, "%s %lf %lf\n", distributions[i + j].to_string().c_str(), best_duration[j], best_gflops[j]);
            fclose(output_file);
        }
    }
    echo(stop_status, "");
    return 0;
}

int spgemmB(int argc, char **argv)
{
    const char *tile_format_name[] = {"COO", "CSR", "ELL", "HYB", "DRW", "DCL", "DNS"};
    MatIndex repeat = 100;
    FILE *distributions_file = fopen(argv[1], "r");
    if (distributions_file == NULL)
    {
        echo(error, "Error: %s", strerror(errno));
        return -1;
    }
    int items;
    fscanf(distributions_file, "%d", &items);
    while (fgetc(distributions_file) != '\n');
    // items = min(items, 40000);
    echo(info, "Total %d distributions", items);
    bit256 *distributions = new bit256[items];
    char line[257];

    for (int i = 0; i < items; i++)
    {
        fscanf(distributions_file, "%s", line);
        bit256 tb(line);
        TileIndex offset = 256 - strlen(line);
        if (strncmp(tb.to_string().c_str() + offset, line, strlen(line) - 1) != 0)
        {
            echo(error, "Error: %s\n       %s", line, tb.to_string().c_str());
        }
        distributions[i] = tb;
    }
    fclose(distributions_file);
    echo(info, "Read distributions finished");

    Timer timer;

    bit256 dense("1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111");
    uint64_t dense_A_bitmap[4] = {0xffffffffffffffffull, 0xffffffffffffffffull, 0xffffffffffffffffull, 0xffffffffffffffffull};

    Tile *dA_tile, hA_tile;
    Tile *dB_tile, *hB_tile = new Tile[min(items, 100000)];
    char *dA_data,  hA_data[2320];
    char *dB_data, *hB_data = new char[2320 * min(items, 100000)];
    MatValue *dC_data;

    TileFormat Afmt = CSR;
    MatIndex dense_A_bitslen, dense_A_valslen;
    dense_A_bitslen = bitmap2codelen(dense, &dense_A_valslen, Afmt);
    hA_tile.fmt = Afmt;
    hA_tile.valslen = 255;
    hA_tile.bitslen = (dense_A_bitslen + 15) / 16 * 16;
    memset(hA_data, 0, sizeof(hA_data));
    memcpy(hA_tile.bitmap, dense_A_bitmap, sizeof(dense_A_bitmap));
    hA_tile.bits_off = 0;
    dense_tile_2_fmt(dense, hA_data, hA_data + 160, Afmt, 1);

    cudaMalloc(&dA_tile, sizeof(Tile));
    cudaMalloc(&dB_tile, sizeof(Tile) * min(items, 100000));
    cudaMalloc(&dB_data, 2320 * min(items, 100000));
    cudaMalloc(&dA_data, 2320);
    cudaMalloc(&dC_data, 256 * sizeof(MatValue));
    cudaMemcpy(dA_data, hA_data, 2320, cudaMemcpyHostToDevice);
    cudaMemcpy(dA_tile, &hA_tile, sizeof(Tile), cudaMemcpyHostToDevice);
    char filename[256];

    TileFormat *best_fmt = new TileFormat[min(items, 100000)];
    double *best_gflops = new double[min(items, 100000)];
    double *best_duration = new double[min(items, 100000)];
    MatIndex *intermidiates = new MatIndex[min(items, 100000)];
    

    for (int i = 0; i < items; i += 100000)
    {
        int end_j = std::min(i + 100000, items);
        
        #pragma omp parallel for
        for (int j = 0; j < end_j - i; ++j)
        {
            best_fmt[j] = COO;
            best_gflops[j] = 0;
            best_duration[j] = 0;
        }
        
        for (TileFormat tfB = COO; tfB <= DNS; tfB++)
        {
            #pragma omp parallel for
            for (int j = 0; j < end_j - i; ++j)
            {
                uint64_t bitmap[4] = {0}, bitmap2[4] = {0};
                memset(&hB_tile[j], 0, sizeof(Tile));
                memset(hB_data + j * 2320, 0, sizeof(hB_data));
                MatIndex B_bitslen, B_valslen;
                B_bitslen = bitmap2codelen(distributions[i + j], &B_valslen, tfB);
                hB_tile[j].valslen = B_valslen - 1;
                B_valslen = (B_valslen + 1) / 2 * 2;
                B_bitslen = (B_bitslen + 15) / 16 * 16;
                hB_tile[j].bits_off = 0;
                hB_tile[j].bitslen = B_bitslen / 16;
                dense_tile_2_fmt(distributions[i + j], hB_data + j * 2320, hB_data + j * 2320 + B_bitslen, tfB, 1);
                hB_tile[j].fmt = tfB;
                bit256_2_uint64_arr(distributions[i + j], bitmap);
                intermidiates[j] = b64_multiply(dense_A_bitmap, bitmap, bitmap2);
            }

            cudaMemcpy(dB_tile, hB_tile, sizeof(Tile) * (end_j - i), cudaMemcpyHostToDevice);
            cudaMemcpy(dB_data, hB_data, 2320 * (end_j - i), cudaMemcpyHostToDevice);

            for (int j = 0; j < end_j - i; ++j)
            {
                echo(start_status, "Format %s: Distribution %d / %d", tile_format_name[tfB], i + j, items);
                MatIndex B_bitslen, B_valslen;
                B_bitslen = bitmap2codelen(distributions[i + j], &B_valslen, tfB);
                B_valslen = (B_valslen + 1) / 2 * 2;
                B_bitslen = (B_bitslen + 15) / 16 * 16;

                SpGEMM_B_CalculateOnly<<<400, 128>>>(dA_tile, dA_data, dB_tile + j, dB_data + j * 2320, dC_data);
                cudaDeviceSynchronize();
                cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess)
                {
                    echo(error, "Error: %s", cudaGetErrorString(e));
                    return -1;
                }

                timer_start(timer);
                for (int _ = 0; _ < repeat; ++_)
                {
                    SpGEMM_B_CalculateOnly<<<400, 128>>>(dA_tile, dA_data, dB_tile + j, dB_data + j * 2320, dC_data);
                }
                cudaDeviceSynchronize();
                timer_end(timer);
                double duration = timer_duration(timer) / repeat;
                double gflops = 400 * 4 * intermidiates[j] * 2 / duration / 1e6;

                if (gflops > best_gflops[j])
                {
                    best_duration[j] = duration;
                    best_gflops[j] = gflops;
                    best_fmt[j] = tfB;
                }
            }
        }

        for (int j = 0; j < end_j - i; ++j)
        {
            sprintf(filename, "gen/spgemm-B/%s.txt", tile_format_name[best_fmt[j]]);
            FILE *output_file = fopen(filename, "a");
            if (output_file == NULL)
            {
                echo(error, "Failed to open file %s!", filename);
                return -1;
            }
            fprintf(output_file, "%s %lf %lf\n", distributions[i + j].to_string().c_str(), best_duration[j], best_gflops[j]);
            fclose(output_file);
        }
    }
    echo(stop_status, "");
    return 0;
}

// int spgemm(int argc, char **argv)
// {
//     const char *tile_format_name[] = {"COO", "CSR", "ELL", "HYB", "DRW", "DCL", "DNS"};
//     const char*batch_file = argv[1];
//     MatIndex repeat = 100;
//     Timer timer;

//     FILE*distribution_fp = fopen(batch_file, "r");
//     if (distribution_fp == NULL) {
//         echo(error, "Failed to open file %s!", batch_file);
//         return -1;
//     }
//     int items;
//     fscanf(distribution_fp, "%d", &items);
//     while(fgetc(distribution_fp) != '\n');
//     items = std::min(items, 100000);
//     echo(info, "Total %d items to be processed!", items);
//     bit256*distributionsA = new bit256[items];
//     bit256*distributionsB = new bit256[items];
//     bit256*distributionsC = new bit256[items];
//     char lineA[257], lineB[257], lineC[257];

//     for (int i = 0; i < items; ++i)
//     {
//         memset(lineA, 0, sizeof(lineA));
//         memset(lineB, 0, sizeof(lineB));
//         memset(lineC, 0, sizeof(lineC));
//         fscanf(distribution_fp, "%256s%256s%256s", lineA, lineB, lineC);
//         distributionsA[i] = bit256(lineA);
//         distributionsB[i] = bit256(lineB);
//         distributionsC[i] = bit256(lineC);
//     }
//     fclose(distribution_fp);
//     echo(success, "Distributions Loaded!");

//     Tile     *dA_tile, hA_tile;
//     char     *dA_data, hA_data[2320];
//     Tile     *dB_tile, hB_tile;
//     char     *dB_data, hB_data[2320];
//     Tile     *dC_tile, hC_tile;
//     char     *dC_data, hC_data[2320];
//     MatValue *dC;

//     cudaMalloc(&dA_tile, sizeof(Tile));
//     cudaMalloc(&dA_data, 2320);
//     cudaMalloc(&dB_tile, sizeof(Tile));
//     cudaMalloc(&dB_data, 2320);
//     cudaMalloc(&dC_tile, sizeof(Tile));
//     cudaMalloc(&dC_data, 2320);
//     cudaMalloc(&dC, sizeof(MatValue) * 25600);
//     char filename[256] = {0};
//     const int use_m = 200;

//     MatIndex total = items, current = 0;
//     for (int i = 0; i < items; ++i) {
//         TileFormat best_tfA = COO, best_tfB = COO, best_tfC = COO;
//         MatValue best_gflops = 0, best_duration = 0;
//         echo(start_status, "Processing %d/%d", ++current, total);
//         memset(hA_tile.bitmap, 0, sizeof(hA_tile.bitmap));
//         memset(hB_tile.bitmap, 0, sizeof(hB_tile.bitmap));
//         memset(hC_tile.bitmap, 0, sizeof(hC_tile.bitmap));
//         bit256_2_uint64_arr(distributionsA[i], hA_tile.bitmap);
//         bit256_2_uint64_arr(distributionsB[i], hB_tile.bitmap);
//         bit256_2_uint64_arr(distributionsC[i], hC_tile.bitmap);
//         MatIndex intermidiate = b64_multiply(hA_tile.bitmap, hB_tile.bitmap, hC_tile.bitmap);

//         for (TileFormat tfA = COO; tfA <= DNS; tfA++){
//             memset(&hA_tile, 0, sizeof(Tile));
//             memset(hA_data, 0, sizeof(hA_data));
//             MatIndex A_bitslen, A_valslen;
//             A_bitslen = bitmap2codelen(distributionsA[i], &A_valslen, tfA);
//             hA_tile.valslen = A_valslen - 1;
//             A_valslen = (A_valslen + 1) / 2 * 2;
//             A_bitslen = (A_bitslen + 15) / 16 * 16;
//             hA_tile.bits_off = 0;
//             hA_tile.bitslen = A_bitslen / 16;
//             dense_tile_2_fmt(distributionsA[i], hA_data, hA_data + A_bitslen, tfA, 1);
//             hA_tile.fmt = tfA;

//             cudaMemcpy(dA_tile, &hA_tile, sizeof(Tile), cudaMemcpyHostToDevice);
//             cudaMemcpy(dA_data, hA_data, A_bitslen + A_valslen * 8, cudaMemcpyHostToDevice);
            
//             for (TileFormat tfB = COO; tfB <= DNS; tfB++)
//             {
//                 memset(&hB_tile, 0, sizeof(Tile));
//                 memset(hB_data, 0, sizeof(hB_data));
//                 MatIndex B_bitslen, B_valslen;
//                 B_bitslen = bitmap2codelen(distributionsB[i], &B_valslen, tfB);
//                 hB_tile.valslen = B_valslen - 1;
//                 B_valslen = (B_valslen + 1) / 2 * 2;
//                 B_bitslen = (B_bitslen + 15) / 16 * 16;
//                 hB_tile.bits_off = 0;
//                 hB_tile.bitslen = B_bitslen / 16;
//                 dense_tile_2_fmt(distributionsB[i], hB_data, hB_data + B_bitslen, tfB, 1);
//                 hB_tile.fmt = tfB;

//                 cudaMemcpy(dB_tile, &hB_tile, sizeof(Tile), cudaMemcpyHostToDevice);
//                 cudaMemcpy(dB_data, hB_data, B_bitslen + B_valslen * 8, cudaMemcpyHostToDevice);

//                 for (TileFormat tfC = COO; tfC <= DNS; ++tfC)
//                 {
//                     memset(hC_data, 0, sizeof(hC_data));
//                     MatIndex bitslen, valslen;
//                     bitslen = bitmap2codelen(distributionsC[i], &valslen, tfC);
//                     hC_tile.valslen = valslen - 1;
//                     bitslen = (bitslen + 15) / 16 * 16;
//                     valslen = (valslen + 1) / 2 * 2;
//                     int bytes = bitslen + valslen * 8;
//                     dense_tile_2_fmt(distributionsC[i], hC_data, hC_data + bitslen, tfC, 0);
//                     hC_tile.bits_off = 0;
//                     hC_tile.bitslen = bitslen / 16;
//                     hC_tile.fmt = tfC;

//                     cudaMemcpy(dC_tile, &hC_tile, sizeof(Tile), cudaMemcpyHostToDevice);
//                     cudaMemcpy(dC_data, hC_data, bytes, cudaMemcpyHostToDevice);

//                     SpGEMM_CalculateOnly<<<use_m, 128>>>(dA_tile, dB_tile, dC_tile, dA_data, dB_data, dC_data, dC);
//                     cudaDeviceSynchronize();
//                     cudaError_t e = cudaGetLastError();
//                     if (e != cudaSuccess) {
//                         echo(error, "Error: %s", cudaGetErrorString(e));
//                         return -1;
//                     }

//                     timer_start(timer);
//                     for (int j = 0; j < repeat; ++j)
//                     {
//                         SpGEMM_CalculateOnly<<<use_m, 128>>>(dA_tile, dB_tile, dC_tile, dA_data, dB_data, dC_data, dC);
//                     }
//                     cudaDeviceSynchronize();
//                     timer_end(timer);
//                     double duration = timer_duration(timer) / repeat;
//                     double gflops = use_m * 4 * intermidiate * 2 / duration / 1e6;

//                     if (gflops > best_gflops)
//                     {
//                         best_duration = duration;
//                         best_gflops = gflops;
//                         best_tfA = tfA;
//                         best_tfB = tfB;
//                         best_tfC = tfC;
//                     }
//                 }
//             }
//         }
//         sprintf(filename, "gen/spgemm/%sx%s-%s.txt", tile_format_name[best_tfA], tile_format_name[best_tfB], tile_format_name[best_tfC]);
//         FILE*fp = fopen(filename, "a");
//         if (fp == NULL) {
//             echo(error, "Failed to open file %s!", filename);
//             return -1;
//         }
//         fprintf(fp, "%s %s %s %lf %lf\n", distributionsA[i].to_string().c_str(), distributionsB[i].to_string().c_str(), distributionsC[i].to_string().c_str(), best_duration, best_gflops);
//         fclose(fp);
//     }
//     echo(stop_status, "");
//     return 0;
// }

int format_data(int argc, char **argv)
{
    FILE *distributions_file = fopen(argv[1], "r");
    TileFormat fmt = atoi(argv[2]);
    if (distributions_file == NULL)
    {
        echo(error, "Error: %s", strerror(errno));
        return -1;
    }
    int items = atoi(argv[3]);
    double old_p = atof(argv[4]);
    double new_p = atof(argv[5]);
    echo(info, "Total %d distributions", items);
    bit256 *distributions = new bit256[items];
    double*times = new double[items];
    double*gflops = new double[items];
    char line[257];

    for (int i = 0; i < items; i++)
    {
        fscanf(distributions_file, "%s%lf%lf", line, times + i, gflops + i);
        bit256 tb(line);
        TileIndex offset = 256 - strlen(line);
        if (strncmp(tb.to_string().c_str() + offset, line, strlen(line) - 1) != 0)
        {
            echo(error, "Error: %s\n       %s", line, tb.to_string().c_str());
        }
        distributions[i] = tb;
    }
    fclose(distributions_file);
    echo(info, "Read distributions finished");

    distributions_file = fopen(argv[1], "w");
    // fprintf(distributions_file, "%d\n", items);
    for (int i = 0; i < items; ++i)
    {
        MatIndex bitslen, valslen;
        bitslen = bitmap2codelen(distributions[i], &valslen, fmt);
        valslen = (valslen + 1) / 2 * 2;
        bitslen = (bitslen + 15) / 16 * 16;
        MatIndex bytes = bitslen + valslen * 8;
        if (old_p != 0) times[i] -= (3200 * 2 * bytes) / old_p / 1e9;
        double time = times[i] + (3200 * 2 * bytes) / new_p / 1e9;
        double gflop = 3200 * 4 * distributions[i].count() / time / 1e6;
        fprintf(distributions_file, "%s %lf %lf\n", distributions[i].to_string().c_str(), time, gflop);
    }
    fclose(distributions_file);
    echo(info, "Write distributions finished");
    delete[] distributions;
    delete[] times;
    delete[] gflops;
    return 0;
}

int main(int argc, char **argv)
{
    std::string op = argv[argc - 1];
    int device_id = atoi(argv[argc - 2]);
    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    echo(info, "Using device %d: %s", device_id, prop.name);
    if (op == "spmv")
        return spmv(argc - 1, argv);
    else if (op == "spmm")
        return spmm(argc - 1, argv);
    else if (op == "spgemmB")
        return spgemmB(argc - 1, argv);
    else if (op == "format-data")
        return format_data(argc - 1, argv);
    else
    {
        echo(error, "Error: Unknown operation %s", op.c_str());
    }
    return 0;
}
