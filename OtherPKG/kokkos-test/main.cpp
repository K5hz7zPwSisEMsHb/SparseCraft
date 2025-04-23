#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_spgemm.hpp>
#include <KokkosKernels_Handle.hpp>
#include <chrono>
#include <mmio_highlevel.h>

// 性能测试函数模板
template <typename ExecSpace>
double run_spmv_test(const KokkosSparse::CrsMatrix<double, int, ExecSpace>& A,
                   int trials, MatIndex nnz) {
    using VectorType = Kokkos::View<double*, typename ExecSpace::memory_space>;
    
    const int N = A.numRows();
    VectorType x("x", N);
    VectorType y("y", N);
    Kokkos::deep_copy(x, 1.0);
    Kokkos::deep_copy(y, 0.0);

    // Warmup
    for (int i = 0; i < 100; ++i) {
        KokkosSparse::spmv("N", 1.0, A, x, 0.0, y);
    }
    Kokkos::fence();

    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; ++i) {
        KokkosSparse::spmv("N", 1.0, A, x, 0.0, y);
    }
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate performance
    double time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double avg_time = time / trials;
    double gflops = 2.0 * nnz / avg_time * 1e-6;

    printf("\n[SpMV Performance]\n");
    printf("Average time per SpMV: %.4f ms\n", avg_time);
    printf("Throughput: %.2f GFlops\n", gflops);

    return gflops;
}

template <typename ExecSpace>
double run_spmm_test(const KokkosSparse::CrsMatrix<double, int, ExecSpace>& A,
                   int trials, MatIndex nnz, int N) {
    using DenseMatrixType = Kokkos::View<double**, typename ExecSpace::memory_space>;
    
    const int nrows = A.numRows();
    DenseMatrixType X("X", nrows, N);
    DenseMatrixType Y("Y", nrows, N);
    Kokkos::deep_copy(X, 1.0);
    Kokkos::deep_copy(Y, 0.0);

    // Warmup
    for (int i = 0; i < 100; ++i) {
        for (int col = 0; col < N; ++col) {
            auto x = Kokkos::subview(X, Kokkos::ALL, col);
            auto y = Kokkos::subview(Y, Kokkos::ALL, col);
            KokkosSparse::spmv("N", 1.0, A, x, 0.0, y);
        }
    }
    Kokkos::fence();

    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < trials; ++i) {
        for (int col = 0; col < N; ++col) {
            auto x = Kokkos::subview(X, Kokkos::ALL, col);
            auto y = Kokkos::subview(Y, Kokkos::ALL, col);
            KokkosSparse::spmv("N", 1.0, A, x, 0.0, y);
        }
    }
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate performance (2 * nnz * N per SpMM)
    double time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double avg_time = time / trials;
    double gflops = 2.0 * nnz * N / avg_time * 1e-6;

    printf("\n[SpMM Performance (N=%d)]\n", N);
    printf("Average time per SpMM: %.4f ms\n", avg_time);
    printf("Throughput: %.2f GFlops\n", gflops);

    return gflops;
}

template <typename ExecSpace>
double run_spgemm_test(const KokkosSparse::CrsMatrix<double, int, ExecSpace>& A, MatIndex nnz, int intermidiate) {
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
        int, int, double, ExecSpace, typename ExecSpace::memory_space, typename ExecSpace::memory_space>;
    
    KernelHandle kh;
    kh.set_team_work_size(16);
    kh.set_dynamic_scheduling(true);

    std::string myalg("SPGEMM_KK_MEMORY");
    KokkosSparse::SPGEMMAlgorithm spgemm_algorithm =
        KokkosSparse::StringToSPGEMMAlgorithm(myalg);
    kh.create_spgemm_handle(spgemm_algorithm);

    // Warmup
    KokkosSparse::CrsMatrix<double, int, ExecSpace> C;
    auto start = std::chrono::high_resolution_clock::now();
    KokkosSparse::spgemm_symbolic(kh, A, false, A, false, C);
    Kokkos::fence();
    auto middle = std::chrono::high_resolution_clock::now();
    KokkosSparse::spgemm_numeric(kh, A, false, A, false, C);
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();
    double warp_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double symbolic_time = std::chrono::duration_cast<std::chrono::milliseconds>(middle - start).count();
    double numeric_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - middle).count();
    numeric_time *= 1e-6;
    printf("Warmup time: %.4f ms, Symbolic time: %.4f ms, Numeric time: %.4f ms\n", warp_time, symbolic_time, numeric_time);
    // print first 10 values
    auto C_values = C.values;
    auto h_C_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C_values);
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %.2f\n", i, h_C_values(i));
    }

    // Calculate performance using actual C matrix nnz
    printf("intermidiate = %d\n", intermidiate);
    double gflops = 2 * intermidiate / (numeric_time + symbolic_time) * 1e-6;

    printf("\n[SpGEMM Performance]\n");
    printf("Numeric Throughput: %.2lf GFlops (Time: %.2lf + %.2lf ms)\n", gflops, symbolic_time, numeric_time);
    
    kh.destroy_spgemm_handle();
    return gflops;
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("Usage: %s <matrix_file> <test_type>\n", argv[0]);
        printf("Available test types:\n");
        printf("  --spmv    Run SpMV test\n");
        printf("  --spmm    Run SpMM test (N=8)\n");
        printf("  --spgemm  Run SpGEMM test\n");
        return 1;
    }

    Kokkos::initialize(argc, argv);
    // 从文件加载矩阵数据
    int m, n, isSymmetric;
    MatIndex nnz;
    MatIndex *csrRowPtr = nullptr;
    MatIndex *csrColIdx = nullptr;
    MatValue *csrVal = nullptr;
    std::string test_type(argv[argc - 1]);

    mmio_allinone(argv[1], &m, &n, &nnz, &isSymmetric, &csrRowPtr, &csrColIdx, &csrVal);
    if (test_type == "--spgemm" && m != n)
    {
        printf("SpGEMM only supports square matrices\n");
        free(csrRowPtr);
        free(csrColIdx);
        free(csrVal);
        return 1;
    }
    {
        using ExecSpace = Kokkos::Cuda;
        using MemSpace = ExecSpace::memory_space;
        using MatrixType = KokkosSparse::CrsMatrix<double, int, ExecSpace>;
        
        const int N = m;
        int trials = 1000;

        auto start = std::chrono::high_resolution_clock::now();
        // 创建矩阵视图并拷贝到设备
        auto h_row_map = Kokkos::View<MatIndex*, Kokkos::HostSpace>(csrRowPtr, N + 1);
        auto h_entries = Kokkos::View<MatIndex*, Kokkos::HostSpace>(csrColIdx, nnz);
        auto h_values = Kokkos::View<MatValue*, Kokkos::HostSpace>(csrVal, nnz);
        
        auto row_map = Kokkos::create_mirror_view_and_copy(MemSpace(), h_row_map);
        auto entries = Kokkos::create_mirror_view_and_copy(MemSpace(), h_entries);
        auto values = Kokkos::create_mirror_view_and_copy(MemSpace(), h_values);
	
        MatrixType A("A", N, N, nnz, values, row_map, entries);
	    auto end = std::chrono::high_resolution_clock::now();
        double init_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // 输出矩阵基本信息
        printf("\nMatrix Information: %lf\n", init_time);
        printf("Dimensions: %d x %d\n", N, N);
        printf("Non-zeros: %d\n", nnz);
        printf("Symmetric: %s\n", isSymmetric ? "Yes" : "No");

        // 根据参数执行测试
        double gflops;
        
        if (test_type == "--spmv") {
            gflops = run_spmv_test(A, trials, nnz);
        } else if (test_type == "--spmm") {
            trials = 100;
            gflops = run_spmm_test(A, trials, nnz, atoi(argv[2]));
        } else if (test_type == "--spgemm") {
            uint64_t intermidiate = 0;
            #pragma omp parallel for reduction(+:intermidiate)
            for (int i = 0; i < m; ++i)
            {
                uint64_t sum = 0;
                for (int aj = csrRowPtr[i]; aj < csrRowPtr[i + 1]; ++aj)
                {
                    int col = csrColIdx[aj];
                    sum += csrRowPtr[col + 1] - csrRowPtr[col];
                }
                intermidiate += sum;
            }
            gflops = run_spgemm_test(A, nnz, intermidiate); 
        } else {
            printf("Invalid test type specified\n");
            return 1;
        }
        printf("%.3lf,%.3lf\n", init_time, gflops);
    }
    // 释放原始内存
    free(csrRowPtr);
    free(csrColIdx);
    free(csrVal);
    Kokkos::finalize();
    return 0;
}
