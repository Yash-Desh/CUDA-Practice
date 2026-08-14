// Author : Yash Deshpande
// Date   : 21-05-2026
// Tutor  : Izzat El Hajj, AUB
//
// Goal : Compare a basic CUDA GEMM kernel (same as 4_matmul.cu) against
//        cuBLAS SGEMM on the SAME input matrices, verify correctness,
//        and print timing + GFLOPS for each.
//
// Compile :  nvcc 29_matmul_cublas.cu -lcublas -o 29_matmul_cublas
// Run     :  ./29_matmul_cublas            (default N = 1024)
//            ./29_matmul_cublas 2048       (custom N)
//            ./29_matmul_cublas 4096       (CPU verification is auto-skipped for large N)

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cuda.h>
#include <cublas_v2.h>          // cuBLAS header
#include <random>               // generate random numbers
#include <chrono>               // std::chrono for CPU timing
#include <ratio>                // std::ratio for metric unit conversions

using std::chrono::duration;
using std::chrono::high_resolution_clock;

// ---------------------------------------------------------------------------
// Lightweight error-checking macros.
// CUDA and cuBLAS calls return status codes that are easy to ignore but
// painful to debug when something silently fails. These macros print the
// failing call's file & line so you catch problems immediately.
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _err = (call);                                             \
        if (_err != cudaSuccess) {                                             \
            std::cerr << "CUDA error " << cudaGetErrorString(_err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t _s = (call);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            std::cerr << "cuBLAS error " << int(_s)                            \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)


// ---------------------------------------------------------------------------
// CPU reference implementation (identical to 4_matmul.cu).
// Used only for correctness verification at small N.
// ---------------------------------------------------------------------------
void matmul_cpu(float *A, float *B, float *C, unsigned int N)
{
    for (int row = 0; row < (int)N; row++)
    {
        for (int col = 0; col < (int)N; col++)
        {
            float sum = 0.0f;
            for (int i = 0; i < (int)N; i++)
            {
                sum += A[row * N + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}


// ---------------------------------------------------------------------------
// Naive GPU kernel (identical to 4_matmul.cu's matmul_kernel).
// One thread computes one output element of C by reading a full row of A
// and a full column of B from global memory. No shared memory, no tiling.
// ---------------------------------------------------------------------------
__global__ void matmul_naive_kernel(float *A, float *B, float *C, unsigned int N)
{
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < (int)N; i++)
        {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}


// ---------------------------------------------------------------------------
// Runs the naive kernel on already-allocated device pointers.
// Returns the kernel time (excluding H2D/D2H copies) in milliseconds.
// We separate allocation/copy from the timed region so we are comparing
// PURE COMPUTE TIME between naive and cuBLAS, which is the fair comparison.
// ---------------------------------------------------------------------------
float run_naive_gpu(float *A_d, float *B_d, float *C_d, unsigned int N)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Warm-up launch. The very first kernel launch on a fresh context pays
    // a one-time JIT / context-creation cost that has nothing to do with
    // the kernel itself. Doing one untimed launch first makes the timed
    // run reflect actual steady-state performance.
    matmul_naive_kernel<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run.
    cudaEventRecord(start);
    matmul_naive_kernel<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}


// ---------------------------------------------------------------------------
// Runs cuBLAS SGEMM on already-allocated device pointers.
// Returns the kernel time (excluding H2D/D2H copies) in milliseconds.
//
// Important: we are storing A and B in ROW-MAJOR layout (the normal C/C++
// way) but cuBLAS interprets memory as COLUMN-MAJOR (Fortran-style).
//
// Mathematical identity used to handle this without an actual transpose:
//     (A * B)^T = B^T * A^T
//
// Our row-major buffer for X, when re-interpreted as column-major with
// leading dimension N, IS literally X^T. So if we pass B and A (in that
// order) to a column-major SGEMM with NoTrans/NoTrans, cuBLAS computes
//     C_colmaj = B_colmaj * A_colmaj
//              = (B^T) * (A^T)       <-- because of the row/col flip
//              = (A * B)^T
// and that result, written column-major into C_d, IS the row-major (A*B)
// we want. So the only code change is: swap A and B in the call.
// ---------------------------------------------------------------------------
float run_cublas_gpu(cublasHandle_t handle,
                     float *A_d, float *B_d, float *C_d,
                     unsigned int N)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // alpha and beta are passed BY POINTER in cuBLAS, not by value.
    // (cuBLAS supports pointing them at device memory too, but host is
    // fine for our case.)
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Warm-up call. cuBLAS' first SGEMM on a fresh handle does internal
    // algorithm selection and may load kernels from disk, so the first
    // call can be misleadingly slow.
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             B_d, N,    // B first  (column-major trick)
                             A_d, N,    // A second
                             &beta,
                             C_d, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed call.
    cudaEventRecord(start);
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             B_d, N,
                             A_d, N,
                             &beta,
                             C_d, N));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}


// ---------------------------------------------------------------------------
// Verify two NxN matrices agree within a small relative tolerance.
// Floating-point matmul is non-associative, so cuBLAS and our naive kernel
// will produce values that are *very* close but not bit-identical. A few
// ULPs of difference is normal; anything much bigger means a real bug.
// ---------------------------------------------------------------------------
bool verify(float *ref, float *test, unsigned int N, float tol = 1e-2f)
{
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    for (unsigned int i = 0; i < N * N; i++)
    {
        double diff = std::fabs(double(ref[i]) - double(test[i]));
        double denom = std::max(std::fabs(double(ref[i])), 1e-6);
        max_abs_err = std::max(max_abs_err, diff);
        max_rel_err = std::max(max_rel_err, diff / denom);
    }
    std::cout << "    max abs error = " << max_abs_err
              << ",  max rel error = " << max_rel_err << "\n";
    return max_rel_err < tol;
}


// ---------------------------------------------------------------------------
// Compute GFLOPS for an NxN * NxN matmul that took `ms` milliseconds.
// Each output element does N multiplies and N adds = 2N flops, and there
// are N*N output elements, so total = 2 * N^3 floating-point operations.
// ---------------------------------------------------------------------------
double gflops(unsigned int N, float ms)
{
    double flops = 2.0 * double(N) * double(N) * double(N);
    return flops / (double(ms) * 1.0e6);   // ms * 1e6 = ns; flops / ns = GFLOPS
}


int main(int argc, char **argv)
{
    CUDA_CHECK(cudaDeviceSynchronize());

    // ######################### Allocate memory & initialize data #########################

    unsigned int N = (argc > 1) ? atoi(argv[1]) : 1024;
    std::cout << "Matrix size: " << N << " x " << N << "\n\n";

    // Print the GPU name so the benchmark numbers have context.
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name
              << "  (SM " << prop.major << "." << prop.minor << ")\n\n";

    float *A      = (float*)malloc(N * N * sizeof(float));
    float *B      = (float*)malloc(N * N * sizeof(float));
    float *C_cpu  = (float*)malloc(N * N * sizeof(float));
    float *C_naive= (float*)malloc(N * N * sizeof(float));
    float *C_cubl = (float*)malloc(N * N * sizeof(float));

    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (unsigned int i = 0; i < N * N; i++)
    {
        A[i] = dist(generator);
        B[i] = dist(generator);
    }

    // Device buffers shared by both GPU implementations.
    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C_d, N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice));

    // ######################### CPU reference (only for small N) #########################

    // For N=1024 the CPU triple loop takes a few seconds, which is fine.
    // For N=4096 it would take minutes, so we skip it automatically.
    bool run_cpu = (N <= 1024);
    double cpu_ms = 0.0;
    if (run_cpu)
    {
        auto t0 = high_resolution_clock::now();
        matmul_cpu(A, B, C_cpu, N);
        auto t1 = high_resolution_clock::now();
        cpu_ms = duration<double, std::milli>(t1 - t0).count();
        std::cout << "[CPU naive triple loop]\n"
                  << "    time   = " << cpu_ms << " ms\n"
                  << "    GFLOPS = " << gflops(N, float(cpu_ms)) << "\n\n";
    }
    else
    {
        std::cout << "[CPU naive triple loop]  SKIPPED (N too large)\n\n";
    }

    // ######################### Naive CUDA kernel #########################

    float naive_ms = run_naive_gpu(A_d, B_d, C_d, N);
    CUDA_CHECK(cudaMemcpy(C_naive, C_d, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[GPU naive kernel  (4_matmul.cu style)]\n"
              << "    time   = " << naive_ms << " ms\n"
              << "    GFLOPS = " << gflops(N, naive_ms) << "\n";
    if (run_cpu) verify(C_cpu, C_naive, N);
    std::cout << "\n";

    // ######################### cuBLAS SGEMM #########################

    // Create one cuBLAS handle. In a real app you'd hold onto this for the
    // lifetime of the program and reuse it for many calls.
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float cublas_ms = run_cublas_gpu(handle, A_d, B_d, C_d, N);
    CUDA_CHECK(cudaMemcpy(C_cubl, C_d, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "[GPU cuBLAS SGEMM]\n"
              << "    time   = " << cublas_ms << " ms\n"
              << "    GFLOPS = " << gflops(N, cublas_ms) << "\n";
    // Verify cuBLAS against naive (which we already verified against CPU
    // if run_cpu was true; otherwise the naive kernel is itself the ref).
    verify(C_naive, C_cubl, N);
    std::cout << "\n";

    CUBLAS_CHECK(cublasDestroy(handle));

    // ######################### Summary #########################

    std::cout << "=========== Summary ===========\n";
    std::cout << std::left << std::setw(28) << "Implementation"
              << std::right << std::setw(12) << "time (ms)"
              << std::setw(14) << "GFLOPS"
              << std::setw(14) << "speedup vs naive GPU\n";
    if (run_cpu)
    {
        std::cout << std::left << std::setw(28) << "CPU triple loop"
                  << std::right << std::setw(12) << cpu_ms
                  << std::setw(14) << gflops(N, float(cpu_ms))
                  << std::setw(14) << (naive_ms / float(cpu_ms)) << "x\n";
    }
    std::cout << std::left << std::setw(28) << "GPU naive kernel"
              << std::right << std::setw(12) << naive_ms
              << std::setw(14) << gflops(N, naive_ms)
              << std::setw(14) << 1.0f << "x\n";
    std::cout << std::left << std::setw(28) << "GPU cuBLAS SGEMM"
              << std::right << std::setw(12) << cublas_ms
              << std::setw(14) << gflops(N, cublas_ms)
              << std::setw(14) << (naive_ms / cublas_ms) << "x\n";

    // ######################### Cleanup #########################

    free(A); free(B); free(C_cpu); free(C_naive); free(C_cubl);
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
    return 0;
}
