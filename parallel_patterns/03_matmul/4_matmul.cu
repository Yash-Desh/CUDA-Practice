// Author : Yash Deshpande
// Date : 29-12-2024
// Tutor : Izzat El Hajj, AUB

#include <iostream>
#include <cuda.h>
#include <random> // generate random numbers
#include <chrono> // std::chrono namespace provides timer functions in C++, for CPU timing
#include <ratio>  // std::ratio provides easy conversions between metric units, for CPU timing

using std::chrono::duration;
using std::chrono::high_resolution_clock;

// prints all the elements of given array
void printArr(float *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}


// performs matrix multiplication on the CPU
void matmul_cpu(float *A, float *B, float *C, unsigned int N)
{
    for(int row=0; row<N; row++)
    {
        for(int col=0; col<N; col++)
        {
            float sum = 0.0f;
            for(int i=0; i<N; i++)
            {
                sum += A[row * N + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

__global__ void matmul_kernel(float *A, float *B, float *C, unsigned int N)
{
    // NOTE : 1 thread per output pixel
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.0f;
        for (int i = 0; i < N; i++)
        {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matmul_gpu(float *A, float *B, float *C, unsigned int N)
{
    // for timing purpose
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory on GPU
    float *A_d, *B_d, *C_d;
    cudaMalloc((void **)&A_d, N * N * sizeof(float));
    cudaMalloc((void **)&B_d, N * N * sizeof(float));
    cudaMalloc((void **)&C_d, N * N * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Run Kernel
    dim3 numThreadsPerBlock(32, 32);
    dim3 numBlocks((N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (N + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

    //////////////////////////////////////
    cudaEventRecord(start);
    matmul_kernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //////////////////////////////////////

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Total GPU Kernel time: " << ms << "ms\n";

    // Copy data from GPU
    cudaMemcpy(C, C_d, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate memory on GPU
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(int argc, char **argv)
{
    cudaDeviceSynchronize();

    // ######################### Allocate memory & initialize data #########################

    // Declare the host arrays
    unsigned int N = (argc > 1) ? atoi(argv[1]) : (1024);
    float *A = (float*)malloc(N * N * sizeof(float));
    float *B = (float*)malloc(N * N * sizeof(float));
    float *C = (float*)malloc(N * N * sizeof(float));

    // Generate Random Numbers
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-10.0, 10.0);

    // Initialize the arrays
    for (int i = 0; i < N*N; i++)
    {
        A[i] = dist(generator);
        B[i] = dist(generator);
    }
    // printArr(A, N * N);
    // printArr(B, N * N);

    // ######################### vector addition on CPU #########################

    high_resolution_clock::time_point start_cpu;
    high_resolution_clock::time_point end_cpu;
    duration<double, std::milli> duration_sec;

    //////////////////////////////////////
    start_cpu = high_resolution_clock::now(); // Get the starting timestamp
    matmul_cpu(A, B, C, N);
    end_cpu = high_resolution_clock::now(); // Get the ending timestamp
    //////////////////////////////////////

    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end_cpu - start_cpu);
    std::cout << "Total CPU time: " << duration_sec.count() << "ms\n";

    // ######################### vector addition on GPU #########################
    cudaEvent_t start_gpu;
    cudaEvent_t stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    //////////////////////////////////////
    cudaEventRecord(start_gpu);
    matmul_gpu(A, B, C, N);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    //////////////////////////////////////

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start_gpu, stop_gpu);
    std::cout << "Total GPU time: " << ms << "ms\n";

    // free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}