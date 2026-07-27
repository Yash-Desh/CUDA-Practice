// Author : Yash Deshpande
// Date : 28-12-2024
// Tutor : Izzat El Hajj, AUB

#include <iostream>
#include <cuda.h>
#include <random>       // generate random numbers
#include <chrono>       // std::chrono namespace provides timer functions in C++, for CPU timing
#include <ratio>        // std::ratio provides easy conversions between metric units, for CPU timing

using std::chrono::duration;
using std::chrono::high_resolution_clock;

// print all the elements of given array
void printArr(float *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

// adds 2 float values
__device__ __host__ float f_add (float a, float b)
{
    return a + b;
}

// serial vector addition on CPU
void vecadd_cpu(float *x, float *y, float *z, int N)
{
    for (int i = 0; i < N; i++)
    {
        // z[i] = x[i] + y[i];
        z[i] = f_add(x[i], y[i]);
    }
}

// CUDA kernel to add 2 vectors
__global__ void vecadd_kernel(float *x, float *y, float *z, int N)
{
    // calculate global index
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        // z[i] = x[i] + y[i];
        z[i] = f_add(x[i], y[i]);
    }
}

// parallel vector addition on GPU
void vecadd_gpu(float *x, float *y, float *z, int N)
{
    // for timing purpose
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

    // Allocate GPU Memory
    float *x_d, *y_d, *z_d;
    cudaMalloc((void **)&x_d, N * sizeof(float));
    cudaMalloc((void **)&y_d, N * sizeof(float));
    cudaMalloc((void **)&z_d, N * sizeof(float));

    // Copy to the GPU
    cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Run the GPU Code
    int numThreadsPerBlock = 512;
    int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    
    //////////////////////////////////////
    cudaEventRecord(start);
    vecadd_kernel<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, z_d, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //////////////////////////////////////

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Total GPU Kernel time: " << ms << "ms\n";
    

    // Copy from GPU
    cudaMemcpy(z, z_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate GPU Memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}

int main(int argc, char **argv)
{
    cudaDeviceSynchronize();

    // ######################### Allocate memory & initialize data #########################
    unsigned int N = (argc > 1) ? atoi(argv[1]) : (1 << 25);
    float *x = (float *)malloc(N * sizeof(float));
    float *y = (float *)malloc(N * sizeof(float));
    float *z = (float *)malloc(N * sizeof(float));

    // Generate Random Numbers b/w [10, 20]
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_real_distribution<float> dist(10., 20.);

    for (int i = 0; i < N; i++)
    {
        x[i] = dist(generator);
        y[i] = dist(generator);
    }
    // printArr(x, N);
    // printArr(y, N);

    // ######################### vector addition on CPU #########################

    high_resolution_clock::time_point start_cpu;
    high_resolution_clock::time_point end_cpu;
    duration<double, std::milli> duration_sec;

    //////////////////////////////////////
    start_cpu = high_resolution_clock::now();       // Get the starting timestamp
    vecadd_cpu(x, y, z, N);
    end_cpu = high_resolution_clock::now();         // Get the ending timestamp
    //////////////////////////////////////

    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end_cpu - start_cpu);
    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    std::cout << "Total CPU time: " << duration_sec.count() << "ms\n";
    std::cout << z[N-1]<<"\n";

    // ######################### vector addition on GPU #########################
    cudaEvent_t start_gpu;
    cudaEvent_t stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    //////////////////////////////////////
    cudaEventRecord(start_gpu);
    vecadd_gpu(x, y, z, N);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    //////////////////////////////////////

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start_gpu, stop_gpu);
    std::cout << "Total GPU time: " << ms << "ms\n";
    std::cout << z[N-1]<<"\n";

    // free memory
    free(x);
    free(y);
    free(z);

    return 0;
}

/*

1. Why is numThreadsPerBlock = 512 ?










*/ 