// Author : Yash Deshpande
// Date   : 18-04-2026
// Tutor  : Izzat El Hajj, AUB
// Link   : https://youtu.be/1G4jfLcnI2w?si=O-hFEcAMJUHCIWFY

#include <iostream>
#include <cuda.h>
#include <random>               // generate random numbers
#include <chrono>               // std::chrono namespace provides timer functions in C++, for CPU timing
#include <ratio>                // std::ratio provides easy conversions between metric units, for CPU timing

using std::chrono::duration;
using std::chrono::high_resolution_clock;

#define BLOCK_DIM 1024

// Inclusive scan (prefix sum) on the CPU.
void scan_cpu(float* input, float* output, unsigned int N) {
    output[0] = input[0];
    for(unsigned int i = 1; i < N; i++) {
        output[i] = output[i-1] + input[i];
    }
}

// Inclusive scan (prefix sum) on the GPU. 
__global__ void scan_kernel(float* input, float* output, float* partialSums, unsigned int N) {
    unsigned int segment = 2 * blockIdx.x * blockDim.x;

    // Load the segment into shared memory.
    __shared__ float buffer_s [2 * BLOCK_DIM];

    // Load 2 elements per thread in coalesced manner.
    buffer_s[threadIdx.x] = input[segment + threadIdx.x];
    buffer_s[threadIdx.x + BLOCK_DIM] = input[segment + threadIdx.x + BLOCK_DIM];
    __syncthreads();

    // Reduction Step
    for(unsigned int stride = 1; stride <= BLOCK_DIM; stride *= 2) {
        // Right source element from the Brent-Kung Algorithm. 
        unsigned int i = (threadIdx.x + 1) * 2 * stride - 1;
        
        // Boundary Check: Access elements within the segment
        if (i < 2 * BLOCK_DIM) {
            buffer_s[i] += buffer_s[i - stride];
        }
        __syncthreads();
    }

    // Post-Reduction Step
    for(unsigned int stride = BLOCK_DIM/2; stride >= 1; stride /= 2) {
        // Right source element from the Brent-Kung Algorithm. 
        unsigned int i = (threadIdx.x + 1) * 2 * stride - 1;

        // Boundary Check: Access elements within the segment
        if (i + stride < 2 * BLOCK_DIM) {
            buffer_s[i + stride] += buffer_s[i];
        }
        __syncthreads();
    }

    // 1st thread of the block copies final result into the partial sums array. 
    if(threadIdx.x == 0) {
        partialSums[blockIdx.x] = buffer_s[BLOCK_DIM - 1];
    }

    // Copy the shared memory buffer to output array in global memory
    // where the overall result should be stored. 
    output[segment + threadIdx.x] = buffer_s[threadIdx.x];
    output[segment + threadIdx.x + BLOCK_DIM] = buffer_s[threadIdx.x + BLOCK_DIM];
}

__global__ void add_kernel(float* output, float* partialSums, unsigned int N) {
    unsigned int segment = 2 * blockIdx.x * blockDim.x;
    if(blockIdx.x > 0) {
        output[segment + threadIdx.x] += partialSums[blockIdx.x - 1];
        output[segment + threadIdx.x + BLOCK_DIM] += partialSums[blockIdx.x - 1];
    }
}

// CPU function that executes the segmented scan algorithm & schedules the correct work on the GPU. 
void scan_gpu_d(float* input_d, float* output_d, unsigned int N) {
    // for timing purpose
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Configurations
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2 * numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;

    // Allocate Partial Sums
    float* partialSums_d;
    cudaMalloc((void**) &partialSums_d, numBlocks * sizeof(float));
    cudaDeviceSynchronize();

    // Call Kernel
    //////////////////////////////////////
    cudaEventRecord(start);
    scan_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, partialSums_d, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //////////////////////////////////////

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Total GPU Kernel time: " << ms << "ms\n";


    // Scan Partial Sums then add. 
    if(numBlocks > 1) {
        
        // Scan partial sums
        scan_gpu_d(partialSums_d, partialSums_d, numBlocks);

        // Add scanned sums
        add_kernel <<<numBlocks, numThreadsPerBlock>>> (output_d, partialSums_d, N);
    }

    // Free Memory.

    cudaFree(partialSums_d);
    cudaDeviceSynchronize();
    
}

void scan_gpu(float* input, float* output, int N) {
    // Allocate GPU memory
    float *input_d;
    float* output_d;
    cudaMalloc((void **)&input_d, N * sizeof(float));
    cudaMalloc((void**)&output_d, N * sizeof(float));

    // Copy Data to GPU
    cudaMemcpy(input_d, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Call the recursive scan function
    scan_gpu_d(input_d, output_d, N);
    cudaDeviceSynchronize();

    // Copy Data from GPU
    cudaMemcpy(output, output_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Deallocate Memory
    cudaFree(input_d);
    cudaFree(output_d);
}

int main(int argc, char **argv)
{
    cudaDeviceSynchronize();

    // ######################### Allocate memory & initialize data #########################

    // Declare the host arrays
    unsigned int N = (argc > 1) ? atoi(argv[1]) : (1048576);
    float *input = (float *)malloc(N * sizeof(float));
    float* output = (float*)malloc(N * sizeof(float));

    // Generate Random Numbers
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    // Initialize the arrays
    for (int i = 0; i < N; i++) {
        input[i] = 1;
    }
    

    // ######################### Scan on CPU #########################

    high_resolution_clock::time_point start_cpu;
    high_resolution_clock::time_point end_cpu;
    duration<double, std::milli> duration_sec;

    //////////////////////////////////////
    start_cpu = high_resolution_clock::now(); // Get the starting timestamp
    scan_cpu(input, output, N);
    end_cpu = high_resolution_clock::now(); // Get the ending timestamp
    //////////////////////////////////////

    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end_cpu - start_cpu);
    std::cout << "Total CPU time: " << duration_sec.count() << "ms\n";
    

    // ######################### Scan on GPU #########################
    cudaEvent_t start_gpu;
    cudaEvent_t stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    //////////////////////////////////////
    cudaEventRecord(start_gpu);
    scan_gpu(input, output, N);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    //////////////////////////////////////

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start_gpu, stop_gpu);
    std::cout << "Total GPU time: " << ms << "ms\n";
    
    // free host memory
    free(input);
    free(output);

    return 0;
}