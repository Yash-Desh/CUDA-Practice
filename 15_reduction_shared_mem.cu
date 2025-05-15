// Author : Yash Deshpande
// Date : 07-1-2024
// Tutor : Izzat El Hajj, AUB

#include <iostream>
#include <cuda.h>
#include <random>               // generate random numbers
#include <chrono>               // std::chrono namespace provides timer functions in C++, for CPU timing
#include <ratio>                // std::ratio provides easy conversions between metric units, for CPU timing

using std::chrono::duration;
using std::chrono::high_resolution_clock;

#define BLOCK_DIM 1024

// prints all the elements of given array
void printArr(float *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}


void reduce_cpu(float *input, unsigned int N)
{   
    float sum = 0.0f;
    for(int i=0; i<N; i++)
    {
        sum += input[i];
    }
    std::cout<<"CPU sum = "<<sum<<std::endl;
}

__global__ void reduce_kernel(float *input, float *partial_sums, unsigned int N)
{
    // select data segment to determine a single partial sum
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    // within data segment, select the first half elements
    unsigned int i = segment + threadIdx.x;

    // shared memory
    __shared__ float input_s[BLOCK_DIM];

    // load the shared memory with 1st iteration of sums
    input_s[threadIdx.x] = input[i] + input[i + BLOCK_DIM];
    __syncthreads();

    // loop over each stride value
    // stride is the distance between elements at each iteration of the loop
    for(unsigned int stride = BLOCK_DIM/2; stride >= 1; stride/=2)
    {
        // Only threads at stride distance apart should perform the next addition
        if(threadIdx.x < stride)
        {
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
        }
        __syncthreads();
    }
    // partial sum of each thread-block is stored at the 0th index
    // store the partial sums
    if(threadIdx.x == 0)
    {
        partial_sums[blockIdx.x] = input_s[threadIdx.x];
    }
}

void reduce_gpu(float *input, unsigned int N)
{
    // for timing purpose
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU memory
    float *input_d;
    cudaMalloc((void **)&input_d, N * sizeof(float));
    
    // Copy Data to GPU
    cudaMemcpy(input_d, input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate Partial Sums
    unsigned int numThreadsPerBlock = BLOCK_DIM;
    unsigned int numElementsPerBlock = 2 * numThreadsPerBlock;
    unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;
    float *partial_sums = (float *)malloc(numBlocks * sizeof(float));
    float *partial_sums_d;
    cudaMalloc((void **)&partial_sums_d, numBlocks * sizeof(float));

    // Call Kernel
    
    //////////////////////////////////////
    cudaEventRecord(start);
    reduce_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, partial_sums_d, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //////////////////////////////////////

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Total GPU Kernel time: " << ms << "ms\n";

    // Copy Data from GPU
    cudaMemcpy(partial_sums, partial_sums_d, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Reduce Partial Sums on the CPU
    float sum = 0.0f;
    for(int i=0; i<numBlocks; i++)
    {
        sum += partial_sums[i];
    }
    std::cout<<"GPU Sum = "<<sum<<std::endl;

    // Deallocate Memory
    cudaFree(input_d);
    cudaFree(partial_sums_d);
    free(partial_sums);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main(int argc, char **argv)
{
    cudaDeviceSynchronize();

    // ######################### Allocate memory & initialize data #########################

    // Declare the host arrays
    unsigned int N = (argc > 1) ? atoi(argv[1]) : (1048576);
    float *input = (float *)malloc(N * sizeof(float));

    // Generate Random Numbers
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    // Initialize the arrays
    for (int i = 0; i < N; i++)
    {
        input[i] = 1;
    }
    

    // ######################### vector addition on CPU #########################

    high_resolution_clock::time_point start_cpu;
    high_resolution_clock::time_point end_cpu;
    duration<double, std::milli> duration_sec;

    //////////////////////////////////////
    start_cpu = high_resolution_clock::now(); // Get the starting timestamp
    reduce_cpu(input, N);
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
    reduce_gpu(input, N);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    //////////////////////////////////////

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start_gpu, stop_gpu);
    std::cout << "Total GPU time: " << ms << "ms\n";
    
    // free host memory
    

    return 0;
}