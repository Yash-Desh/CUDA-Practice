// Author : Yash Deshpande
// Date : 07-01-2025
// Tutor : Izzat El Hajj, AUB

#include <iostream>
#include <cuda.h>
#include <random>
#include <chrono>
#include <ratio>

using std::chrono::duration;
using std::chrono::high_resolution_clock;

#define BLOCK_DIM 8
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
#define C0 -6
#define C1 1

// prints all the elements of given array
void printArr(float *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

void stencil_cpu(float *in, float *out, unsigned int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                // Boundary Condition : Last elements will not be calculated
                if (i >= 1 && i < (N - 1) && j >= 1 && j < (N - 1) && k >= 1 && k < (N - 1))
                {
                    // hard-code the elements to be added
                    // i * N * N + j * N + k = (i*N + j)*N + k
                    out[(i * N + j) * N + k] = C0 * (in[(i * N + j) * N + k]) +
                                               C1 * (in[(i * N + j) * N + (k - 1)] +
                                                     in[(i * N + j) * N + (k + 1)] +
                                                     in[(i * N + (j - 1)) * N + k] +
                                                     in[(i * N + (j + 1)) * N + k] +
                                                     in[((i - 1) * N + j) * N + k] +
                                                     in[((i + 1) * N + j) * N + k]);
                }
            }
        }
    }
}

__global__ void stencil_tiled_kernel(float *in, float *out, unsigned int N)
{
    int i = blockDim.z * OUT_TILE_DIM + threadIdx.z - 1;
    int j = blockDim.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockDim.x * OUT_TILE_DIM + threadIdx.x - 1;

    // declare & load shared memory
    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    // Boundary Condition for Shared Memory
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N)
    {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[(i * N + j) * N + k];
    }
    __syncthreads();

    // Boundary Condition : Last elements will not be calculated
    if (i >= 1 && i < (N - 1) && j >= 1 && j < (N - 1) && k >= 1 && k < (N - 1))
    {
        // Boundary Condition : stencil operation carried out only for internal elements
        if ((threadIdx.x >= 1 && threadIdx.x < blockDim.x - 1) && (threadIdx.y >= 1 && threadIdx.y < blockDim.y - 1) && (threadIdx.z >= 1 && threadIdx.z < blockDim.z - 1))
        {
            // hard-code the elements to be added
            // i * N * N + j * N + k = (i*N + j)*N + k
            out[(i * N + j) * N + k] = C0 * (in_s[threadIdx.z][threadIdx.y][threadIdx.x]) +
                                       C1 * (in_s[threadIdx.z][threadIdx.y][threadIdx.x-1] +
                                             in_s[threadIdx.z][threadIdx.y][threadIdx.x+1] +
                                             in_s[threadIdx.z][threadIdx.y-1][threadIdx.x] +
                                             in_s[threadIdx.z][threadIdx.y+1][threadIdx.x] +
                                             in_s[threadIdx.z-1][threadIdx.y][threadIdx.x] +
                                             in_s[threadIdx.z+1][threadIdx.y][threadIdx.x]);
        }
    }
}

void stencil_tiled_gpu(float *in, float *out, unsigned int N)
{
    // for timing purpose
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU memory
    float *in_d, *out_d;
    cudaMalloc((void **)&in_d, N * N * N * sizeof(float));
    cudaMalloc((void **)&out_d, N * N * N * sizeof(float));

    // Copy Data to GPU
    cudaMemcpy(in_d, in, N * N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Call Kernel
    dim3 numThreadsPerBlock(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 numBlocks((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    //////////////////////////////////////
    cudaEventRecord(start);
    stencil_tiled_kernel<<<numBlocks, numThreadsPerBlock>>>(in_d, out_d, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //////////////////////////////////////

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Total GPU Kernel time: " << ms << "ms\n";

    // Copy Data from GPU
    cudaMemcpy(out, out_d, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate GPU Memory
    cudaFree(in_d);
    cudaFree(out_d);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char **argv)
{
    cudaDeviceSynchronize();

    // ######################### Allocate memory & initialize data #########################

    // Declare the host arrays
    unsigned int N = (argc > 1) ? atoi(argv[1]) : (512);

    float *in = (float *)malloc(N * N * N * sizeof(float));
    float *out = (float *)malloc(N * N * N * sizeof(float));

    // Generate Random Numbers
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    // Initialize the arrays
    for (int i = 0; i < (N * N * N); i++)
    {
        in[i] = dist(generator);
    }

    // ######################### vector addition on CPU #########################

    high_resolution_clock::time_point start_cpu;
    high_resolution_clock::time_point end_cpu;
    duration<double, std::milli> duration_sec;

    //////////////////////////////////////
    start_cpu = high_resolution_clock::now(); // Get the starting timestamp
    stencil_cpu(in, out, N);
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
    stencil_tiled_gpu(in, out, N);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    //////////////////////////////////////

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start_gpu, stop_gpu);
    std::cout << "Total GPU time: " << ms << "ms\n";

    // free host memory
    free(in);
    free(out);

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}