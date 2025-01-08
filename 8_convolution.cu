// Author : Yash Deshpande
// Date : 07-01-2025
// Tutor : Izzat El Hajj, AUB

#include <iostream>
#include <cuda.h>
#include <random> // generate random numbers
#include <chrono> // std::chrono namespace provides timer functions in C++, for CPU timing
#include <ratio>  // std::ratio provides easy conversions between metric units, for CPU timing

using std::chrono::duration;
using std::chrono::high_resolution_clock;

#define OUT_TILE_DIM 32
#define MASK_RADIUS 2
#define MASK_DIM ((MASK_RADIUS) * 2 + 1)

// prints all the elements of given array
void printArr(float *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << (arr[i]) << " ";
    }
    std::cout << std::endl;
}

void convolution_cpu(float mask[][MASK_DIM], float *input, float *output, unsigned int width, unsigned int height)
{
    // loop over each row output element
    for (int out_row = 0; out_row < height; out_row++)
    {
        // loop over each col of output elements
        for (int out_col = 0; out_col < width; out_col++)
        {
            float sum = 0.0f;
            // loop over each row of mask
            for (int mask_row = 0; mask_row < MASK_DIM; mask_row++)
            {
                // loop through each col of mask
                for (int mask_col = 0; mask_col < MASK_DIM; mask_col++)
                {
                    int in_row = out_row - MASK_RADIUS + mask_row;
                    int in_col = out_col - MASK_RADIUS + mask_col;

                    // boundary condition for the input elements
                    if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width)
                    {
                        sum += input[in_row * height + in_col] * mask[mask_row][mask_col];
                    }
                }
            }
            output[out_row * width + out_col] = sum;
        }
    }
}

// declare array in constant memory
__constant__ float mask_c[MASK_DIM][MASK_DIM];

__global__ void convolution_kernel(float *input, float *output, unsigned int width, unsigned int height)
{
    // NOTE : 1 thread per every output element
    int out_row = blockDim.y * blockIdx.y + threadIdx.y;
    int out_col = blockDim.x * blockIdx.x + threadIdx.x;

    // Boundary condition for the output elements
    if (out_row < height && out_col < width)
    {
        float sum = 0.0f;

        // loop through the rows of the mask
        for (int mask_row = 0; mask_row < MASK_DIM; mask_row++)
        {
            // loop through the col of the mask
            for (int mask_col = 0; mask_col < MASK_DIM; mask_col++)
            {
                int in_row = out_row - MASK_RADIUS + mask_row;
                int in_col = out_col - MASK_RADIUS + mask_col;

                // boundary condition for the input elements
                if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width)
                {
                    sum += input[in_row * height + in_col] * mask_c[mask_row][mask_col];
                }
            }
        }

        output[out_row * width + out_col] = sum;
    }
}

void convolution_gpu(float mask[][MASK_DIM], float *input, float *output, unsigned int width, unsigned int height)
{
    // for timing purpose
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU Memory
    float *input_d, *output_d;
    cudaMalloc((void **)&input_d, width * height * sizeof(float));
    cudaMalloc((void **)&output_d, width * height * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(input_d, input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Copy Mask to constant memory
    cudaMemcpyToSymbol(mask_c, mask, MASK_DIM * MASK_DIM * sizeof(float));

    // Call the kernel
    dim3 numThreadsPerBlock(OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    //////////////////////////////////////
    cudaEventRecord(start);
    convolution_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //////////////////////////////////////

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Total GPU Kernel time: " << ms << "ms\n";

    // Copy data from GPU
    cudaMemcpy(output, output_d, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate GPU Memory
    cudaFree(input_d);
    cudaFree(output_d);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char **argv)
{
    cudaDeviceSynchronize();

    // ######################### Allocate memory & initialize data #########################

    // Declare the host arrays

    unsigned int width = (argc > 1) ? atoi(argv[1]) : (1024);
    unsigned int height = (argc > 2) ? atoi(argv[2]) : (1024);

    float *input = (float *)malloc(width * height * sizeof(float));
    float *output = (float *)malloc(width * height * sizeof(float));
    float mask[MASK_DIM][MASK_DIM];

    // Generate Random Numbers
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_real_distribution<float> dist1(-10.0, 10.0);
    std::uniform_real_distribution<float> dist2(-1.0, 1.0);

    // Initialize the arrays
    for (int i = 0; i < (width * height); i++)
    {
        input[i] = dist1(generator);
    }
    // printArr(input, width * height);

    for (int i = 0; i < MASK_DIM; i++)
    {
        for (int j = 0; j < MASK_DIM; j++)
        {
            mask[i][j] = dist2(generator);
        }
    }

    // ######################### vector addition on CPU #########################

    high_resolution_clock::time_point start_cpu;
    high_resolution_clock::time_point end_cpu;
    duration<double, std::milli> duration_sec;

    //////////////////////////////////////
    start_cpu = high_resolution_clock::now(); // Get the starting timestamp
    convolution_cpu(mask, input, output, width, height);
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
    convolution_gpu(mask, input, output, width, height);
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

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}