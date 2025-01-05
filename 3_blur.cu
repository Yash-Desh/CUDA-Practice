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

// Blur Operation : output pixel = average of 3x3 input pixels
// x x x
// x o x
// x x x
#define BLUR_SIZE 1

// prints all the elements of given array
void printArr(unsigned char *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << int(arr[i]) << " ";
    }
    std::cout << std::endl;
}

void blur_cpu(unsigned char *image, unsigned char *blurred, unsigned int width, unsigned int height)
{
    for (int out_row = 0; out_row < height; out_row++)
    {
        for (int out_col = 0; out_col < width; out_col++)
        {
            unsigned int average = 0;
            for (int in_row = out_row - BLUR_SIZE; in_row < out_row + BLUR_SIZE + 1; in_row++)
            {
                for (int in_col = out_col - BLUR_SIZE; in_col < out_col + BLUR_SIZE + 1; in_col++)
                {
                    // boundary condition for input pixels
                    if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width)
                    {
                        average += image[in_row * width + in_col];
                    }
                }
            }
            blurred[out_row * width + out_col] = (unsigned char)(average / ((2 * BLUR_SIZE + 1) * (2 * BLUR_SIZE + 1)));
        }
    }
}

__global__ void blur_kernel(unsigned char *image, unsigned char *blurred, unsigned int width, unsigned int height)
{
    // NOTE : 1 thread per every output pixel
    // index of the output pixels
    int out_row = blockDim.y * blockIdx.y + threadIdx.y;
    int out_col = blockDim.x * blockIdx.x + threadIdx.x;

    // boundary condition for output pixels
    if (out_row < height && out_col < width)
    {
        unsigned int average = 0;
        for (int in_row = out_row - BLUR_SIZE; in_row < out_row + BLUR_SIZE + 1; in_row++)
        {
            for (int in_col = out_col - BLUR_SIZE; in_col < out_col + BLUR_SIZE + 1; in_col++)
            {
                // boundary condition for input pixels
                if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width)
                {
                    average += image[in_row * width + in_col];
                }
            }
        }
        blurred[out_row * width + out_col] = (unsigned char)(average / ((2 * BLUR_SIZE + 1) * (2 * BLUR_SIZE + 1)));
    }
}

void blur_gpu(unsigned char *image, unsigned char *blurred, unsigned int width, unsigned int height)
{
    // for timing purpose
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU Memory
    unsigned char *image_d, *blurred_d;
    cudaMalloc((void **)&image_d, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&blurred_d, width * height * sizeof(unsigned char));

    // Copy data to GPU
    cudaMemcpy(image_d, image, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Call Kernel
    dim3 numThreadsPerBlock(16, 16);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (height + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

    //////////////////////////////////////
    cudaEventRecord(start);
    blur_kernel<<<numBlocks, numThreadsPerBlock>>>(image_d, blurred_d, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //////////////////////////////////////

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Total GPU Kernel time: " << ms << "ms\n";

    // Copy data from GPU
    cudaMemcpy(blurred, blurred_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Deallocate GPU memory
    cudaFree(image_d);
    cudaFree(blurred_d);
}

int main(int argc, char **argv)
{
    cudaDeviceSynchronize();

    // ######################### Allocate memory & initialize data #########################

    // Declare the host arrays
    unsigned int width = (argc > 1) ? atoi(argv[1]) : (1024);
    unsigned int height = (argc > 2) ? atoi(argv[2]) : (1024);

    unsigned char *image = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned char *blurred = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    // Generate Random Numbers
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_int_distribution<int> dist(0, 255);

    // Initialize the arrays
    for (int i = 0; i < (width * height); i++)
    {
        image[i] = dist(generator);
    }
    // printArr(image, width * height);

    // ######################### vector addition on CPU #########################

    high_resolution_clock::time_point start_cpu;
    high_resolution_clock::time_point end_cpu;
    duration<double, std::milli> duration_sec;

    //////////////////////////////////////
    start_cpu = high_resolution_clock::now(); // Get the starting timestamp
    blur_cpu(image, blurred, width, height);
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
    blur_gpu(image, blurred, width, height);
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

/*

1. Why is the variable "average" an unsigned int ?
range of unsigned char is 0-255
The sum of multiple unsigned char could be greater than 255 which will exceed unsigned char range
when we divide the sum by the number of elements, the resultant average value will again be within
the correct range of unsigned char,
hence before we set it in the output matrix, we convert the average to unsigned char


*/