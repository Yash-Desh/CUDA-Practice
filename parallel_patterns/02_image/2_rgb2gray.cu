// Author : Yash Deshpande
// Date : 29-12-2024
// Tutor : Izzat El Hajj, AUB

#include <iostream>
#include <cuda.h>
#include <random>               // generate random numbers
#include <chrono>               // std::chrono namespace provides timer functions in C++, for CPU timing
#include <ratio>                // std::ratio provides easy conversions between metric units, for CPU timing

using std::chrono::duration;
using std::chrono::high_resolution_clock;

// prints all the elements of given array
void printArr(unsigned char *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << int(arr[i]) << " ";
    }
    std::cout << std::endl;
}

// performs RGB to Gray operation on the CPU
void rgb2gray_cpu(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *gray, unsigned int width, unsigned int height)
{
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            int i = row * width + col;
            gray[i] = 3 / 10 * red[i] + 6 / 10 * green[i] + 1 / 10 * blue[i];
        }
    }
}


__global__ void rgb2gray_kernel(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *gray, unsigned int width, unsigned int height)
{
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < height && col < width)
    {
        unsigned int i = row * width + col;
        // weighted average
        gray[i] = 3 / 10 * red[i] + 6 / 10 * green[i] + 1 / 10 * blue[i];
    }
}


// performs RGB to Gray operation on the GPU
void rgb2gray_gpu(unsigned char *red, unsigned char *green, unsigned char *blue, unsigned char *gray, unsigned int width, unsigned int height)
{
    // for timing purpose
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate GPU memory
    unsigned char *red_d, *green_d, *blue_d, *gray_d;
    cudaMalloc((void **)&red_d, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&green_d, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&blue_d, width * height * sizeof(unsigned char));
    cudaMalloc((void **)&gray_d, width * height * sizeof(unsigned char));
    cudaDeviceSynchronize();

    // Copy data to GPU
    cudaMemcpy(red_d, red, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Call kernel
    dim3 numThreadsPerBlock(32, 32);
    // dim3 numBlocks ((width+32-1)/32, (height+32-1)/32);
    dim3 numBlocks((width + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (height + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);

    //////////////////////////////////////
    cudaEventRecord(start);
    rgb2gray_kernel<<<numBlocks, numThreadsPerBlock>>>(red_d, green_d, blue_d, gray_d, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //////////////////////////////////////

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Total GPU Kernel time: " << ms << "ms\n";

    // Copy data from GPU
    cudaMemcpy(gray, gray_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Deallocate GPU memory
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);
}



int main(int argc, char **argv)
{
    // cudaDeviceSynchronize();

    // // ######################### Allocate memory & initialize data #########################
    unsigned int width = (argc > 1) ? atoi(argv[1]) : (1024);
    unsigned int height = (argc > 2) ? atoi(argv[2]) : (1024);
    
    unsigned char *red = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned char *green = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned char *blue = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    unsigned char *gray = (unsigned char *)malloc(width * height * sizeof(unsigned char));

    // // Generate Random Numbers b/w [10, 20]
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_int_distribution<int> dist(0, 255);

    for (int i = 0; i < (width * height); i++)
    {
        red[i] = dist(generator);
        green[i] = dist(generator);
        blue[i] = dist(generator);
    }
    // printArr(red, width * height);
    // printArr(green, width * height);
    // printArr(blue, width * height);

    // ######################### vector addition on CPU #########################

    high_resolution_clock::time_point start_cpu;
    high_resolution_clock::time_point end_cpu;
    duration<double, std::milli> duration_sec;

    //////////////////////////////////////
    start_cpu = high_resolution_clock::now(); // Get the starting timestamp
    rgb2gray_cpu(red, green, blue, gray, width, height);
    end_cpu = high_resolution_clock::now(); // Get the ending timestamp
    //////////////////////////////////////

    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end_cpu - start_cpu);
    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    std::cout << "Total CPU time: " << duration_sec.count() << "ms\n";
    

    // ######################### vector addition on GPU #########################
    cudaEvent_t start_gpu;
    cudaEvent_t stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    //////////////////////////////////////
    cudaEventRecord(start_gpu);
    rgb2gray_gpu(red, green, blue, gray, width, height);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    //////////////////////////////////////

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start_gpu, stop_gpu);
    std::cout << "Total GPU time: " << ms << "ms\n";
    
    // free memory
    free(red);
    free(green);
    free(blue);
    free(gray);

    return 0;
}

/*

1. Why is unsigned char used to store RGB values ?
RGB values are 8-bit intensities ranging from 0-255

2. Why is numThreadsPerBlock = (32, 32) ?
Explained later in the course

3. Why are arguments to the kernel always 1-dimensional ?
    In c when you allocate data dynamically, you get a pointer that points to a one-dimensional array.
    You can't really allocate multi-dimensional arrays dynamically where both dimensions of the arrays are unknown



*/