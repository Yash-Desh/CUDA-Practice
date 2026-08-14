// Author: Yash Deshpande
// Date  : 25-04-2026
// Tutor : Izzat El Hajj, AUB
// Link  : https://youtu.be/VXv-9miExOU?si=LLiDqmRhDzHkvbHi

#include <iostream>
#include <cuda.h>
#include <random>               // generate random numbers
#include <chrono>               // std::chrono namespace provides timer functions in C++, for CPU timing
#include <ratio>                // std::ratio provides easy conversions between metric units, for CPU timing

using std::chrono::duration;
using std::chrono::high_resolution_clock;

#define NUM_BINS 256

__global__ void histogram_kernel(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < width * height) {
        unsigned char b = image[i];
        atomicAdd(&bins[b], 1);
    }
}

void histogram_gpu(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    // for timing purpose
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate GPU Memory
    unsigned char* image_d;
    unsigned int* bins_d;
    cudaMalloc((void**) &image_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**) &bins_d, NUM_BINS * sizeof(unsigned int));
    // cudaDeviceSynchronize();

    // Copy to the GPU
    cudaMemcpy(image_d, image, width * height, cudaMemcpyHostToDevice);
    cudaMemset(bins_d, 0, NUM_BINS * sizeof(unsigned int));
    // cudaDeviceSynchronize();

    // Run the GPU Code
    int numThreadsPerBlock = 1024;
    int numBlocks = (width * height + numThreadsPerBlock - 1)/numThreadsPerBlock;
    
    //////////////////////////////////////
    cudaEventRecord(start);
    histogram_kernel<<<numBlocks, numThreadsPerBlock>>> (image_d, bins_d, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //////////////////////////////////////

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Total GPU Kernel time: " << ms << "ms\n";

    // Copy from GPU
    cudaMemcpy(bins, bins_d, NUM_BINS, cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();

    // Deallocate GPU Memory
    cudaFree(image_d);
    cudaFree(bins_d);
    // cudaDeviceSynchronize();
}

void histogram_cpu(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
    for(unsigned int i = 0; i < width * height; i++) {
        unsigned char b = image[i];
        ++bins[b];
    }
}

int main(int argc, char **argv) {
    cudaDeviceSynchronize();

    // ######################### Allocate memory & initialize data #########################
    unsigned int N = (argc > 1) ? atoi(argv[1]) : (1 << 11);
    unsigned int width = N;
    unsigned int height = N;
    unsigned char* image = (unsigned char*)malloc(width * height * sizeof(unsigned char));
    unsigned int* bins = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));

    // Generate Random Numbers b/w [0, 255]
    std::random_device entropy_source;
    std::mt19937_64 generator(entropy_source());
    std::uniform_int_distribution<unsigned char> dist(0., 255.);

    for (int i = 0; i < width * height; i++)
    {
        image [i] = dist(generator);
    }
    // printArr(x, N);
    // printArr(y, N);

    // ######################### vector addition on CPU #########################

    high_resolution_clock::time_point start_cpu;
    high_resolution_clock::time_point end_cpu;
    duration<double, std::milli> duration_sec;

    //////////////////////////////////////
    start_cpu = high_resolution_clock::now();       // Get the starting timestamp
    histogram_cpu(image, bins, width, height);
    end_cpu = high_resolution_clock::now();         // Get the ending timestamp
    //////////////////////////////////////

    // Convert the calculated duration to a double using the standard library
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end_cpu - start_cpu);
    // Durations are converted to milliseconds already thanks to std::chrono::duration_cast
    std::cout << "Total CPU time: " << duration_sec.count() << "ms\n";
    std::cout << bins[NUM_BINS-1]<<"\n";

    // ######################### vector addition on GPU #########################
    cudaEvent_t start_gpu;
    cudaEvent_t stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    //////////////////////////////////////
    cudaEventRecord(start_gpu);
    histogram_gpu(image, bins, width, height);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    //////////////////////////////////////

    // Get the elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, start_gpu, stop_gpu);
    std::cout << "Total GPU time: " << ms << "ms\n";
    std::cout << bins[NUM_BINS-1]<<"\n";

    // free memory
    free(image);
    free(bins);

    return 0;
}
