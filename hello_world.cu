#include <stdio.h>
#include <cuda.h>
// Kernel function to print "Hello, World!" from the GPU
__global__ void helloWorldKernel() {
    printf("Hello, World from GPU thread %d!\n", threadIdx.x);
}

int main() {
    // Launch the kernel with 1 block and 10 threads
    helloWorldKernel<<<1, 10>>>();

    // Synchronize to wait for GPU to complete
    cudaDeviceSynchronize();

    return 0;
}
