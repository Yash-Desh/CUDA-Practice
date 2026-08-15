# CUDA-Practice
Repository for all the CUDA &amp; OpenMP practice code

Resources 

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [[AMD] High Performance Gluon Kernels for gfx9](https://github.com/ROCm/gfx950-gluon-tutorials)

Open Questions

CUDA
1. [Shared Memory] What happens when your thread block requires more shared memory than what is allowed ? 
2. [Global Memory] Is memory coalescing applicable to both loads & stores ? 
3. What is a GEMM ? --> [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
4. What is GFLOPs ? How is it calculated ? 
5. What is arithmetic intensity ? How is it calculated ? 
6. When does occupancy not relate to better performance ? 



OpenMP
1. [OpenMP][L9] What are thread pools ? 
2. [OpenMP][L9] What is OpenMP run time library ? 
3. What is a loop-carried dependency ? 

