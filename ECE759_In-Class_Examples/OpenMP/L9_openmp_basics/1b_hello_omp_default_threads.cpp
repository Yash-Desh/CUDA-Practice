// Author: Yash Deshpande
// Date  : 11-08-2026
// Tutor : T.W. Huang
// Ref   : reading_material/UW_ece759/9 openmp-basics.pdf
//         slide 8 (PDF page 8) - "Example: Hello World in OpenMP"
//         Same example, but with a bare "#pragma omp parallel" instead of
//         "num_threads(4)". Team size then comes from omp_get_max_threads(),
//         see slides 26-27 (PDF pages 26-27).

// Build:  g++ 1b_hello_omp_default_threads.cpp -fopenmp
// Run:    ./a.out
//         OMP_NUM_THREADS=4 ./a.out    <- pick the team size from the shell

#include <iostream>
#include <omp.h>

int main() {
  // No num_threads() clause here, so we don't hard-code the team size. The
  // runtime forms the team with omp_get_max_threads() threads, which by default
  // follows OMP_NUM_THREADS (or the core count if that isn't set).
  #pragma omp parallel
  {
    int myId    = omp_get_thread_num();
    int nThreads = omp_get_num_threads();

    std::cout << "Hello World. I'm thread " << myId
              << " out of " << nThreads << ".\n";

    for( int i = 0; i < 2; i++ )
      std::cout << "Iter:" << i << "\n";
  }
  // Implicit barrier at the closing brace: the master thread only gets here
  // once every thread in the team has finished the region.
  std::cout << "All done here..." << std::endl;
}
