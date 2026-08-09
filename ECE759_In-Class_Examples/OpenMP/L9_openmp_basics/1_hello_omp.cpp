// Author: Yash Deshpande
// Date  : 03-08-2026
// Tutor : T.W. Huang
// Ref   : reading_material/UW_ece759/9 openmp-basics.pdf
//         slide 8 (PDF page 8) - "Example: Hello World in OpenMP"

// Build:  g++ 1_hello_omp.cpp -fopenmp
// Run:    ./a.out

#include <iostream>
#include <omp.h>

int main() {
  // All OpenMP directives start with #pragma omp. This one asks the compiler to
  // generate parallel code for the following (structured) block.
  #pragma omp parallel num_threads(4)
  {
    int me = omp_get_thread_num();
    int NT = omp_get_num_threads();

    std::cout << "Hello World. I'm thread " << me << " out of " << NT << ".\n";

    for (int i = 0; i < 2; i++)
      std::cout << "Iter:" << i << "\n";
  }
  std::cout << "All done here..." << std::endl;
}
