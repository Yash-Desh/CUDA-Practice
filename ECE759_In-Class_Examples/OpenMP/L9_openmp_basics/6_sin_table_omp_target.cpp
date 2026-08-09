// Author: Yash Deshpande
// Date  : 03-08-2026
// Tutor : T.W. Huang
// Ref   : reading_material/UW_ece759/9 openmp-basics.pdf
//         slide 22 (PDF page 21) - "Example: Calculate Entries in a Table
//         using GPU w/ OpenMP" (Version 4)

// Build:  g++ 6_sin_table_omp_target.cpp -fopenmp
// Run:    ./a.out
// Note:   Per the slide, GPU offloading is not supported by all compilers, and
//         the sin function must also exist on the target device. Plain g++
//         without an offload-capable build will just run this on the host.

#include <omp.h>
#include <cmath>  // std::sin (not on the slide, but needed to compile)

constexpr auto PIE = 3.14159265358979323846;

int main()
{
  const int size = 256;
  double sinTable[size];  // sin table to be initialized

  // Version 4: offload the loop to a GPU. "target" moves execution to the
  // device, "teams distribute" spreads iterations across teams of threads, and
  // map(from:...) copies the finished table back to the host.
  #pragma omp target teams distribute parallel for map(from:sinTable[0:256])
  for (int n = 0; n < size; ++n)
    sinTable[n] = std::sin(2 * PIE * n / size);
  // the table is now initialized
}
