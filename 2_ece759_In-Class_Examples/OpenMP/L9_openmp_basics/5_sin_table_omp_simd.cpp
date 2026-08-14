// Author: Yash Deshpande
// Date  : 03-08-2026
// Tutor : T.W. Huang
// Ref   : reading_material/UW_ece759/9 openmp-basics.pdf
//         slide 21 (PDF page 20) - "Example: Calculate Entries in a Table
//         using SIMD w/ OpenMP" (Version 3)

// Build:  g++ 5_sin_table_omp_simd.cpp -fopenmp
// Run:    ./a.out

#include <omp.h>
#include <cmath>  // std::sin (not on the slide, but needed to compile)

constexpr auto PIE = 3.14159265358979323846;

int main()
{
  const int size = 256;
  double sinTable[size];  // sin table to be initialized

  // Version 3: "simd" asks for vectorization rather than threading - the loop
  // still runs on one thread, but uses wide registers and vector operations to
  // compute several iterations per instruction.
  #pragma omp simd
  for (int n = 0; n < size; ++n)
    sinTable[n] = std::sin(2 * PIE * n / size);
  // the table is now initialized
}
