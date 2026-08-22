// Author: Yash Deshpande
// Date  : 03-08-2026
// Tutor : T.W. Huang
// Ref   : reading_material/UW_ece759/9 openmp-basics.pdf
//         slide 20 (PDF page 19) - "Example: Calculate Entries in a Table
//         in Parallel w/ OpenMP" (Version 2)

// Build:  g++ 4_sin_table_omp_for.cpp -fopenmp
// Run:    ./a.out

#include <omp.h>
#include <cmath>  // std::sin (not on the slide, but needed to compile)

constexpr auto PIE = 3.14159265358979323846;

int main()
{
  const int size = 256;
  double sinTable[size];  // sin table to be initialized

  // Version 2: "parallel for" forks a team of threads and splits the loop's
  // iteration space among them. Legal here only because the iterations are
  // independent - each one writes its own sinTable[n].
  #pragma omp parallel for
  for (int n = 0; n < size; ++n)
    sinTable[n] = std::sin(2 * PIE * n / size);
  // the table is now initialized
}
