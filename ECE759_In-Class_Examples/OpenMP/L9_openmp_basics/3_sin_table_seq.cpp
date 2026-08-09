// Author: Yash Deshpande
// Date  : 03-08-2026
// Tutor : T.W. Huang
// Ref   : reading_material/UW_ece759/9 openmp-basics.pdf
//         slide 19 (PDF page 18) - "Example: Calculate Entries in a Table
//         Sequentially" (Version 1)

// Build:  g++ 3_sin_table_seq.cpp -fopenmp
// Run:    ./a.out

#include <omp.h>
#include <cmath>  // std::sin (not on the slide, but needed to compile)

constexpr auto PIE = 3.14159265358979323846;

int main()
{
  const int size = 256;
  double sinTable[size];  // sin table to be initialized

  // Version 1: plain sequential loop. Every iteration is independent of the
  // others, which is exactly what makes this loop a candidate for the
  // "#pragma omp parallel for" version that comes next in the lecture.
  for (int n = 0; n < size; ++n)
    sinTable[n] = std::sin(2 * PIE * n / size);
  // the table is now initialized
}
