// Author: Yash Deshpande
// Date  : 11-08-2026
// Tutor : T.W. Huang
// Ref   : reading_material/UW_ece759/9 openmp-basics.pdf
//         slide 42 (PDF page 38) - "A Short Side Trip: Timing an OpenMP
//         Application"

// Build:  g++ 12_omp_timing.cpp -fopenmp
// Run:    ./a.out

// omp_get_wtime()  - seconds elapsed since some fixed timepoint in the past.
//                    The timepoint itself is arbitrary but is guaranteed not to
//                    change while the program runs, so differences of two calls
//                    are meaningful even though a single value is not.
// omp_get_wtick()  - seconds between successive clock ticks, i.e. the timer's
//                    resolution. 1/wtick is that same resolution expressed in
//                    ticks per second.

#include <omp.h>

// Not on the slide, but needed for printf / rand / pow.
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
  int const N = 100000;
  double dummy[N];

  // Bracket the work with two wtime calls and subtract - that difference is the
  // wall-clock time, which is what you want for a parallel program (CPU time
  // would add up across threads instead of telling you how long you waited).
  double start = omp_get_wtime();
  for (int i = 0; i < N; i++) {
    int temp = std::rand();
    dummy[i] = 2.*temp / (std::pow(temp*temp, 1.5) + 0.2);
  }
  double end = omp_get_wtime();

  std::printf("start = %.16g\n", start);
  std::printf("end   = %.16g\n", end);
  std::printf("diff  = %.16g\n", end - start);

  double wtick = omp_get_wtick();
  std::printf("wtick   = %.16g\n", wtick);
  std::printf("1/wtick = %.16g\n", 1.0 / wtick);
}
