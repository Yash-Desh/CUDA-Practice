// Author: Yash Deshpande
// Date  : 14-08-2026
// Tutor : T.W. Huang
// Ref   : 3_reading_material/1_UW_ece759/10_openmp-loop-scheduling.pdf
//         slide 1 (PDF page 1) - "Code of the Day"
//         "Is the following outcome guaranteed?"

// Build:  g++ 0_code_of_the_day_data_race.cpp -std=c++17 -O3 -fopenmp
// Run:    OMP_NUM_THREADS=2 ./a.out
//         OMP_NUM_THREADS=3 ./a.out
//         OMP_NUM_THREADS=4 ./a.out

#include <cstdio>
#include <omp.h>

int main() {
  int N = 12;
  int a = 0;
  double start = omp_get_wtime();

#pragma omp parallel
  {
    // Only ONE thread prints the count. #pragma omp single has an implicit
    // barrier at its end, so every thread waits here before the loop.
#pragma omp single
    std::printf("# threads: %d\n", omp_get_num_threads());

    // NOTE: this is a plain for loop, NOT "#pragma omp for". There is no
    // work-sharing, so the iteration space is NOT split across threads -
    // EVERY thread runs all N iterations. That is the crux of the slide.
    for (int i = 0; i < N; i++) {
      a++; // <- THE BUG: unsynchronized read-modify-write on shared `a`
    }
  }

  double end = omp_get_wtime();
  std::printf("Work took: %f seconds\n", end - start);
  std::printf("Result: a = %d\n", a);
  return 0;
}

