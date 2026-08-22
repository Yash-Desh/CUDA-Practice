// Author: Yash Deshpande
// Date  : 14-08-2026
// Tutor : T.W. Huang
// Ref   : 3_reading_material/1_UW_ece759/10_openmp-loop-scheduling.pdf
//         slide 8 (PDF page 8) - "Use OpenMP atomic to avoid Data Race during
//         Work Sharing". Slide 6 states the data race, slide 7 defines the
//         atomic directive, slide 9 covers how it is implemented.
//         This is 0_code_of_the_day_data_race.cpp with ONE line added.

// Build:  g++ 1_atomic_fixes_data_race.cpp -std=c++17 -O3 -fopenmp
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
#pragma omp single
    std::printf("# threads: %d\n", omp_get_num_threads());

    // Still a plain for loop, NOT "#pragma omp for" - so every thread still
    // runs all N iterations and the answer is still N*nthreads, not N.
    // atomic fixes the RACE, not the replicated work. See "WHAT THIS DOES
    // NOT FIX" below.
    for (int i = 0; i < N; i++) {
#pragma omp atomic
      a++; // makes the update on "a" atomic
    }
  }

  double end = omp_get_wtime();
  std::printf("Work took: %f seconds\n", end - start);
  std::printf("Result: a = %d\n", a);
  return 0;
}

