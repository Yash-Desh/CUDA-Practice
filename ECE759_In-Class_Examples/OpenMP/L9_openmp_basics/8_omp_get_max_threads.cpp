// Author: Yash Deshpande
// Date  : 11-08-2026
// Tutor : T.W. Huang
// Ref   : reading_material/UW_ece759/9 openmp-basics.pdf
//         slide 27 (PDF page 27) - "Example: Get the Maximum Number of Threads"

// Build:  g++ -std=c++17 8_omp_get_max_threads.cpp -fopenmp
// Run:    ./a.out
// Try:    uncomment the omp_set_num_threads(5) line and re-run - every
//         "max threads" number becomes 5, but the num_threads(3) region stays 3.

#include <cstdio>
#include <omp.h>

int main() {
  //omp_set_num_threads(5);

  // omp_get_max_threads() answers "how big a team *would* I get if I opened a
  // parallel region right now?". Unlike omp_get_num_threads(), it is meaningful
  // in serial code. Default here is the core count (or OMP_NUM_THREADS if set).
  std::printf("I can go w/ this many threads:%d\n", omp_get_max_threads());

  // No clause, so the team really is omp_get_max_threads() threads wide.
  #pragma omp parallel
  #pragma omp master
  {
    std::printf("Here's how many threads I use in this parallel region: %d\n",
                omp_get_num_threads());
  }

  #pragma omp parallel num_threads(3)
  #pragma omp master
  {
    // These two disagree on purpose: num_threads(3) sized *this* team, but it
    // did not change the runtime's default for future regions, which is what
    // omp_get_max_threads() keeps reporting.
    std::printf("Max. number of threads: %d\n", omp_get_max_threads());
    std::printf("Actual number of threads used in this other parallel region: %d\n",
                omp_get_num_threads());
  }

  // Unchanged by the num_threads(3) clause above.
  std::printf("Here's the max number of threads at end:%d\n", omp_get_max_threads());

  return 0;
}
