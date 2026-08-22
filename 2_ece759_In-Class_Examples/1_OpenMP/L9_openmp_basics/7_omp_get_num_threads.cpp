// Author: Yash Deshpande
// Date  : 11-08-2026
// Tutor : T.W. Huang
// Ref   : reading_material/UW_ece759/9 openmp-basics.pdf
//         slide 26 (PDF page 26) - "Example: Set/Get Number of Threads in a
//         Parallel Region"

// Build:  g++ 7_omp_get_num_threads.cpp -fopenmp
// Run:    ./a.out

#include <iostream>
#include <omp.h>

int main() {
  // Outside any parallel region there is only the master thread running, so
  // omp_get_num_threads() reports the size of the *current* team, which is 1.
  std::cout << "Non parallel block, beginning of test: "
            << omp_get_num_threads() << "\n";

  omp_set_num_threads(2); // set the maximum number of threads to two

  // Still 1: omp_set_num_threads() only changes what the *next* parallel region
  // will fork with; it does not change the team we're in right now.
  std::cout << "Non parallel block, after omp_set_num_threads call: "
            << omp_get_num_threads() << "\n";

  // No num_threads() clause, so this team picks up the 2 we just requested.
  // "omp master" makes only thread 0 print, otherwise every thread would.
  #pragma omp parallel
  #pragma omp master
  {
    std::cout << "Inside a parallel block: " << omp_get_num_threads() << "\n";
  }

  // Back to a team of one after the region closes.
  std::cout << "No parallel block here: " << omp_get_num_threads() << std::endl;

  // changed the number of threads to be used inside parallel block
  // using a compiler directive
  // The num_threads(3) clause overrides the omp_set_num_threads(2) above, but
  // only for this one region.
  #pragma omp parallel num_threads(3)
  #pragma omp master
  {
    std::cout << "Second parallel block: " << omp_get_num_threads() << "\n";
  }

  std::cout << "Outside parallel block: " << omp_get_num_threads() << std::endl;
}
