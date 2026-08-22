// Author: Yash Deshpande
// Date  : 13-08-2026
// Tutor : T.W. Huang
// Ref   : reading_material/UW_ece759/9 openmp-basics.pdf
//         slide 49 (PDF page 45) - "Example: Running a Function Once in a
//         Nested Parallel Region"

// Build:  g++ 13_nested_single.cpp -fopenmp
// Run:    ./a.out          <- run it several times, the line order varies

#include <omp.h>
#include <cstdio>   // not on the slide, but printf needs it

// function to print level by just one thread
void report_num_threads(int whichLevel) {
  // Any ONE thread of the team runs this - whichever arrives first, unlike
  // "master" which always picks thread 0. The rest of the team waits at the
  // implicit barrier at the end of the single region.
  #pragma omp single
  printf("Level %d: number of threads in the team - %d\n", whichLevel, omp_get_num_threads());
}

int main() {
  omp_set_dynamic(1);   // let the runtime adjust team sizes if it wants to
  omp_set_nested(1);    // deprecated in OpenMP 5.0; modern spelling is
                        // omp_set_max_active_levels(3) - see note at the bottom
  #pragma omp parallel num_threads(2)
  {
    report_num_threads(1);
    #pragma omp parallel num_threads(3)
    {
      report_num_threads(2);
      #pragma omp parallel num_threads(4)
      {
        report_num_threads(3);
      }
    }
  }
  return(0);
}

