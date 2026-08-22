// Author: Yash Deshpande
// Date  : 11-08-2026
// Tutor : T.W. Huang
// Ref   : reading_material/UW_ece759/9 openmp-basics.pdf
//         slide 40 (PDF page 36) - "Example: Printing Messages from Multiple
//         Threads"

// Build:  g++ 11_garbled_output.cpp -fopenmp
// Run:    ./a.out          <- run it several times, the output differs each run

#include <omp.h>
#include <iostream>

void whatsUpQuestionMark()
{
  int me = omp_get_thread_num();

  // This ONE source line is really three separate calls to operator<<, because
  // "<<" is just a function on std::cout. Nothing stops another thread from
  // running its own calls in between ours, so the pieces of different threads'
  // messages get interleaved. Roughly what the compiler is doing here:
  //
  //   void whatsUpQuestionMark() {
  //     int me = omp_get_thread_num();
  //     printf("What's up?, asks thread ");
  //     printf("%d", me);
  //     printf("\n", me);
  //   }
  //
  // The statement is not atomic - only each individual << is.
  std::cout << "What's up?, asks thread " << me << "\n";
}

int main()
{
  #pragma omp parallel num_threads(4)
  {
    whatsUpQuestionMark();
  }
  // The order in which the four threads execute the function is
  // non-deterministic, so expect garbled, run-to-run-different output above.
  // Only "all done..." is reliably last, thanks to the implicit barrier at the
  // closing "}" of the parallel region.
  std::cout<< "all done...\n";
  return 0;
}
