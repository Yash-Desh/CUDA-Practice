// Author: Yash Deshpande
// Date  : 11-08-2026
// Tutor : T.W. Huang
// Ref   : reading_material/UW_ece759/9 openmp-basics.pdf
//         slide 37 (PDF page 34) - "What's the Scope of a Directive like
//         #pragma omp parallel ?", right-hand box: Not a "structured block"

// THIS FILE IS MEANT TO FAIL TO COMPILE. It is the counter-example from the
// slide: the gotos branch *into* and *out of* the parallel region, so the code
// the directive covers is not a structured block.
//
// Build:  g++ -std=c++17 10_structured_block_bad.cpp -fopenmp
//
// g++ 13.3.0 rejects it with, among others:
//   error: jump to label 'more'
//   note:   enters OpenMP structured block
//   error: jump to label 'done'
//   note:   exits OpenMP structured block
//
// To see it compile, comment out the two offending jumps: the "goto more;"
// before the region and the "goto done;" inside it.

#include <cstdio>
#include <omp.h>

static int  res[64];
static bool go_now()          { return false; }
static int  do_big_job(int i) { return i * i; }
static bool conv(int)         { return true; }
static bool really_done()     { return true; }

int main() {
  // Illegal: jumps from outside the region to a label inside it.
  if (go_now()) goto more;

  #pragma omp parallel
  {
    int id = omp_get_thread_num();
  more: res[id] = do_big_job(id);
    // Illegal: jumps from inside the region to a label outside it. This would
    // let a thread skip the implicit barrier at the closing "}", so OpenMP
    // forbids it - only exit()- or return-styled calls may leave the block.
    if ( conv (res[id]) ) goto done;
    goto more;
  }
done: if (!really_done()) goto more;
}
