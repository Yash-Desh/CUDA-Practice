// Author: Yash Deshpande
// Date  : 11-08-2026
// Tutor : T.W. Huang
// Ref   : reading_material/UW_ece759/9 openmp-basics.pdf
//         slide 37 (PDF page 34) - "What's the Scope of a Directive like
//         #pragma omp parallel ?", left-hand box: A "structured block"

// Build:  g++ -std=c++17 9_structured_block_ok.cpp -fopenmp
// Run:    ./a.out

#include <cstdio>
#include <omp.h>

// Stand-ins so the slide snippet actually builds and runs. do_big_job() just
// squares the id; not_conv() reports "not converged yet" only on the first
// visit per thread, so the goto loop runs twice and then falls out.
static int  res[64];
static bool visited[64];

static int  do_big_job(int i) { return i * i; }

static bool not_conv(int id) {
  if (visited[id]) return false;   // second time around: converged, stop looping
  visited[id] = true;
  return true;                     // first time: loop back to "more"
}

int main() {
  // The scope of the directive is the structured block below: everything from
  // the opening "{" to its matching "}".
  #pragma omp parallel num_threads(4)
  {
    int id = omp_get_thread_num();

    // A backward goto is fine here because it stays *inside* the block - it
    // never branches into or out of the structured block, so every thread
    // still reaches the closing "}".
  more: res[id] = do_big_job (id);
    if( not_conv(res[id]) )goto more;

    std::printf("thread %d converged with res=%d\n", id, res[id]);
  }
  // IMPORTANT FACT: there is an implicit barrier at that closing "}". Threads
  // executing the block wait on each other to finish - that is the point at
  // which the forked threads join. So this line is printed once, by the master,
  // and only after all four threads are done.
  printf ("I'm outside par. region!\n");
}
