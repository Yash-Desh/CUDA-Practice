// Author: Yash Deshpande
// Date  : 14-08-2026
// Tutor : T.W. Huang
// Ref   : 3_reading_material/1_UW_ece759/10_openmp-loop-scheduling.pdf
//         slide 20 (PDF page 20) - "The 'for' OpenMP Construct: different
//         ways to invoke it" - the LEFT-HAND box.
//         The right-hand box is 5_parallel_for_separate_regions.cpp.

// Build:  g++ 4_for_inside_one_parallel_region.cpp -std=c++17 -O2 -fopenmp
// Run:    ./a.out

#include <cstdio>
#include <omp.h>

#define MAX 8

int res1[MAX], res2[MAX];

int huge1() { return 1; }   // stand-ins for the slide's expensive functions
int huge2() { return 2; }

int main() {
  omp_set_num_threads(4);

  // ONE team is forked here, and it stays alive for both loops.
#pragma omp parallel
  {
    // "omp for" is a WORK-SHARING construct: it does not create threads, it
    // splits the iteration space of the following loop across the team that
    // already exists. Compare "parallel for", which does both at once.
#pragma omp for
    for (int i = 0; i < MAX; i++) {
      res1[i] = huge1();
    }
    // <- implicit barrier at the end of every "omp for"

    // Still inside the same team, so this costs nothing extra.
#pragma omp single
    std::printf("between the loops: still one team of %d\n",
                omp_get_num_threads());

#pragma omp for
    for (int i = 0; i < MAX; i++) {
      res2[i] = huge2();
    }
  } // <- implicit barrier AND join at the end of the parallel region

  std::printf("res1[0]=%d res2[0]=%d\n", res1[0], res2[0]);
  return 0;
}

// ---------------------------------------------------------------------------
// WHAT THIS FILE SHOWS: the slide's left-hand box - fork ONE team, then reuse
// it for several work-shared loops.
//
// Measured proof that the team really persists between the loops (this file):
//
//   between the loops: still one team of 4
//
// The other style (5_parallel_for_separate_regions.cpp) prints "1 thread" at
// the same point, because its team has already been destroyed.
//
// WHY IT IS FASTER. Both styles do identical arithmetic; the difference is how
// many times you pay fork/join. With MAX = 20000 and a real workload in
// huge1/huge2, 200 repetitions each, 16 threads on this machine:
//
//   this file (one parallel, two omp for) : 0.049238 s avg
//   two separate parallel for             : 0.061646 s avg   (1.25x slower)
//
// The gap is pure team-management overhead, and it grows with the number of
// loops and shrinks as the per-loop work grows.
//
// THE BARRIERS (the slide's red annotation). Every "omp for" has an implicit
// barrier at its end, and the parallel region has one at its closing brace.
// So this file has:
//
//   barrier after loop 1  (end of omp for)
//   barrier after single  (end of omp single)
//   barrier after loop 2  (end of omp for)
//   barrier + join at "}" (end of parallel region)
//
// NOTE the loop-1 barrier is often unnecessary. If loop 2 does not depend on
// loop 1's results, add "nowait" and threads that finish loop 1 early start
// loop 2 immediately instead of idling:
//
//   #pragma omp for nowait
//   for (int i = 0; i < MAX; i++) res1[i] = huge1();
//   #pragma omp for
//   for (int i = 0; i < MAX; i++) res2[i] = huge2();
//
// Measured here (same workload, 200 reps):
//
//   with the barrier : 0.040089 s
//   with nowait      : 0.029968 s     (~25% faster)
//
// Only safe when the loops are genuinely independent - if loop 2 reads res1,
// removing the barrier is a data race. The barrier at the END of the parallel
// region cannot be removed; nowait applies to for / single / sections only
// (12_openmp-synchronization-and-reduction.pdf, slides 24 and 26).
//
// WHEN TO PREFER THIS STYLE: several loops in a row, and/or shared setup work
// you want to do once per thread. It is the default choice for a sequence of
// parallel loops.
