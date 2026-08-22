// Author: Yash Deshpande
// Date  : 14-08-2026
// Tutor : T.W. Huang
// Ref   : 3_reading_material/1_UW_ece759/10_openmp-loop-scheduling.pdf
//         slide 20 (PDF page 20) - "The 'for' OpenMP Construct: different
//         ways to invoke it" - the RIGHT-HAND box.
//         The left-hand box is 4_for_inside_one_parallel_region.cpp.

// Build:  g++ 5_parallel_for_separate_regions.cpp -std=c++17 -O2 -fopenmp
// Run:    ./a.out

#include <cstdio>
#include <omp.h>

#define MAX 8

int res1[MAX], res2[MAX];

int huge1() { return 1; }   // stand-ins for the slide's expensive functions
int huge2() { return 2; }

int main() {
  omp_set_num_threads(4);

  // "parallel for" is the COMBINED construct: it forks a team AND work-shares
  // the loop across it, then joins. Equivalent to writing "omp parallel"
  // wrapped around a lone "omp for".
#pragma omp parallel for
  for (int i = 0; i < MAX; i++) {
    res1[i] = huge1();
  } // <- implicit barrier AND join: the team is destroyed here

  // We are back to single-threaded execution between the loops.
  std::printf("between the loops: serial, %d thread\n", omp_get_num_threads());

  // A brand-new team is forked for the second loop.
#pragma omp parallel for
  for (int i = 0; i < MAX; i++) {
    res2[i] = huge2();
  } // <- implicit barrier AND join again

  std::printf("res1[0]=%d res2[0]=%d\n", res1[0], res2[0]);
  return 0;
}

// ---------------------------------------------------------------------------
// WHAT THIS FILE SHOWS: the slide's right-hand box - a separate fork/join per
// loop. Same results as 4_for_inside_one_parallel_region.cpp, more overhead.
//
// Measured proof that the team does NOT persist (this file):
//
//   between the loops: serial, 1 thread
//
// The other style prints "still one team of 4" at the same point. That single
// line is the whole difference between the two boxes on the slide.
//
// COST. Both styles do identical arithmetic. With MAX = 20000 and a real
// workload in huge1/huge2, 200 repetitions each, 16 threads on this machine:
//
//   one parallel + two omp for (file 4) : 0.049238 s avg
//   this file (two parallel for)        : 0.061646 s avg   (1.25x slower)
//
// You pay fork/join twice instead of once. With N loops it is N forks vs 1.
// The penalty shrinks as the per-loop work grows, and matters most for many
// short loops.
//
// THE BARRIERS (the slide's red annotation). Each "parallel for" ends with an
// implicit barrier plus a join, so this file has 2 of each. Note you CANNOT
// put "nowait" on a combined "parallel for" - the barrier at the end of a
// parallel region is mandatory and cannot be removed
// (12_openmp-synchronization-and-reduction.pdf, slide 24: "parallel -
// necessary barrier - cannot be removed"). That optimisation is only available
// in the file-4 style, where the inner "omp for" barriers are removable.
//
// SO WHY USE THIS STYLE AT ALL?
//   - One loop to parallelise: it is shorter and clearer, with no cost
//     difference at all.
//   - The loops are far apart in the code, or in different functions.
//   - You WANT serial code to run between them (as printed above).
//
// RULE OF THUMB: several loops close together -> file 4's style. A single
// isolated loop -> this style.
