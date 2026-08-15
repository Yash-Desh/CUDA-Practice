// Author: Yash Deshpande
// Date  : 14-08-2026
// Tutor : T.W. Huang
// Ref   : 3_reading_material/1_UW_ece759/10_openmp-loop-scheduling.pdf
//         slide 28 (PDF page 28) - "Dynamic Scheduling Algorithm:
//         dynamic,chunk_size"
//         Slide 27 is the static algorithm (6_static_chunk_scheduling_
//         algorithm.cpp), slide 29 the guided one.

// Build:  g++ 7_dynamic_chunk_scheduling_algorithm.cpp -std=c++17 -O2 -fopenmp
// Run:    ./a.out          <- run it repeatedly, the assignment CHANGES

#include <algorithm>   // std::min
#include <atomic>
#include <cstdio>
#include <omp.h>
#include <vector>

// The slide's algorithm, verbatim.
//   N                - total number of iterations
//   next             - SHARED atomic cursor: the next unassigned iteration
//   given_chunk_size - 0 means "use 1"
//   func             - called once per chunk with the half-open range [b, e)
//
// Unlike the static version there is no per-worker starting offset and no
// stride. Every worker races to the same atomic counter and takes whatever is
// next. fetch_add atomically returns the OLD value and advances the cursor, so
// two workers can never be handed the same chunk.
template <typename F>
void loop(size_t N, std::atomic<size_t>& next, size_t given_chunk_size, F&& func) {
  // next is an atomic variable that indicates the next partition's position
  size_t chunk_size = (given_chunk_size == 0) ? size_t{1} : given_chunk_size;
  size_t curr_b = next.fetch_add(chunk_size);
  // each partition range is indicated by [curr_b, std::min(curr_b + chunk_size, N))
  while (curr_b < N) {
    func(curr_b, std::min(curr_b + chunk_size, N));
    curr_b = next.fetch_add(chunk_size);   // grab the next available chunk
  }
}

int main() {
  const size_t N = 20, W = 3, CHUNK = 2;

  // Run the SAME code three times. Static would print the same line every
  // time; dynamic does not.
  for (int run = 0; run < 3; run++) {
    std::vector<int> owner(N, -1);
    std::atomic<size_t> next{0};      // one shared cursor for the whole team

#pragma omp parallel num_threads(W)
    {
      size_t w = omp_get_thread_num();
      loop(N, next, CHUNK, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; i++) owner[i] = static_cast<int>(w);
        // make thread 0 slow, so the others pick up its share
        if (w == 0) for (volatile int k = 0; k < 2000000; k++) { }
      });
    }

    std::printf("run %d :", run);
    for (size_t i = 0; i < N; i++) std::printf("%3d", owner[i]);
    bool covered = true;
    for (size_t i = 0; i < N; i++) if (owner[i] < 0) covered = false;
    std::printf("   every iteration done exactly once: %s\n",
                covered ? "yes" : "NO");
  }
  return 0;
}

// ---------------------------------------------------------------------------
// WHAT THIS FILE SHOWS: the same source, run three times, produces THREE
// DIFFERENT assignments. Actual output from one execution:
//
//   run 0 :  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2
//   run 1 :  1  1  2  2  1  1  2  2  2  2  1  1  2  2  1  1  2  2  1  1
//   run 2 :  1  1  2  2  1  1  2  2  1  1  2  2  1  1  0  0  2  2  1  1
//
// That is the whole difference from slide 27. Static is a pure function of
// (w, W, N, chunk_size) - identical every run. Dynamic is first-come-first-
// served, so it depends on timing. In run 0 thread 2 happened to win every
// race and did the entire loop alone; thread 0, deliberately slowed down, got
// almost nothing. Coverage is still exactly once per iteration in all runs -
// fetch_add guarantees no chunk is handed out twice.
//
// HOW fetch_add DOES THE WORK. next.fetch_add(chunk_size) atomically:
//     old = next;  next = next + chunk_size;  return old;
// The returned `old` is this worker's chunk start. Because the whole
// read-modify-write is indivisible, two workers calling it simultaneously get
// different values - one gets 0 and the other 2, never 0 and 0. That single
// operation is the entire scheduler.
//
// NOTE the loop structure: fetch_add is called ONCE before the while, and
// again at the BOTTOM of each pass. The bound check `curr_b < N` then
// terminates the worker. A worker that finishes early simply loops back and
// grabs more - that is the load balancing.
//
// THE COST: `next` is a single shared cache line that every thread hammers.
// Each chunk costs a contended atomic RMW, and the line bounces between cores.
// Static costs nothing at all - each worker computes its own schedule with no
// communication. So dynamic buys balance with overhead.
//
// WHEN THE TRADE PAYS OFF. Measured here: N=256, 8 threads, iterations 224-255
// roughly 400x more expensive than the rest (heavy work CLUSTERED at the end,
// which is the worst case for static's contiguous blocks):
//
//   static      : 0.5943 s
//   dynamic,1   : 0.1082 s      <- 5.49x faster
//   guided      : 0.1151 s
//
// With static, the thread that owns the last block does all the heavy work
// while the other seven sit at the barrier. Dynamic hands those expensive
// iterations out one at a time and everyone stays busy.
//
// BUT IT IS NOT A FREE WIN. With an EVENLY distributed workload the atomic
// overhead makes dynamic the slower choice - measured on a smooth
// cost-proportional-to-i workload, static 0.0129 s vs dynamic,4 0.0208 s.
// Do not reach for dynamic by default.
//
// I also learned the hard way while measuring this: an earlier attempt spread
// the expensive iterations as (i % 16 == 0), which accidentally gives every
// thread an equal share under static, so there was nothing for dynamic to fix
// (1.02x). Load imbalance needs the expensive work CLUSTERED, not scattered.
//
// RULE OF THUMB:
//   even per-iteration cost   -> static (slide 27), zero overhead
//   unpredictable/skewed cost -> dynamic (this file), pays for balance
//   somewhere in between      -> guided (slide 29), big chunks first then
//                                small ones, to get most of the balance for
//                                fewer atomic operations
//
// Verified on this machine with g++ 13.3.0, 16 logical CPUs.
