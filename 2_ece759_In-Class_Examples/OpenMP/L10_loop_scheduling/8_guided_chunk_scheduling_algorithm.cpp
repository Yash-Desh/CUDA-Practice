// Author: Yash Deshpande
// Date  : 14-08-2026
// Tutor : T.W. Huang
// Ref   : 3_reading_material/1_UW_ece759/10_openmp-loop-scheduling.pdf
//         slide 29 (PDF page 29) - "Guided Scheduling Algorithm:
//         guided,chunk_size"
//         Slide 27 = static  -> 6_static_chunk_scheduling_algorithm.cpp
//         Slide 28 = dynamic -> 7_dynamic_chunk_scheduling_algorithm.cpp

// Build:  g++ 8_guided_chunk_scheduling_algorithm.cpp -std=c++17 -O2 -fopenmp
// Run:    ./a.out          <- watch the chunk sizes shrink

#include <algorithm>   // std::min
#include <atomic>
#include <cstdio>
#include <mutex>
#include <omp.h>
#include <vector>

std::mutex io;   // only so the trace below prints one line at a time

// The slide's algorithm. NOTE: slide 29 shows a code FRAGMENT with no function
// signature - the parameters below (N, W, next, chunk_size, func) are inferred
// from slides 27 and 28, which use the same shape.
//
// Idea: hand out BIG chunks while there is a lot left, shrinking as the
// remaining work runs out, then switch to plain dynamic for the tail. You get
// most of dynamic's load balancing for far fewer atomic operations.
template <typename F>
void loop(size_t N, size_t W, std::atomic<size_t>& next, size_t chunk_size,
          F&& func) {
  size_t p1 = 2 * W * (chunk_size + 1);      // threshold: when to go fine-grained
  float  p2 = 1 / static_cast<float>(W);     // the "divided by workers" factor
  size_t curr_b = next.load();

  while (curr_b < N) {
    size_t r = N - curr_b;                   // iterations still unassigned

    // ---- fine-grained tail: identical to slide 28's dynamic scheduler ----
    if (r < p1) {
      while (1) {
        curr_b = next.fetch_add(chunk_size);
        if (curr_b >= N) {
          return;
        }
        func(curr_b, std::min(curr_b + chunk_size, N));
      }
      break;
    }
    // ---- coarse-grained: take a proportional slice of what is left ----
    else {
      size_t q = static_cast<size_t>(p2 * r);   // q = remaining / W
      if (q < chunk_size) {                     // never go below the floor
        q = chunk_size;
      }
      size_t curr_e = std::min(curr_b + q, N);  // slide's note: clamp the last one
      // CAS, not fetch_add: the chunk SIZE is computed from curr_b, so if
      // another worker moved `next` in the meantime our whole calculation is
      // stale and must be redone. On failure CAS refreshes curr_b for us and
      // the while loop recomputes r and q. Compare 2_quiz_cas_assertion.cpp -
      // this is the same retry pattern, done correctly.
      if (next.compare_exchange_strong(curr_b, curr_e)) {
        func(curr_b, curr_e);
        curr_b = next.load();
      }
    }
  } // end of while loop
}

int main() {
  const size_t N = 200, W = 4, CHUNK = 2;

  std::atomic<size_t> next{0};
  std::vector<int> owner(N, -1);

#pragma omp parallel num_threads(W)
  {
    size_t w = omp_get_thread_num();
    loop(N, W, next, CHUNK, [&](size_t b, size_t e) {
      {
        std::lock_guard<std::mutex> g(io);
        std::printf("  thread %zu took [%3zu,%3zu)  size %zu\n", w, b, e, e - b);
      }
      for (size_t i = b; i < e; i++) owner[i] = static_cast<int>(w);
    });
  }

  bool covered = true;
  for (size_t i = 0; i < N; i++) if (owner[i] < 0) covered = false;
  std::printf("all %zu iterations covered exactly once: %s\n", N,
              covered ? "yes" : "NO");
  return 0;
}

// ---------------------------------------------------------------------------
// WHAT THIS FILE SHOWS: the chunk sizes decaying. Actual output (N=200, W=4,
// chunk_size=2), reordered by chunk start for readability:
//
//   [  0, 50)  size 50      <- 200 remaining / 4 workers
//   [ 50, 87)  size 37      <- 150 / 4
//   [ 87,115)  size 28      <- 113 / 4
//   [115,136)  size 21
//   [136,152)  size 16
//   [152,164)  size 12
//   [164,173)  size  9
//   [173,179)  size  6
//   [179,181)  size  2      <- switched to fine-grained here
//   [181,183)  size  2
//   ... 2s all the way down ...
//   [199,200)  size  1      <- the std::min clamp, exactly as the slide warns
//   all 200 iterations covered exactly once: yes
//
// Geometric decay: each chunk is 1/W of what is left, so the remainder shrinks
// by a factor of (1 - 1/W) = 3/4 each time. Big chunks early (cheap, few
// atomics), small chunks late (fine balance where it matters).
//
// THE SWITCH POINT. p1 = 2 * W * (chunk_size + 1) = 2*4*3 = 24. Once fewer
// than 24 iterations remain, the code abandons the proportional strategy and
// falls into the dynamic scheduler from slide 28. In the trace above that
// happens at iteration 179, when r = 21 < 24 - and every chunk after it is
// exactly chunk_size. This is why guided is described as "gradually decrease
// to the given chunk size": it does not decay smoothly to 1, it decays until
// it hits the floor and then runs dynamic.
//
// WHY CAS INSTEAD OF fetch_add. Dynamic can use fetch_add because its chunk
// size is a constant - grabbing "the next 2" is always valid. Guided cannot:
// q depends on r, which depends on curr_b. If another worker advances `next`
// between our load and our update, q was computed from a stale remainder.
// compare_exchange_strong makes "claim [curr_b, curr_e) only if next is still
// curr_b" a single indivisible check, and refreshes curr_b on failure so the
// loop recomputes. Note the loop body correctly recomputes r and q on retry -
// contrast the bug in 2_quiz_cas_assertion.cpp, which failed to recompute.
//
// DOES IT MATCH REAL OpenMP? Mostly. libgomp's schedule(guided,2) on the same
// problem produced chunk sizes 50 38 28 21 16 12 ... - the same geometric
// series (50, 37/38, 28, 21, 16, 12) off by rounding. The tail differs and the
// grouping varies run to run, so guided is NOT reproducible the way static is;
// like dynamic, it depends on who wins the races. A second run of the built-in
// gave a completely different grouping (174 7 19), where one thread happened
// to sweep most of the loop before the others got going.
//
// WHERE IT SITS BETWEEN THE OTHER TWO. Measured earlier on a badly imbalanced
// workload (N=256, 8 threads, expensive iterations clustered at the end):
//
//   static      : 0.5943 s
//   dynamic,1   : 0.1082 s
//   guided      : 0.1151 s     <- essentially matches dynamic here
//
// guided got dynamic's load balancing while issuing far fewer atomic
// operations (a handful of big chunks instead of 256 individual grabs).
//
// RULE OF THUMB:
//   even cost per iteration     -> static  (zero overhead, no communication)
//   wildly unpredictable cost   -> dynamic (max balance, max atomic traffic)
//   moderately skewed / unknown -> guided  (most of the balance, few atomics)
//
// Verified on this machine with g++ 13.3.0, 16 logical CPUs.
