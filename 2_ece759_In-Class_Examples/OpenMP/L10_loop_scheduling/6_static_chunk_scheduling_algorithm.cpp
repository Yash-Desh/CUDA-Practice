// Author: Yash Deshpande
// Date  : 14-08-2026
// Tutor : T.W. Huang
// Ref   : 3_reading_material/1_UW_ece759/10_openmp-loop-scheduling.pdf
//         slide 27 (PDF page 27) - "Static Scheduling Algorithm:
//         static,chunk_size"
//         Slide 28 is the dynamic algorithm, slide 29 the guided one.

// Build:  g++ 6_static_chunk_scheduling_algorithm.cpp -std=c++17 -O2 -fopenmp
// Run:    ./a.out

#include <algorithm>   // std::min
#include <cstdio>
#include <omp.h>
#include <vector>

// The slide's algorithm, verbatim.
//   N          - total number of iterations
//   W          - number of workers (threads) in the team
//   curr_b     - this worker's FIRST chunk start; caller passes w * chunk_size
//   chunk_size - iterations per chunk
//   func       - called once per chunk with the half-open range [b, e)
//
// Each worker walks the iteration space in steps of `stride`, so the chunks
// land on the threads round-robin: worker w takes chunks w, w+W, w+2W, ...
template <typename F>
void loop(size_t N, size_t W, size_t curr_b, size_t chunk_size, F&& func) {
  // defining the stride size for each round
  size_t stride = W * chunk_size;
  // each partition range is indicated by [curr_b, curr_e)
  while (curr_b < N) {
    size_t curr_e = std::min(curr_b + chunk_size, N);  // min() clamps the tail
    func(curr_b, curr_e);
    curr_b += stride;
  }
}

int main() {
  const size_t N = 20, W = 3, CHUNK = 2;

  std::vector<int> by_hand(N, -1), by_omp(N, -1);

  // Drive the slide's algorithm by hand, one worker per thread. Note the
  // starting offset w * CHUNK - that is what staggers the workers so they do
  // not all begin at chunk 0.
#pragma omp parallel num_threads(W)
  {
    size_t w = omp_get_thread_num();
    loop(N, W, w * CHUNK, CHUNK, [&](size_t b, size_t e) {
      for (size_t i = b; i < e; i++) by_hand[i] = static_cast<int>(w);
    });
  }

  // Let OpenMP do the same thing with its built-in clause.
#pragma omp parallel for num_threads(W) schedule(static, CHUNK)
  for (size_t i = 0; i < N; i++) by_omp[i] = omp_get_thread_num();

  std::printf("iteration    :");
  for (size_t i = 0; i < N; i++) std::printf("%3zu", i);
  std::printf("\nslide algo   :");
  for (size_t i = 0; i < N; i++) std::printf("%3d", by_hand[i]);
  std::printf("\nschedule()   :");
  for (size_t i = 0; i < N; i++) std::printf("%3d", by_omp[i]);
  std::printf("\nidentical    : %s\n", by_hand == by_omp ? "YES" : "NO");
  return 0;
}

// ---------------------------------------------------------------------------
// WHAT THIS FILE SHOWS: slide 27's pseudocode is not an analogy - it is what
// schedule(static, chunk_size) actually does. Running it (N=20, W=3, chunk=2)
// gives a byte-for-byte identical assignment:
//
//   iteration    :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
//   slide algo   :  0  0  1  1  2  2  0  0  1  1  2  2  0  0  1  1  2  2  0  0
//   schedule()   :  0  0  1  1  2  2  0  0  1  1  2  2  0  0  1  1  2  2  0  0
//   identical    : YES
//
// READING THE PATTERN: chunks of 2, dealt round-robin like playing cards.
// stride = W * chunk_size = 6, so thread 0 handles [0,2), [6,8), [12,14),
// [18,20) - each start is 6 apart, exactly as the slide's diagram shows.
//
// THE TWO SUBTLE LINES:
//   1. curr_b starts at w * chunk_size, NOT 0. That offset is what gives each
//      worker a different starting chunk. It is passed in by the caller, which
//      is why it is a parameter rather than computed inside loop().
//   2. std::min(curr_b + chunk_size, N) clamps the LAST chunk. With N = 20 and
//      chunk 2 it never triggers, but with N = 19 thread 0's final chunk would
//      be [18,19) instead of [18,20). Without the min() you would run off the
//      end of the array.
//
// "STATIC" MEANS DECIDED UP FRONT. Every worker computes its own chunk starts
// from w, W, N and chunk_size alone - no shared state, no communication, no
// atomics. That is the entire advantage over dynamic scheduling (slide 28),
// which needs an atomic `next` counter that all threads contend on. Static has
// near-zero scheduling overhead; the price is that it cannot react if some
// iterations turn out to be slower than others.
//
// CHUNK SIZE CHANGES THE SHAPE COMPLETELY. Measured here with N=20, W=3:
//
//   schedule(static)     :  0 0 0 0 0 0 0 1 1 1 1 1 1 1 2 2 2 2 2 2
//   schedule(static,1)   :  0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1
//   schedule(static,7)   :  0 0 0 0 0 0 0 1 1 1 1 1 1 1 2 2 2 2 2 2
//
//   - No chunk_size given: ONE contiguous block per thread, roughly N/W each
//     (slide 22: "each thread is assigned std::ceil(n/thread_count)
//     iterations"). This is the default and is NOT the same as static,1.
//   - chunk_size 1: pure round-robin, one iteration at a time. Best load
//     balance for the static family, worst cache locality.
//   - chunk_size 7 here coincides with the default because ceil(20/3) = 7.
//
// TRADE-OFF: big chunks give good cache locality (each thread sweeps a
// contiguous run) but poor balance if per-iteration cost varies. Small chunks
// balance better but scatter memory access. Static is the right default when
// every iteration costs about the same; when it does not, see slide 28's
// dynamic or slide 29's guided.
//
// Verified on this machine with g++ 13.3.0.
