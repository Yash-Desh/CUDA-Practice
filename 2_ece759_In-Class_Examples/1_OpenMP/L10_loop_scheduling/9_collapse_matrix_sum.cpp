// Author: Yash Deshpande
// Date  : 14-08-2026
// Tutor : T.W. Huang
// Ref   : 3_reading_material/1_UW_ece759/10_openmp-loop-scheduling.pdf
//         slide 33 (PDF page 33) - "Example of OpenMP collapse"
//         Slide 31 motivates collapse, slide 32 lists its caveats.

// Build:  g++ 9_collapse_matrix_sum.cpp -std=c++17 -O2 -fopenmp
// Run:    ./a.out <m> <n> <num_threads>
//         ./a.out 2000 2000 8
// NOTE:   This is the slide's code AS GIVEN. It is correct but SLOW - see the
//         measurements at the bottom, it gets slower as you add threads.

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <omp.h>

using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::cout;

float reduce(const float* arr, const size_t m, const size_t n) {
  float sum = 0.0;
  // collapse(2) fuses the i and j loops into one iteration space of m*n, then
  // schedules that. Legal here: the loops are perfectly nested (nothing
  // between the two "for" headers) and the bounds are known up front - the two
  // structural caveats from slide 32.
  //
  // But every iteration updates the SAME `sum`, so the accumulation must be
  // protected. The slide uses atomic. That is correct, and it is also the
  // performance bug - see the numbers at the bottom.
#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
#pragma omp atomic
      sum += arr[i * n + j];
    }
  }
  return sum;
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage: " << argv[0] << " <m> <n> <num_threads>\n";
    return 1;
  }
  size_t m = std::atoi(argv[1]);
  size_t n = std::atoi(argv[2]);
  size_t t = std::atoi(argv[3]);
  float *arr = new float[m*n];

#pragma omp parallel for
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      arr[i * n + j] = 1.1f;
    }
  }

  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_sec;

  float res;
  omp_set_num_threads(t);
  start = high_resolution_clock::now();
  for (size_t i = 0; i < 10; i++) {   // Run reduce 10 times, to get a good average time
    res = reduce(arr, m, n);
  }
  end = high_resolution_clock::now();
  duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start) / 10.;

  cout << "Result: " << res << "\n";
  cout << t << " " << duration_sec.count() << "\n";
  delete[] arr;
  return 0;
}

// ---------------------------------------------------------------------------
// THE COLLAPSE PART IS FINE. Unlike slide 32's counterexample, these loops ARE
// perfectly nested, so collapse(2) compiles and gives the scheduler m*n
// iterations to spread instead of m. That is the whole point of the construct.
//
// THE ATOMIC PART IS A TRAP. Measured on this machine (2000x2000, g++ 13.3.0
// -O2, 16 logical CPUs), running the slide's code exactly as written:
//
//   threads :   1      2       4       8      16
//   ms      : 58.0   89.1   123.6   144.1   166.0
//
// It gets SLOWER with every thread you add - 2.9x worse at 16 threads than at
// 1. This is a textbook anti-scaling curve. Every one of the 4,000,000
// iterations does a locked read-modify-write on ONE address, so the threads
// serialise on that single cache line and spend their time bouncing it between
// cores. More threads = more contention = worse.
//
// THE FIX: reduction(+:sum) instead of atomic. Each thread accumulates into a
// PRIVATE copy with no contention at all, and the copies are combined once at
// the end.
//
//     #pragma omp parallel for collapse(2) reduction(+:sum)
//     for (size_t i = 0; i < m; i++)
//       for (size_t j = 0; j < n; j++)
//         sum += arr[i * n + j];        // no atomic needed
//
//   threads :   1      2       4       8      16
//   ms      : 10.1    8.4    15.5    22.1    25.9
//
// 5.8x faster at one thread. Note it still does not scale past 2 threads -
// summing an array is memory-bandwidth-bound, not compute-bound, so extra
// threads cannot help once the memory bus is saturated.
//
// THE UNCOMFORTABLE BASELINE. A plain serial loop, no OpenMP at all:
//
//   serial                 :   1.2 ms
//   atomic + collapse  (1t):  109.1 ms      <- 89x SLOWER than serial
//   reduction + collapse(1t):  26.0 ms
//
// The serial version wins outright. At -O2 the compiler vectorises it and
// streams through memory once; there is simply not enough arithmetic per byte
// to justify any parallel machinery. This example teaches collapse syntax, not
// a workload worth parallelising.
//
// A CORRECTNESS SURPRISE I HIT WHILE MEASURING. `sum` is a float, and the
// answers disagree across configurations:
//
//   exact value      : 4400000.0
//   serial / 1 thread: 4220502.0     <- 4% low!
//   4 threads        : 4443682.0
//   16 threads       : 4389003.5
//
// Nothing is racing - all of these are "correct" programs. Adding 4 million
// values of 1.1f into a single float loses precision catastrophically: once
// the running total is large, adding 1.1 to it rounds away entirely. The
// PARALLEL versions are closer to the true answer than the serial one, because
// partial sums stay small enough to keep resolution. Floating-point addition
// is not associative, so the thread count changes the result. Use double for
// the accumulator if the value matters.
//
// SUMMARY OF WHAT TO TAKE AWAY:
//   - collapse(2) is used correctly here; the loops satisfy slide 32's caveats.
//   - "#pragma omp atomic" in an inner loop is almost always the wrong tool for
//     accumulation - reach for reduction (see 1_atomic_fixes_data_race.cpp,
//     which makes the same point on a smaller example).
//   - Always compare against the serial baseline before declaring victory.
//
// Verified on this machine with g++ 13.3.0, 16 logical CPUs.
