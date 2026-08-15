// Author: Yash Deshpande
// Date  : 14-08-2026
// Tutor : T.W. Huang
// Ref   : 3_reading_material/1_UW_ece759/10_openmp-loop-scheduling.pdf
//         slide 16 (PDF page 16) - "Quiz: Can the Assertion fail?"
//         This is the FIXED version of slide 15. The broken one is
//         2_quiz_cas_assertion.cpp - read that first.

// Build:  g++ 3_quiz_cas_assertion_fixed.cpp -std=c++17 -pthread -O2
// Run:    ./a.out
//         for i in $(seq 3000); do ./a.out; done | sort | uniq -c
// NOTE:   This version always prints 2. The assertion cannot fail.

#include <atomic>
#include <cassert>
#include <cstdio>
#include <thread>

std::atomic<int> counter(0);

void increment() {
  int expected = counter.load(), desired = expected + 1;

  // THE FIX vs slide 15: one line in the loop body.
  //
  // On failure, compare_exchange_strong has already refreshed `expected` with
  // the counter's real current value. We now recompute `desired` from that
  // refreshed value, so the next attempt always stores current + 1 rather than
  // a stale target computed from the very first load.
  while (!counter.compare_exchange_strong(expected, desired)) {
    desired = expected + 1;
  }
}

int main() {
  std::thread t1(increment);
  std::thread t2(increment);
  t1.join();
  t2.join();
  std::printf("counter = %d\n", counter.load());
  assert(counter.load() == 2);
  return 0;
}

