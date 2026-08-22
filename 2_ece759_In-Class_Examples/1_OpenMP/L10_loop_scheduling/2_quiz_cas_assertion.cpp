// Author: Yash Deshpande
// Date  : 14-08-2026
// Tutor : T.W. Huang
// Ref   : 3_reading_material/1_UW_ece759/10_openmp-loop-scheduling.pdf
//         slide 15 (PDF page 15) - "Quiz: Can the Assertion fail?"
//         Slide 13 defines compare_exchange_strong, slide 14 benchmarks
//         CAS vs atomic increment vs mutex.
//         The FIXED version is slide 16 -> 3_quiz_cas_assertion_fixed.cpp

// Build:  g++ 2_quiz_cas_assertion.cpp -std=c++17 -pthread -O2
// Run:    ./a.out                      <- usually prints 2 and passes
//         for i in $(seq 3000); do ./a.out; done | sort | uniq -c
//                                      <- rarely prints 1 and aborts; you may
//                                         need CPU load to see it (see below)
// NOTE:   Answer to the quiz is YES, the assertion CAN fail.

#include <atomic>
#include <cassert>
#include <cstdio>
#include <thread>

// an integer that multiple threads may safely read and modify concurrently.
std::atomic<int> counter(0);

void increment() {
  // Read the current value once, and compute the value we want to store.
  // A single declaration statement creating two ordinary int variables — note neither is atomic
  int expected = counter.load(), desired = expected + 1;

  // compare_exchange_strong(expected, desired) atomically does:
  //     if (counter == expected) { counter = desired; return true;  }
  //     else                     { expected = counter; return false; }
  // Note the side effect on FAILURE: it overwrites `expected` with what the
  // counter actually held. That is the whole trick of this quiz.

  while (!counter.compare_exchange_strong(expected, desired));
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