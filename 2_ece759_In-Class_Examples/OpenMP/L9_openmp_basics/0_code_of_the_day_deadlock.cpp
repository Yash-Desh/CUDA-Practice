// Author: Yash Deshpande
// Date  : 14-08-2026
// Tutor : T.W. Huang
// Ref   : 3_reading_material/1_UW_ece759/9_openmp-basics.pdf
//         slide 1 (PDF page 1) - "Code of the Day"
//         "What problems can you see from this code?"

// Build:  g++ 0_code_of_the_day_deadlock.cpp -std=c++17 -pthread
// Run:    timeout 5 ./a.out     <- NOTE: this program HANGS FOREVER by design.
//                                  Use timeout (exit code 124 = it deadlocked)
//                                  or Ctrl-C to get your shell back.

#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

std::mutex mutex1, mutex2;

void task1() {
  // Lock mutex1 first, then mutex2
  std::lock_guard<std::mutex> lock1(mutex1);
  std::cout << "Task 1 locked mutex1\n";

  // Simulate some work. This sleep is what makes the deadlock *deterministic*:
  // without it one thread often grabs both mutexes before the other even
  // starts, and the program looks like it works. The sleep guarantees both
  // threads hold their first lock before either reaches its second.
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  std::lock_guard<std::mutex> lock2(mutex2);
  std::cout << "Task 1 locked mutex2\n";   // never reached
}

void task2() {
  // Lock mutex2 first, then mutex1  <- THE BUG: the opposite order from task1.
  // (Note the slide's misleading names: the guard called "lock1" holds mutex2.)
  std::lock_guard<std::mutex> lock1(mutex2);
  std::cout << "Task 2 locked mutex2\n";

  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  std::lock_guard<std::mutex> lock2(mutex1);
  std::cout << "Task 2 locked mutex1\n";   // never reached
}

int main() {
  std::thread t1(task1);
  std::thread t2(task2);
  t1.join();     // blocks forever
  t2.join();
  return 0;      // "finished" never happens
}

// ---------------------------------------------------------------------------
// Observed output (g++ 13.3.0, this machine):
//
//   Task 1 locked mutex1
//   Task 2 locked mutex2
//   <hangs; timeout kills it with exit code 124>
//
// PROBLEM 1 - DEADLOCK (the main one).
//   task1 holds mutex1 and waits for mutex2; task2 holds mutex2 and waits for
//   mutex1. Circular wait, and lock_guard only releases at scope exit, which
//   neither thread can reach. All four Coffman conditions are present: mutual
//   exclusion, hold-and-wait, no preemption, circular wait.
//
//   Root cause: inconsistent lock ORDER. Nothing else.
//
// PROBLEM 2 - unsynchronized std::cout.
//   Both threads write the same stream with no mutex. "std::cout << a << b" is
//   several separate calls, so lines can interleave mid-sentence. Not the hang,
//   but a real defect. (Same effect as 11_garbled_output.cpp.)
//
// PROBLEM 3 - misleading variable names in task2.
//   The guard named "lock1" holds mutex2 and "lock2" holds mutex1. Compiles and
//   behaves fine, but it camouflages the ordering bug from a reader.
//
// FIX A - impose one global lock order everywhere (always mutex1 then mutex2).
//         Breaks the circular-wait condition.
//
// FIX B - let the library do it (C++17, preferred):
//
//         void task1() { std::scoped_lock lock(mutex1, mutex2); ... }
//         void task2() { std::scoped_lock lock(mutex2, mutex1); ... }
//
//         std::scoped_lock acquires all mutexes atomically using a
//         deadlock-avoidance algorithm, so the argument order no longer
//         matters. (std::lock is the older, more verbose equivalent.)
//
// WHY THE LECTURE OPENS WITH THIS: it motivates OpenMP's higher-level
// constructs. "#pragma omp critical" and "reduction" cannot produce this bug
// because you never name or order the locks yourself. Lecture 8's pros/cons
// table says it directly - raw std::thread "requires manual synchronization,
// increasing risk of race conditions and deadlocks" (8_cpp-thread.pdf, slide 30).
