// Author: Yash Deshpande
// Date  : 03-08-2026
// Tutor : T.W. Huang
// Ref   : reading_material/UW_ece759/9 openmp-basics.pdf
//         slide 9 (PDF page 9) - "Example: Hello World in C++ Thread
//         (for Comparison Purpose)"

// Build:  g++ 2_hello_thread.cpp 
// Run:    ./a.out

#include <iostream>
#include <thread>

int main() {
  std::thread threads[4];

  // The std::thread equivalent of what "#pragma omp parallel num_threads(4)"
  // does for us: spawn the threads by hand, each running the same lambda.
  for (int t = 0; t < 4; t++) {
    threads[t] = std::thread([me = t, NT = 4] {

      std::cout << "Hello World. I'm thread " << me << " out of " << NT << ".\n";

      for (int i = 0; i < 2; i++) {
        std::cout << "Iter:" << i << "\n";
      }
    });
  }

  // ...and join them by hand too, which the implicit barrier at the end of an
  // omp parallel region would have taken care of.
  for (int t = 0; t < 4; t++) {
    threads[t].join();
  }
  std::cout << "All done here..." << std::endl;
}
