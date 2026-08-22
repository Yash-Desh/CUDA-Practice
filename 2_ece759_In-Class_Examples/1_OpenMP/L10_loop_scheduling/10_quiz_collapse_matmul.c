/* Author: Yash Deshpande
 * Date  : 14-08-2026
 * Tutor : T.W. Huang
 * Ref   : 3_reading_material/1_UW_ece759/10_openmp-loop-scheduling.pdf
 *         slide 35 (PDF page 35) - "Quiz: Collapsing Loop using OpenMP
 *         collapse" - "Is this code correct? If not, how can we fix it?"
 *         Slide 32 lists the collapse caveats this code violates.
 *
 * Build:  gcc 10_quiz_collapse_matmul.c -std=c11 -O2 -fopenmp -lm
 * Run:    ./a.out
 *
 * WHY .c AND NOT .cpp: the slide's signature uses variable-length array
 * parameters - double A[N][N] with a runtime N. That is legal C99/C11 but NOT
 * legal C++; g++ rejects it with "use of parameter outside function body".
 * Everything else in this directory is C++; this one file is C so the slide's
 * signature can be reproduced as written.
 *
 * ANSWER TO THE QUIZ: NO, the code is not correct. It has TWO separate
 * defects, and the first one stops it compiling before you can even hit the
 * second. See the analysis at the bottom.
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>

#define N 200

static double A[N][N], B[N][N], C[N][N], REF[N][N];

/* ------------------------------------------------------------------ *
 * THE SLIDE'S CODE, exactly as printed. This DOES NOT COMPILE.
 * Uncomment to see the error:
 *
 *   void matrix_multiply(int n, double A[n][n], double B[n][n], double C[n][n]) {
 *     #pragma omp parallel for collapse(3)
 *     for (int i = 0; i < n; i++) {
 *       for (int j = 0; j < n; j++) {
 *         C[i][j] = 0;                 // <- DEFECT 1: breaks perfect nesting
 *         for (int k = 0; k < n; k++) {
 *           C[i][j] += A[i][k] * B[k][j];
 *         }
 *       }
 *     }
 *   }
 *
 *   error: not enough perfectly nested loops before 'C'
 * ------------------------------------------------------------------ */

/* ATTEMPTED FIX 1 - hoist the initialisation so collapse(3) is legal.
 * This compiles. It is still WRONG: see the numbers at the bottom.
 * schedule(static,1) is added only to make the latent race visible; the bug
 * exists without it too. */
void broken_collapse3(void) {
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      C[i][j] = 0;

#pragma omp parallel for collapse(3) schedule(static, 1)
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      for (int k = 0; k < N; k++)
        C[i][j] += A[i][k] * B[k][j];   /* DEFECT 2: shared accumulator */
}

/* THE REAL FIX - collapse only the two INDEPENDENT loops, and accumulate into
 * a local. The k loop is a reduction and must stay inside one thread. */
void correct_collapse2(void) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      double s = 0;                     /* thread-local accumulator */
      for (int k = 0; k < N; k++)
        s += A[i][k] * B[k][j];
      C[i][j] = s;                      /* single write, one owner */
    }
  }
}

static void init(void) {
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      A[i][j] = (i + 1) * 0.01 + j * 0.001;
      B[i][j] = (j + 1) * 0.02 - i * 0.001;
    }
}

static void reference(void) {
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      double s = 0;
      for (int k = 0; k < N; k++) s += A[i][k] * B[k][j];
      REF[i][j] = s;
    }
}

static int wrong_entries(void) {
  int w = 0;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      if (fabs(C[i][j] - REF[i][j]) > 1e-9) w++;
  return w;
}

int main(void) {
  init();
  reference();
  for (int r = 0; r < 3; r++) {
    broken_collapse3();
    printf("collapse(3), hoisted init : %5d / %d entries wrong\n",
           wrong_entries(), N * N);
  }
  for (int r = 0; r < 3; r++) {
    correct_collapse2();
    printf("collapse(2), local sum    : %5d / %d entries wrong\n",
           wrong_entries(), N * N);
  }
  return 0;
}

/* ---------------------------------------------------------------------------
 * MEASURED OUTPUT (gcc 13.3.0, -O2, 16 logical CPUs):
 *
 *   collapse(3), hoisted init :    77 / 40000 entries wrong
 *   collapse(3), hoisted init :   143 / 40000 entries wrong
 *   collapse(3), hoisted init :   177 / 40000 entries wrong
 *   collapse(2), local sum    :     0 / 40000 entries wrong
 *   collapse(2), local sum    :     0 / 40000 entries wrong
 *   collapse(2), local sum    :     0 / 40000 entries wrong
 *
 * Note the broken version is wrong by a DIFFERENT amount every run - the
 * signature of a race rather than a systematic error.
 *
 * DEFECT 1 - NOT PERFECTLY NESTED (a compile error).
 *   "C[i][j] = 0;" sits between the j header and the k header. collapse(3)
 *   fuses three headers into one flat iteration space, and in a flat space
 *   there is no longer any point that runs "once per (i,j), before the k's".
 *   So the compiler refuses outright:
 *       error: not enough perfectly nested loops before 'C'
 *   This is slide 32's first caveat. It is purely STRUCTURAL - satisfying it
 *   tells you nothing about whether the code is correct.
 *
 * DEFECT 2 - THE k LOOP IS A REDUCTION (a data race).
 *   Hoisting the initialisation silences the compiler but does not fix
 *   anything. collapse(3) spreads all N*N*N triples (i,j,k) across threads, so
 *   several threads can be doing "C[i][j] += ..." on the SAME element at the
 *   same time. That is an unsynchronised read-modify-write - the same bug as
 *   0_code_of_the_day_data_race.cpp.
 *
 *   The i and j loops are genuinely independent: each (i,j) writes its own
 *   C[i][j]. The k loop is NOT - every k contributes to one accumulator. Only
 *   2 of the 3 loops were ever collapsible. This is slide 32's third caveat.
 *
 *   WHY IT CAN LOOK CORRECT: with the default schedule the fused space is cut
 *   into contiguous blocks, which usually keeps a whole k run inside one
 *   thread. Measured: 0/40000 wrong with the default schedule, but 77-177
 *   wrong with schedule(static,1), which interleaves the triples. A race that
 *   passes under one schedule and fails under another - do not trust a clean
 *   run.
 *
 * THE FIX: collapse(2), not collapse(3), plus a local accumulator.
 *
 *     #pragma omp parallel for collapse(2)
 *     for (i...) for (j...) {
 *       double s = 0;
 *       for (k...) s += A[i][k] * B[k][j];
 *       C[i][j] = s;
 *     }
 *
 *   Three things this buys:
 *     - Only independent loops are collapsed, so no two threads share a C[i][j].
 *     - The accumulator `s` is a local, hence automatically private per thread.
 *     - C[i][j] is written ONCE instead of N times, so the initialisation to 0
 *       disappears entirely and the perfect-nesting problem never arises.
 *   N*N iterations is ample parallelism (40,000 here for 16 threads); the k
 *   loop was never needed to fill the machine.
 *
 * ALTERNATIVE that keeps collapse(3): "#pragma omp atomic" on the update, or
 * reduction(+:C[:N][:N]). Both are CORRECT but far slower - see
 * 9_collapse_matrix_sum.cpp, where an atomic in the inner loop made the code
 * 2.9x slower at 16 threads than at 1. Do not serialise a reduction you can
 * simply privatise.
 *
 * THE GENERAL LESSON: collapse(k) is only safe when ALL k loops are mutually
 * independent. Count the independent levels first, then pass that number - do
 * not pass the total loop depth and hope.
 *
 * Verified on this machine with gcc 13.3.0.
 */
