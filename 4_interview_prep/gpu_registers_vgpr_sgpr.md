# GPU Registers: VGPRs & SGPRs (AMD) and Their NVIDIA Equivalents

A pre-interview refresher on how GPUs store per-thread vs. shared values in registers,
why AMD splits them into two files (VGPRs and SGPRs), what NVIDIA's equivalents are, how
they're allocated, and where the program counter actually lives.

---

## TL;DR (the 30-second version)

- A GPU runs threads in lockstep bundles: AMD calls them **wavefronts (64 work-items)**,
  NVIDIA calls them **warps (32 threads)**.
- Some values differ per thread (e.g. `a[i]`); some are the same for the whole bundle
  (e.g. a constant, a base address, a loop counter). GPUs store these two cases differently.
- **VGPR (Vector GPR)** = per-work-item register (one copy per lane). **SGPR (Scalar GPR)**
  = one value shared by the whole wavefront. AMD gives each its own register file and its
  own math pipeline (Vector ALU vs Scalar ALU).
- **NVIDIA equivalents:** VGPR ≈ ordinary per-thread registers (`R0`–`R255`); SGPR ≈
  **uniform registers** (`UR0`–`UR63`), which NVIDIA added in Turing (2018). Before Turing,
  NVIDIA had no separate scalar register file at all.
- **Registers are allocated per wavefront/warp, in coarse chunks, out of a fixed on-chip
  pool.** Using more registers per thread means fewer wavefronts fit at once → lower
  occupancy → worse latency hiding.
- **The program counter is NOT an SGPR / uniform register.** It's dedicated per-wavefront
  scheduler state. AMD lets you *copy* it into an SGPR pair; NVIDIA hides it entirely.

---

## 1. Background: what a register is, and why GPUs bundle threads

A **register** is the fastest, smallest storage in a processor, sitting right next to the
math units (ALUs). When hardware computes `c = a + b`, all three live in registers. A
register read/write costs about **1 cycle**, versus roughly **500 cycles** for global
memory — so registers are where all the actual computation happens.

GPUs do not run one thread at a time. They group threads and run them in lockstep under the
**SIMT** ("Single Instruction, Multiple Threads") model: one instruction is fetched, and
every thread in the bundle executes it on its own data.

- **AMD:** the bundle is a **wavefront** = **64 work-items** ("work-item" = AMD's word for
  a thread).
- **NVIDIA:** the bundle is a **warp** = **32 threads**.

This bundling is the whole reason the scalar/vector register distinction exists.

---

## 2. The core idea: two kinds of data → two kinds of registers

Inside a bundle, some values are **different for every thread**, and many are **identical
across all threads**. Consider this kernel:

```c
__global__ void vsadd(int y[], int a)
{
    int idx = threadIdx.x;
    y[idx] = y[idx] + a;
    if (y[idx] > THRESHOLD)
        y[idx] = Y_MAX_VALUE;
}
```

- `y[idx]` is **per-thread** data — thread 0 touches `y[0]`, thread 1 touches `y[1]`, and
  so on. This is **vector** data.
- `a`, `THRESHOLD`, `Y_MAX_VALUE`, and the base address `&y[0]` are **uniform** — the same
  for every thread in the bundle. This is **scalar** data.

Two useful formal terms:

- **Uniform variable:** same value for every thread. Can be stored once in a single scalar
  register and reused by all threads.
- **Affine variable:** a linear function of the thread ID (e.g. the address `&y[idx] =
  &y[0] + 4*idx`). Can be stored compactly as a `(base, stride)` pair instead of a full
  per-thread vector.

AMD builds **two separate register files and two separate math pipelines** to match:

| Register | Holds | One value per | Runs on |
|----------|-------|---------------|---------|
| **VGPR** (Vector GPR) | per-work-item data (`y[idx]`) | each of the 64 lanes | Vector ALU (VALU) |
| **SGPR** (Scalar GPR) | uniform data (`a`, addresses, loop counters) | the whole wavefront | Scalar ALU (SALU) |

**Analogy:** a teacher (scalar unit) with 64 students (lanes). The date on the whiteboard
is the same for everyone, so the teacher writes it once → that's an SGPR (one shared copy).
Each student's own answers differ → each has a personal answer sheet → that's a VGPR
(64 copies).

---

## 3. VGPRs and SGPRs, precisely (AMD CDNA)

**VGPRs — Vector General-Purpose Registers**

- Named `V0`–`V255`, each **32 bits** wide.
- Each VGPR physically exists **64 times** — once per lane. "Using 10 VGPRs" means every
  one of the 64 lanes gets its own 10 registers.
- CDNA adds a second vector pool, **Accumulation VGPRs** (`AV0`–`AV255`), used to hold the
  running results of matrix-multiply (MFMA / "Matrix Core") instructions. *(The
  regular + accumulation split is an AMD CDNA-specific detail.)*

**SGPRs — Scalar General-Purpose Registers**

- Named `S0`–`S103`, each **32 bits** wide.
- Only **one copy** per wavefront; all 64 work-items read the same value.
- The scalar register file also physically holds wavefront-wide special registers such as
  `VCC` (Vector Condition Code, the 64-bit "which lanes passed the compare" mask) and it
  works alongside the `EXEC` mask (the 64-bit "which lanes are active" mask). These masks
  are 64 bits precisely because a wavefront has 64 lanes.

**Seeing it in machine code** — AMD scalar instructions start with `s_`, vector with `v_`:

```
v_cmp_gt_f32   r0, r1            // vector: compare a>b, per work-item
s_mov_b64      s0, exec          // scalar: save the 64-bit exec mask into an SGPR pair
s_and_b64      exec, vcc, exec   // scalar: update EXEC to enter the "if"
s_cbranch_vccz label0           // scalar: branch controls the whole wavefront
v_mul_f32      r2, r0, r0        // vector: result = a*a, per work-item
```

---

## 4. Why two separate register files? (Three reasons)

1. **Save space, energy, and bandwidth.** If the uniform value `a` sat in a VGPR, the
   hardware would store 64 identical copies and 64 lanes would redundantly recompute the
   same thing. One SGPR stores it once and computes it once. Sharing operands across
   threads saves register-file and memory bandwidth and energy.

2. **Control flow needs a "whole-bundle" brain.** Loops, `if/else`, branch targets, and
   loop counters are decisions for the bundle as a unit. AMD runs *all* control flow on the
   scalar unit using SGPRs, keeping the expensive vector units free for real math.

3. **Addresses and kernel arguments are usually uniform.** Kernel args, base pointers, and
   workgroup IDs are the same for the whole wavefront, so they naturally live in SGPRs. At
   launch, the hardware pre-loads these into SGPRs and puts the per-work-item index into
   `VGPR0`.

Bonus (AMD-specific): the compiler even uses the scalar register file to **emulate the
divergence stack** that tracks which lanes are on/off during branches — work that would
otherwise need dedicated hardware.

---

## 5. The NVIDIA equivalents

### Vocabulary map

| AMD term | NVIDIA term |
|----------|-------------|
| Work-item | Thread |
| Wavefront = **64** work-items | Warp = **32** threads |
| Compute Unit (CU) | Streaming Multiprocessor (SM) |
| Local Data Share (LDS) | Shared memory |
| **VGPR** (vector register) | **Register** (per-thread `R0`–`R255`) |
| **SGPR** (scalar register) | **Uniform register** `UR0`–`UR63` *(Turing 2018+)* |

### VGPR ≈ NVIDIA's per-thread registers

When CUDA programmers say "this kernel uses 32 registers per thread," those per-thread
registers (`R0`–`R255` in NVIDIA's SASS assembly) are the direct analog of AMD's VGPRs:
private to each thread, drawn from a large per-SM register file (**64K 32-bit registers per
SM**, max **255 registers per thread**).

### SGPR ≈ NVIDIA's uniform registers — with a historical caveat

- **Historically (through Volta), NVIDIA had no separate scalar register file.** There was
  only the per-thread register file. Uniform values were kept in ordinary per-thread
  registers (replicated across lanes) or read from **constant memory**. The closest thing
  to scalar registers were **predication registers** used for control flow.
- **From Turing (2018) onward, NVIDIA added the true analog: a "uniform datapath" plus
  "uniform registers" (`UR0`–`UR63`).** These are shared once per warp (not per thread),
  live in a small dedicated register file with their own ALU, and hold exactly the
  AMD-SGPR kind of thing: loop counters, array offsets, pointer math, broadcast constants,
  block index. The compiler (`ptxas`) decides what goes there automatically; it is
  invisible in PTX and appears only in SASS. Newer architectures expanded the count
  (e.g. 256 uniform registers on Blackwell-class parts).

### Philosophical difference to remember

AMD **exposes a full scalar ISA**: separate `s_` instructions, a documented scalar register
file, and a scalar unit that also runs control flow. NVIDIA's uniform datapath is
**opportunistic and compiler-hidden** (mostly integer/address work, auto-detected, no
programmer control, not visible in PTX). Same underlying idea — "don't redo warp-uniform
work 32 times" — but very different amounts of programmer/ISA visibility.

---

## 6. How registers are allocated, and at what granularity

Golden rule (both vendors): **registers are allocated per wavefront/warp — not per
individual thread — out of a fixed on-chip pool, in coarse chunks. This is what limits how
many wavefronts/warps can run at once (occupancy).**

### AMD (CDNA)

- **SGPRs:** allocated per wavefront, from **16 up to ~102**, always **in units of 16**
  dwords (a "dword" = 32 bits = 4 bytes). The scalar file also holds `VCC` and, with a trap
  handler, extra reserved SGPRs.
- **VGPRs:** allocated per wavefront (each lane gets its own copy), in **groups of 8**
  dwords. Up to **512 total** per wavefront, split as up to **256 regular + 256
  accumulation**. 64-bit values must start on an even-numbered VGPR.

So the allocation granularity is **16 (SGPR)** and **8 (VGPR)** dwords: ask for 17 SGPRs and
you get 32; ask for 9 VGPRs and you get 16.

### NVIDIA

- Register file: **64K (65,536) 32-bit registers per SM**; max **255 per thread**.
- **Register Allocation Unit Size = 256** registers per warp → equivalently, per-thread
  counts round up to a **multiple of 8** (256 ÷ 32 threads).
- **Register Allocation Granularity = warp** (allocated per warp, like AMD's per-wavefront).
- **Warp Allocation Granularity = 4** (warps handed out 4 at a time on modern archs; 2 on
  some older ones).
- Uniform registers count *within* the same total register budget.

### Why granularity matters: occupancy

The register file is a fixed pool shared by all resident wavefronts/warps. **More registers
per thread → fewer bundles fit → less ability to hide long memory latencies** (the GPU
hides a stalled bundle by switching to a ready one).

**Worked NVIDIA example:** a kernel with 672 threads/block and 48 registers/thread on
compute capability 8.6 → 21 warps, **rounded up to 24** (granularity 4) → 24 × 32 × 48 =
36,864 registers, which is more than half of 65,536, so **only 1 block** fits per SM instead
of 2. That rounding is exactly why granularity is worth understanding.

---

## 7. Where does the program counter live? (Is it an SGPR?)

Short answer: **No — the live program counter is not an SGPR or a uniform register on either
vendor.** It is dedicated per-wavefront/per-warp state owned by the scheduler.

### AMD

- The PC is a distinct **48-bit** per-wavefront register, separate from `S0`–`S103`. It is
  initialized when the wavefront is created.
- Software **cannot directly read or write** it. It can only be **copied to/from an
  even-aligned SGPR pair** using `S_GETPC` / `S_SETPC` / `S_SWAPPC`. (A *pair* is needed
  because the PC is 48-bit but an SGPR is only 32-bit, and 64-bit values require even
  alignment.)
- SGPRs do routinely hold **control-flow-adjacent** data: branch targets, loop counters,
  and **saved return addresses** (e.g. `S_CALL_B64` stashes `PC+4` into an SGPR pair). So
  SGPRs are the PC's scratch/backup area — but the live PC is its own register.

### NVIDIA

- The warp's PC is maintained by the **hardware warp scheduler** and is **not part of the
  register file at all** — not the per-thread registers (`R`), not the uniform registers
  (`UR`). It is not even an addressable operand. This is *more* hidden than AMD, which at
  least lets you copy the PC into SGPRs.

### Refining "one PC per warp"

- **Baseline model (pre-Volta NVIDIA, and AMD's normal wavefront execution):** effectively
  **one PC per bundle**, plus a small **divergence stack** ("SIMT stack") to sequence
  branches. Each stack entry holds a next-PC, a reconvergence-PC, and an active mask that
  says which lanes run. On AMD this is managed with the `EXEC` mask and SGPRs (which emulate
  the stack), plus a tiny hardware branch-stack pointer (`CSP`).
- **NVIDIA Volta and later (Independent Thread Scheduling):** adds **per-thread PC state**
  (a per-thread resume-PC) plus convergence barriers, so it is no longer strictly one PC
  per warp — but that state still lives in dedicated scheduler registers, not in the
  general-purpose or uniform register files.

### Quick "where does the PC live" table

| | Where the live PC lives | Is it an SGPR / uniform reg? | Can software touch it? |
|---|---|---|---|
| **AMD (CDNA)** | Dedicated 48-bit per-wavefront PC | No (separate from `S0`–`S103`) | Only by copying to/from an even-aligned SGPR pair |
| **NVIDIA (pre-Volta)** | 1 warp PC in the scheduler (+ SIMT stack) | No (not in the register file) | No — opaque scheduler state |
| **NVIDIA (Volta+)** | Per-thread resume-PC in scheduler registers | No | No |

---

## 8. AMD vs NVIDIA at a glance

| Aspect | AMD (CDNA) | NVIDIA |
|--------|------------|--------|
| Thread bundle | Wavefront = 64 work-items | Warp = 32 threads |
| Per-thread registers | VGPRs `V0`–`V255` (+ `AV0`–`AV255` accumulation) | Registers `R0`–`R255` |
| Scalar/uniform registers | SGPRs `S0`–`S103` (first-class, in the ISA) | Uniform registers `UR0`–`UR63` (Turing+; compiler-hidden) |
| Scalar pipeline | Dedicated scalar unit runs `s_` instructions + all control flow | Uniform datapath (opportunistic, mostly integer/address) |
| Register file size | Per-SIMD/CU files; per-wavefront limits (≤512 VGPR, ≤~102 SGPR) | 64K 32-bit regs/SM, ≤255/thread |
| Allocation granularity | SGPR: 16 dwords; VGPR: 8 dwords; per wavefront | 256 regs/warp unit (multiples of 8/thread); warps in groups of 4 |
| Program counter | Dedicated 48-bit reg; copyable into SGPR pair | Scheduler-only; not in the register file |

---

## 9. Common interview questions to prepare

1. **What is a warp / wavefront, and how big is each?** (Warp = 32 threads on NVIDIA;
   wavefront = 64 work-items on AMD CDNA.)
2. **What's the difference between a scalar register and a vector register on a GPU?**
   (Scalar = one value shared by the whole bundle; vector = one value per thread/lane.)
3. **Why does AMD have separate scalar and vector register files, but NVIDIA historically
   didn't?** (Efficiency: store/compute uniform values once; run control flow on a cheap
   scalar unit. NVIDIA used the unified per-thread file + constant memory, then added
   uniform registers in Turing.)
4. **What is the NVIDIA equivalent of an AMD SGPR?** (Uniform registers `UR0`–`UR63`,
   introduced in Turing; before that, no dedicated scalar register file.)
5. **What are uniform and affine variables, and why do they matter? / What is
   scalarization?** (Values that are constant or linear in thread ID; storing/computing them
   once saves registers, bandwidth, and energy.)
6. **How are GPU registers allocated, and at what granularity?** (Per warp/wavefront, in
   fixed chunks, from a fixed pool. AMD: 16-dword SGPR / 8-dword VGPR units. NVIDIA:
   256-reg-per-warp unit, warps in groups of 4.)
7. **What is occupancy, and how do registers affect it?** (Ratio of active warps to the max;
   more registers per thread → fewer resident warps → less latency hiding.)
8. **What happens if a kernel uses too many registers?** (Lower occupancy; and beyond the
   per-thread limit, the compiler spills registers to (slow) local memory.)
9. **Is there one program counter per thread or per warp? Where does it live?** (Baseline:
   one PC per bundle + a divergence stack. Volta+ ITS: per-thread resume-PC. It lives in
   scheduler state, not in the register file — though AMD can copy it into an SGPR pair.)
10. **What is the SIMT stack / how is branch divergence handled?** (A stack of
    (reconvergence-PC, next-PC, active-mask) entries that serializes divergent paths and
    reconverges lanes.)
11. **What are the EXEC mask and VCC on AMD?** (64-bit per-lane masks: which lanes execute,
    and which passed the last vector compare; both associated with the scalar register
    file.)
12. **Why is the GPU register read latency ~1 cycle but global memory ~500 cycles, and how
    do GPUs hide that gap?** (On-chip registers vs off-chip DRAM; hidden by switching
    between many resident warps — hence the value of high occupancy.)

---

## References

- **AMD Instinct MI300 CDNA3 Instruction Set Architecture** (reference guide,
  5-Aug-2025), `reading_material/AMD_specific/amd-instinct-mi300-cdna3-instruction-set-architecture (1).pdf`:
  - Table 1 (Terminology), p. 4 — wavefront = 64 work-items; SALU = one value per
    wavefront; VALU = unique per work-item.
  - Chapter 2 (Program Organization), pp. 4–5 — scalar vs vector ALU/memory.
  - Table 2 (State Overview), p. 8 — `V0`–`V255`, `AV0`–`AV255`, `S0`–`S103`, `PC` (48-bit),
    `EXEC` (64-bit), `VCC` (64-bit).
  - §3.2 (Program Counter), p. 9; §3.5 (Mode register: `CSP`), p. 10.
  - §3.6.2 (SGPR allocation: 16–102 in units of 16), §3.6.3 (even alignment, incl. PC),
    §3.6.4 (VGPR allocation: groups of 8; up to 512 = 256 regular + 256 accumulation),
    p. 12.
  - §3.9 (VCC resides in the SGPR file), p. 13; §3.13 (GPR initialization), p. 17;
    Chapter 4 (`S_CALL_B64`, control flow via SGPRs), p. 18.
- **Aamodt, Fung & Rogers, _General-Purpose Graphics Processor Architectures_** (Morgan &
  Claypool, 2018), `reading_material/synthLect-gpgpus.pdf`:
  - §2.2.1–2.2.2, pp. 14–20 — NVIDIA PTX/SASS registers; AMD GCN separate scalar/vector
    instructions and registers; "single 32-bit value shared by all threads in a wavefront";
    key difference vs NVIDIA (incl. Volta).
  - §3.1, p. 22; §3.1.1, p. 25 — warp PC used by the scheduler; SIMT stack entries
    (RPC, next-PC, active mask).
  - §3.1.2, pp. 27–29 — NVIDIA Volta Independent Thread Scheduling; per-thread resume-PC
    stored in hardware warp-scheduler registers.
  - §3.4.4, pp. 54–55 — AMD scalar register file emulates the SIMT stack; NVIDIA predication
    registers as "scalar registers dedicated to control flow".
  - §3.5, pp. 57–61 — uniform vs affine variables; scalar register file; scalarization.
- **Hwu, Kirk & El Hajj, _Programming Massively Parallel Processors_ (PMPP), 4th ed., 2022**
  — `reading_material/CUDA_Course/pdf/Chapter 04 - Compute Architecture and Scheduling.pdf`
  (SM structure, warp = 32, SIMD, 64K registers/SM, occupancy) and `Chapter 05 - Memory
  Architecture and Data Locality.pdf` (registers as per-thread storage, ~1-cycle vs
  ~500-cycle latency, occupancy constrained by registers/thread).
- **NVIDIA "Dissecting the Turing T4 GPU with Microbenchmarks", GTC 2019** — uniform
  datapath and uniform registers `UR0`–`UR62` (+`URZ`); 256 total register limit including
  uniform:
  https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9839-discovering-the-turing-t4-gpu-architecture-with-microbenchmarks.pdf
- **NVIDIA Turing architecture slides, Hot Chips 2019** — "New Uniform registers and
  datapath", automatic promotion of warp-uniform ops/data ("reverse vectorization"):
  https://old.hotchips.org/hc31/HC31_2.12_NVIDIA_final.pdf
- **NVIDIA Ampere GPU Architecture Tuning Guide** — 64K 32-bit registers per SM; max 255
  registers per thread; occupancy:
  https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
- **NVIDIA Developer Forums (Nsight Compute occupancy values)** — Register Allocation Unit
  Size = 256, Register Allocation Granularity = warp, Warp Allocation Granularity = 4 (2 on
  older archs):
  https://forums.developer.nvidia.com/t/what-is-warp-allocation-granulatity-for/365269/8 and
  https://forums.developer.nvidia.com/t/why-is-my-register-count-limiting-the-active-thread-blocks-per-sm/324078
- **"SASS King, Part 1: Reading NVIDIA SASS from First Principles"** (summary of Jia et al.,
  "Dissecting the NVIDIA Turing GPU via Microbenchmarking") — uniform registers are one copy
  per warp in a dedicated SRAM/datapath; compiler-driven; expanded on newer architectures:
  https://florianmattana.com/posts/sass_king/
