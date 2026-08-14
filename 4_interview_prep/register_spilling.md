# Register Spilling, the 255-Register Cap, and Occupancy

A pre-interview refresher on what happens when a CUDA kernel demands too many registers:
when the excess **spills** to local memory (and still runs), versus when compilation
**fails outright**. It resolves a common point of confusion — spilling below the cap vs.
the hard per-thread limit — and ties the whole thing back to occupancy.

---

## TL;DR (the 30-second version)

- Every variable a kernel uses burns a register at some point. What matters is the **max
  number of registers simultaneously live** during the kernel's execution.
- There are **two different limits**, and they trigger different outcomes:
  - **Soft limit (a register budget):** exceed it → **spill** excess values to local
    memory. Kernel still compiles and runs, just slower.
  - **Hard limit (255 registers/thread):** exceed it → **compilation fails**. There is
    nowhere left to spill the *addressing*; it's an encoding limit.
- "Local memory" is a misnomer: it physically lives in **off-chip global (device) memory**,
  so spilling is slow (high latency, ~hundreds of cycles).
- Your intuition "occupancy should just auto-decrease to free registers" is **correct — up
  to a point**. Occupancy steps down block-by-block, but it bottoms out at **one resident
  block**. Below that floor, the only relief valve left is spilling.
- **Three spill triggers, all compile-time:** (1) you capped registers with `-maxrregcount`
  or `__launch_bounds__`; (2) the compiler heuristically elects to spill a little to keep
  occupancy high; (3) an array/struct that simply can't live in registers. The **hardware
  never spills** — if a launch config needs more registers than the SM has and you gave no
  cap, the **launch fails** ("too many resources requested for launch"), it does not spill.
- **Compile-time vs runtime split:** *registers per thread*, and *whether/how much to spill*
  (the `STL`/`LDL` instructions), are **frozen at compile time** by `ptxas` and baked into
  the binary. *Occupancy* (how many blocks co-reside on an SM) is resolved **at launch**.

---

## 1. What "using a register" actually means

A **register** is the fastest storage on the chip, sitting next to the ALUs — a read/write
costs ~1 cycle versus ~hundreds of cycles for global memory. Any variable you touch in a
kernel needs to "live" somewhere, and for it to be operated on it occupies a register at
that moment.

- `int a = 1;` — `a` lives in a register.
- Even a value stored in shared memory eats a register at the moment it's operated on.
- As execution advances, the count of occupied registers rises and falls. **The number
  that matters is the maximum registers required at any single point** ("high register
  pressure" = that maximum is large).

Who decides the allocation? **The compiler** — you don't control it directly (though you
can constrain it; see §5).

---

## 2. The two limits — and why they seem contradictory

Two facts that look like they conflict but don't:

| Source statement | Outcome |
|---|---|
| "Too many registers needed → spill to local memory (runtime handles it)" | Runs, slower |
| "A thread can be assigned up to 255 registers — anything beyond causes compilation to fail" | Doesn't build |

They describe **two separate limits**:

1. **Register spilling** happens against a *budget* that is usually **below** 255. When the
   kernel wants more than the current budget, excess automatic variables are placed in
   local memory. No failure.
2. **The 255-per-thread cap** is the **architectural encoding limit**. The hardware / PTX
   cannot address more than 255 registers for a single thread, so if a thread genuinely
   needs more than 255 *simultaneously live* values, the compiler errors out.

**Mental model:** soft limit → spill (pressure-relief valve); hard limit → fail (ceiling).
Spilling happens *below* 255; failure happens *at* 255.

---

## 3. What actually gets placed in local memory

The compiler places an automatic variable in local memory in these cases:

- **Arrays it can't prove are indexed with constant quantities** (can't keep them in
  registers if the index isn't known at compile time).
- **Large structures or arrays** that would consume too much register space.
- **Any variable, if the kernel uses more registers than available** — this is exactly what
  "register spilling" means.

Local memory is thread-private but **physically resides in device (global) memory**, so
every spilled access pays global-memory latency and bandwidth. That's why spilling hurts
performance even though the kernel still works.

---

## 4. Why occupancy doesn't just "handle it" — the one-block floor

The natural question: *if a kernel wants more registers, why not just let occupancy drop so
each thread gets a bigger share? Why spill at all below 255?*

Because **occupancy has a hard floor: one resident block must fit, or the kernel won't
launch.** The relationship is:

```
registers_per_thread  ≤  register_file_per_SM / (threads that must be resident)
```

The denominator can't shrink past your **block size** — one block has to fit. That sets the
real ceiling on registers-per-thread.

### Worked example (64K = 65,536 registers per SM)

| Block size | Regs/thread ceiling to fit ONE block | Kernel wants 100 regs/thread → |
|---|---|---|
| 256 threads  | 65,536 / 256  = **256** (≈ the 255 cap) | fits, no spill |
| 512 threads  | 65,536 / 512  = **128** | fits, no spill |
| 1024 threads | 65,536 / 1024 = **64**  | occupancy already at floor → **must spill to 64** |

With a **1024-thread block**, the moment the kernel wants more than **64** registers,
occupancy is *already* as low as it can go (one block). It can't drop further to hand you
more registers — so the compiler spills. That's spilling at 64, far below 255.

So: occupancy auto-decrease is the **first** lever (trade resident warps for registers), but
it bottoms out at one block. Below that floor there are only two outcomes — **spill** (if the
compiler was told, at compile time, to keep registers low) or **launch failure** (if it
wasn't). See §5a for why the hardware itself never spills. Compilation failure at 255 is a
separate, third wall.

---

## 5. The three spill triggers (all decided at compile time)

An earlier, sloppier framing called one trigger "hardware-forced." That's wrong and worth
un-learning: **the hardware never spills.** A spill is real instructions (`STL`/`LDL`, §5a)
that only the compiler can emit. Every genuine spill trigger is therefore a *compile-time*
decision:

| # | Who decides | Spill is… | Mechanism |
|---|---|---|---|
| 1 | **You** — `-maxrregcount N` (per file) or `__launch_bounds__(...)` (per kernel) | forced | You tell the compiler "stay under N regs"; excess spills to honor it. Deliberately buying occupancy with spill cost. `__launch_bounds__` also lets the compiler pre-emptively spill for a *runtime* block size, because you gave it the block size at compile time. |
| 2 | **Compiler** — register-allocator heuristic | chosen | Even with no cap, it may spill a few registers because high occupancy (better latency hiding) is worth more than a couple of local-memory accesses. |
| 3 | **Compiler** — value can't live in a register at all | forced | Arrays it can't prove are constant-indexed, or large structs/arrays — placed in local memory regardless of pressure (see §3). |

**What is NOT a spill trigger:** you launch with no cap, and at runtime the requested
`block_size` needs more registers than the SM's file can supply even for one block. The
hardware does **not** rescue this by spilling — the **launch fails** with
`cudaErrorLaunchOutOfResources` (code 701; classic message *"too many resources requested for
launch"*). Spilling only happens if the compiler was told, at compile time, to keep registers
under a budget.

The compiler's stated goal is to **minimize register usage while keeping both register
spilling and instruction count to a minimum** — a balancing act, not a single objective.

---

## 5a. Compile-time vs runtime: what's frozen and what's resolved at launch

This is the crux that ties the whole topic together. Two *different* decisions happen at two
*different* times:

| Decision | What it fixes | When | Who |
|---|---|---|---|
| **Register allocation** | *How many* registers **each thread** uses (regs/thread) | **Compile time** | Compiler (`ptxas`) |
| **Spill decision** | *Whether* and *how many bytes* to spill — the `STL`/`LDL` instructions | **Compile time** | Compiler (`ptxas`) |
| **Register partitioning** | *How* the SM's 64K register file is split among resident blocks → **occupancy** | **Runtime (launch)** | Hardware scheduler |

**Registers per thread is a compile-time constant.** The compiler fixes it and bakes it into
the SASS/cubin; it does not change launch to launch. "Dynamically partitioned" (deck p.38)
does **not** mean regs/thread varies at runtime — it means the *register file* is carved up
among however many blocks happen to co-reside. Inspect the constant with:

```
nvcc -Xptxas -v ...
→ "Used 72 registers, 24 bytes spill stores, 24 bytes spill loads, ..."
```

**A spill is instructions, so it's compile-time too.** To spill, the compiler emits a spill
**store** (`STL` — store to local) and later a spill **load** (`LDL` — load from local).
Those instructions sit in the binary; the GPU just runs them. It can neither invent nor
remove a spill at runtime. So *whether to spill* and *how much* (those "spill stores/loads
bytes" above) are **compile-time constants**, exactly like regs/thread.

**Registers per block is a derived quantity, not always constant:**

```
registers_per_block = registers_per_thread (compile-time)  ×  threads_per_block
```

`threads_per_block` is a **launch parameter** (`kernel<<<grid, block>>>`). If you compute the
block size at runtime, registers-per-block is only known at launch. If you pin the block size
(a literal, or via `__launch_bounds__`), it's effectively fixed — but because *you* pinned
it, not the compiler.

**Registers per SM / occupancy is always a runtime outcome:**

```
blocks_resident = floor( register_file_per_SM / (registers_per_thread × threads_per_block) )
```

The inputs (regs/thread) are compile-time; the result (how many blocks pack in, and whether
even one fits) is resolved at launch. That's the "dynamic partitioning" of deck p.38, and the
same one-block-floor math from §4.

**Summary of the split:**
- Compile-time & baked into the binary: registers per thread; whether/how much to spill.
- Runtime & resolved at launch: occupancy (blocks per SM); whether the launch succeeds at all.

---

## 6. Occupancy ≠ performance (but it's a good proxy)

Worth knowing for the follow-up interview question:

- Higher occupancy does **not** guarantee higher performance, but it's a decent proxy.
- **Low-occupancy** SMs struggle to hide latency on **memory-bound** kernels — with few
  resident warps, there's nothing to switch to while waiting on memory.
- This is *why* the heuristic trigger (§5 #2) exists: sometimes "spill a little, stay at high
  occupancy" beats "no spill, collapse to low occupancy," especially for memory-bound work.
- Counterpoint: with high exposed **instruction-level parallelism (ILP)**, a *low*-occupancy
  kernel can sometimes fully hide latency anyway — so more registers per thread (lower
  occupancy, no spill) can win. It's a genuine tradeoff, not a rule.

---

## 7. The clean summary to recite in an interview

1. Register pressure = max registers simultaneously live in a kernel.
2. Occupancy steps down to free registers per thread — **but only until one block fits**.
3. Past that floor (or an explicit `-maxrregcount` / `__launch_bounds__` cap, or a compiler
   heuristic choice), excess values **spill to local memory** — off-chip, slow, but the
   kernel runs.
4. Past **255 live registers per thread**, there's nothing left to spill the addressing into
   → **compilation fails**.
5. Soft limit → spill. Hard limit → fail. Spilling lives *below* 255; failure is *at* 255.

---

## 8. Common interview questions on this topic

- **Q: What is register spilling and where do spilled registers go?**
  Excess per-thread values the compiler can't keep in registers are placed in *local
  memory*, which physically resides in off-chip global/device memory — so spills are slow.

- **Q: What's the maximum number of registers per thread, and what happens if you exceed it?**
  255 (32-bit registers per thread) on all current NVIDIA compute capabilities. Exceeding it
  is a hard encoding limit → **compilation fails**, not a spill.

- **Q: If occupancy can decrease to give more registers per thread, why does spilling ever
  happen below 255?**
  Because occupancy bottoms out at one resident block. For large blocks the ceiling
  `register_file / block_size` can be far below 255 (e.g. 1024 threads → 64 regs on a 64K-reg
  SM). Beyond that ceiling the only option left is to spill.

- **Q: How can you control register usage?**
  `-maxrregcount N` (per compilation unit) or `__launch_bounds__(maxThreadsPerBlock,
  minBlocksPerSM)` (per kernel). Both cap registers, trading potential spills for guaranteed
  occupancy.

- **Q: Does more occupancy always mean more performance?**
  No — it's a proxy. Low occupancy hurts latency hiding on memory-bound kernels, but a
  compute-bound kernel with high ILP can perform well at low occupancy. Spilling to preserve
  occupancy is only worth it when latency hiding is the bottleneck.

- **Q: Why is it called "local" memory if it's slow?**
  "Local" refers to *scope* (private to one thread), not physical locality. It lives in
  global memory, so it is not fast like registers or shared memory.

- **Q: What does the compiler optimize for when allocating registers?**
  It minimizes register usage while keeping register spilling *and* instruction count low —
  a balance, since aggressively cutting registers forces more spills.

- **Q: Is registers-per-thread a compile-time constant?**
  Yes. The compiler (`ptxas`) fixes it and bakes it into the SASS/cubin; it doesn't vary per
  launch. See it with `nvcc -Xptxas -v`. Registers-*per-block* = regs/thread × block_size, so
  it's only constant if the block size is a compile-time constant; occupancy (blocks per SM)
  is a runtime outcome.

- **Q: Is the decision to spill (and how much) compile-time or runtime?**
  Compile-time. A spill is real instructions — `STL` (store to local) and `LDL` (load from
  local) — emitted into the binary. The GPU just executes them; it can't add or drop a spill
  at runtime. `nvcc -Xptxas -v` even prints the constant "spill stores/loads" byte counts.

- **Q: If a kernel needs more registers than the SM has and you gave no `-maxrregcount` /
  `__launch_bounds__`, does the hardware spill to save the launch?**
  No. The hardware never spills. The **launch fails** with *"too many resources requested for
  launch."* Spilling only happens if the compiler was told at compile time to keep registers
  under a budget (or chose to for occupancy).

- **Q: The deck says registers are "dynamically partitioned" — doesn't that contradict
  regs/thread being fixed?**
  No. "Dynamically partitioned" refers to how the SM's register file is split among whatever
  blocks co-reside at runtime — that's occupancy. The per-thread register count each block
  uses was already frozen at compile time. Two different decisions, two different times.

---

## References

- NVIDIA, *CUDA C++ Programming Guide* — "Device Memory Accesses → Local Memory" (list of
  automatic variables placed in local memory; defines register spilling as "any variable if
  the kernel uses more registers than available"):
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses
- NVIDIA, *CUDA C++ Programming Guide* — "Compute Capabilities → Technical Specifications per
  Compute Capability" (table row: **Maximum number of 32-bit registers per thread = 255**;
  register file = 64K 32-bit registers per SM):
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
- NVIDIA, *CUDA C++ Best Practices Guide* — §10.2.7.1 "Register Pressure" ("Register pressure
  occurs when there are not enough registers available for a given task"; occupancy vs.
  register spilling to local memory; `-maxrregcount` and `__launch_bounds__`):
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#register-pressure
- Course deck: `reading_material/UW_ece759/17 kernel-scheduling.pdf`, p. 38 — "A thread can
  have assigned by the compiler up to 255 registers – anything beyond will cause the
  compilation to fail"; also "Registers are dynamically partitioned across all Blocks and
  assigned to the SM" (the runtime partitioning / occupancy step); p. 37 — what "register"
  means in CUDA and "max number of registers required during the execution of the kernel";
  p. 40 — "Who controls register allocation? → The compiler" (the compile-time allocation
  step).
- NVIDIA, *CUDA C++ Best Practices Guide* — verbose compiler reporting: "`--ptxas-options=-v`
  or `-Xptxas=-v` lists per-kernel register, shared, and constant memory usage," and "the
  compiler reports total local memory usage per kernel (**lmem**) when run with the
  `--ptxas-options=-v` option." (Verified 2026-07-26. Note: the flag reports *lmem* / register
  counts; the finer "N bytes spill stores / N bytes spill loads" breakdown is emitted by
  `ptxas` in practice but the exact phrase is not quoted on this page.)
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#register-pressure
- NVIDIA, *CUDA Binary Utilities* — SASS instruction set table: "**STL** — Store to Local
  Memory" and "**LDL** — Load within Local Memory Window" (the store/load-to-local
  instructions the compiler emits to spill). Verified 2026-07-26.
  https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
- NVIDIA, *CUDA Runtime API* — `cudaErrorLaunchOutOfResources = 701`: "a launch did not occur
  because it did not have appropriate resources ... usually indicates ... the kernel launch
  specifies too many threads for the kernel's register count." This is the error when a
  launch config's register demand can't fit the SM — the hardware does not spill to rescue
  it. (The classic runtime message string is "too many resources requested for launch.")
  Verified 2026-07-26.
  https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
- Course deck: `reading_material/UW_ece759/18 shared-memory-and-thread-synchronization.pdf`,
  p. 12 — "Register spill, if too many registers are needed ... (high register pressure)";
  p. 18 — registers "can spill into local memory and global memory (operated by runtime)".
- Course deck: `reading_material/UW_ece759/20 occupancy.pdf`, p. 26 — register file size per
  SM (64K 4-byte registers on Kepler/Maxwell/Pascal/Volta/Ampere) as an occupancy limiter;
  p. 31 — "Occupancy != Performance [yet a pretty good proxy]"; low-occupancy SMs struggle to
  hide latency on memory-bound kernels.
