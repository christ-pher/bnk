# Shared performance work, DeepSeek first (branch `perf-shared`, 2026-11-25)

The ideas list in `docs/DEEPSEEK-NEXT.md` ("Ideas list (2026-11-25)"), ordered so each step lands in code the
engines share where it can: the expert cache (`src/core/expert_cache`) and the CPU expert pool (`src/cpu`) serve
both MoE engines (DeepSeek and the Qwen engine: Flash-Next IQ3_S, CYBER-FROST); the dense engine (27B) has its own
tensor-parallel path.

| Step | What | DeepSeek | Other models |
| --- | --- | --- | --- |
| 1 | `tools/gpu_busy.py`: each GPU's busy and memory share in a request's prompt read and decode (NVML samples) | what is idle | every engine |
| 2 | Warm start: the resident experts saved beside the counts file, loaded first | restart 20 -> ~30 tok/s (to verify) | Qwen MoE engine |
| 3 | CPU pool: SMT siblings without spinning; faster one-token IQ2 dots | ~20 ms a token on the CPU | the same pool |
| 4 | Two DeepSeek slots in two lanes (`forward_pipe`) | two agents at once | (Qwen has it) |
| 5 | Disk parking for the Qwen engines; parking for the dense engine | done in v0.6.1 | Qwen MoE, 27B |
| 6 | DSpark's experts in VRAM, on the idle card | cheaper draft rounds | |
| 7 | 27B exchanges in fp16 / peer stores | | 27B |
| 8 | GPU 1 on a prompt's skipped groups (`PREFILL-SKIP-PLAN.md` 2b) | GPU 1 idles in prompt reads | |

## Step 1: measured (2026-11-25, production ds-xs server, master before v0.6.1)

`tools/gpu_busy.py --prompt-chars 20000 --max-tokens 300`: 6,475-token prompt in 5.5 s, 300 tokens at 28.2 tok/s.

| Phase | GPU 0 busy | GPU 0 memory | GPU 1 busy | GPU 1 memory |
| --- | --- | --- | --- | --- |
| Prompt read | 84% | 24% | **12%** | 1.4% |
| Decode | **40%** | 6.4% | **40%** | 9.8% |

- In a 6K prompt read GPU 1 is nearly idle: the long-prompt skip leaves the second stage only the tail group. So step
  8 matters at ordinary agent sizes, not only at 70K.
- In decode each card is idle ~60% of the time (busy counts any kernel, so a kernel waiting on the CPU's answer
  counts as busy: 40% is an upper bound). Memory traffic is low: the cards wait on the CPU experts and on each other.

## Step 2: warm start

`<counts>.resident` (BNKR v1: n_layer, n_expert, then (layer, expert) pairs hottest first by the adaptive score) is
written with the counts (every 60 s while serving, and at quit). At load, `warm_ranking` puts it ahead of the counts'
ranking (`ExpertCache::init` drops repeats and completes the rest), so the cache starts where the last run left it.
`BNK_WARM_START=0` turns it off. Both MoE engines. Test: `build/test_expert_counts`.

Measured (2026-11-25, ds-xs, both GPUs, DSpark on): the cache adapted over four code requests (30.2 -> 37.8 tok/s), the
server stopped (the resident file: 4,296 experts, as resident), then restarted with the same files twice. The same
seeded 400-token request three times after each start:

| | request 1 | 2 | 3 |
| --- | --- | --- | --- |
| before the restart | 34.2 | 38.2 | |
| restart, `BNK_WARM_START=0` (master) | 26.4 | 32.1 | 33.4 |
| restart, warm start | **31.3** | **38.1** | **36.4** |

The first request after a start is 18% faster and the next ones are at the pre-restart speed; without it the cache
needs several requests to adapt back. (Run-to-run spread of one request is ~10%: 34.2 vs 38.2 on the same seed.)

## Step 3: the CPU pool (measured 2026-11-25, `bench_ds_cpu` layer 10, 3 experts a call, pinned, 8 chunks a thread)

The production server was up but idle (~2 cores busy) during these runs.

| Threads | T=1 decode, 3 experts | T=4 verify |
| --- | --- | --- |
| 24 (default) | 0.676-0.690 ms (3 runs: stable) | 1.16 ms |
| 32 | 0.630 | 1.12 |
| 36 | 0.615 / 0.690 / **1.509** | |
| 40 | 0.608 / 0.804 / 0.620 / 0.624 | 1.12 |
| 44 | **9.38** | 1.08 |
| 48 | **1.48** | 1.96 |

- The SMT siblings buy ~10% at best (0.68 -> 0.61 ms) and make the pool unstable: a spinning barrier with more
  workers than free CPUs waits out a preempted worker's scheduler slice (44 threads: 14x slower). Not shippable as
  is. It would need a barrier that survives preemption (spin briefly, then futex), and the server's own threads
  (Python, GPU service, Engram) leave fewer free CPUs than the bench sees.
- One-token kernels: ggml's `vec_dot` 0.676-0.683 ms, bnk's `mdot` from one token (`BNK_CPU_MDOT1=1`) 0.713-0.764,
  the signed-grid table (`BNK_CPU_XXS_CT=1`) 0.804-0.848. ggml's stays fastest. The pool reaches 43 GB/s of weights
  here (~40% of the RAM's bandwidth) at roughly 33 cycles a 64-weight step: load-port bound by the grid and sign
  table lookups (~30 loads a step), not by arithmetic or memory. A faster kernel needs fewer lookups per weight (a
  new decode scheme), not tuning. Left open; no quick win.

## Step 4: two DeepSeek slots in two lanes

`--slots 2` failed at once: "commit: the slot was not in the last forward", raised inside a commit graph's capture
(the GPU context was lost). In two lanes (`forward_pipe`) GPU 0 runs lane A's batch while GPU 1 runs lane B's, so when
a slot commits, a stage's last forward can be the other lane's: the commit looked its window up in that forward's
`wins_`, and its kernel read the window's position from `d_par_`, which the other lane's forward had overwritten.
Fixed: the window parameters (`h_par_` / `d_par_`) and the window list are kept per lane (`lane_wins_`), and the
commit reads its own lane's. Also: the commit graphs were cached by (lane, first row, count) but read the window's
`d_par_` entry by its index in the batch, so two batch shapes with the same first row could share a graph; the
key now includes that index (older than this branch; only multi-slot batches hit it).

Measured (2026-11-25, `tools/multi_agent_bench.py 2 40000 3 300`, both agents' conversations already on disk, so the
numbers are decode and turn overhead, not cold reads):

| | wall | combined tok/s | per agent | GPU busy (scaled to the run) |
| --- | --- | --- | --- | --- |
| `--slots 1` | 58-59 s | 30.3-30.8 | 31-34 tok/s, in turn | ~38% |
| `--slots 2` (two lanes) | 43-45 s | **39.8-42.0** (+31-36%) | 18.5-24 tok/s, at once | ~55-59% |
