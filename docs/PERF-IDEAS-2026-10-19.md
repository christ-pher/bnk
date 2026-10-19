# Performance inventory and ideas (2026-10-19)

Where the time goes on the two V100s (FP4-Local and IQ3_S through the MoE engine, Qwen3.8-27B through the dense
engine), what bounds each loss, and which items are genuine opportunities. Compiled from the measured docs
(`PERF-2026-10-16.md`, `DUAL-GPU.md`, `NEXT.md`, `PLAN.md`, `DENSE.md`, `TOKS.md`) and the engine's own profiles;
nothing here is a new measurement. The numbered items are referred to by number elsewhere (branches, commits).

## Inventory

1. **Decode forward chain (MoE, one conversation).** A round is ~23 ms: ~11 + 10 ms of GPU time on the two
   cards in turn, plus 1.5 ms of drafting. Each card moves ~2.2 GiB of dense weights and 1.5-3.5 GiB of experts
   per forward, which HBM could deliver in 4-6.5 ms; the rest is 972 dependent launches (~37 per layer) and their
   ramp. Measured dead ends: persistent per-layer kernel (0.77-1.03x), tensor parallel (slower), GEMV fusion and
   stacking (2%), Q8_0 hyper-connection weights (0%). Volta has no programmatic dependent launch. Verdict: ~2x of
   theoretical headroom per card that this silicon does not give back; 5-10% of launch trimming left, at high cost.
2. **The second card idles during single-stream decode.** The pipeline alternates cards, so one conversation
   uses each GPU ~45%. Tensor parallel was the fix and lost (latency-bound halves). Two lanes recover it with 2+
   conversations (+21-28%). Verdict: structural; only multi-stream or speculation (ideas below) uses it.
3. **Tokens per round.** 2.0-2.8 accepted of 4 drafted, and the forward's cost is nearly flat from 1 to 8 rows.
   Every extra accepted token is a free ~40% of a round. Verdict: the largest genuine software lever on decode,
   untouched beyond draft-length tuning.
4. **FP4-Local prompt path: PCIe staging.** 21-25% of experts live outside VRAM; a 4K read copies 20 GB per card
   at the link's 12.9 GB/s, GPUs 64% busy; a short follow-up pays ~0.75 s of copies before compute. Cold 4K 1,337
   tok/s against IQ3_S's 1,990; follow-up 0.94 s against 0.56 s. Every residency lever was measured (buffer
   knobs, chunk pairs, CPU share: all net losses). Verdict: VRAM capacity and PCIe 3.0; residual only by shrinking
   bytes (idea D).
5. **Prompt-path expert kernel (`moe_tc.cu`).** Latency-bound at 8 warps/SM, 40% issue, ~4x under the
   tensor-core bound; two occupancy attempts were slower. ~60% of GPU time in a cold read. Verdict: genuine but
   hard; a successful register diet would be worth 10-20% on cold reads.
6. **One-chunk follow-up reads idle each card ~50%** (stage 0 then stage 1). Splitting into two chunks measured
   slower. Verdict: structural to the pipeline; other conversations' decode already fills the gap.
7. **Dense 27B decode.** 23 ms per forward against a 14.6 ms bandwidth bound (63%); 128 host exchanges ~3 ms,
   head on rank 0 0.8 ms, drafter 1.75 ms per round. Verdict: genuine, 10-15% from fp16 exchanges, a sliced head
   and a smaller draft head; low-medium cost.
8. **Dense 27B prompt path.** 1,330 tok/s is ~36 TFLOPS per card (Q8_0 unpacked to fp16 for `mma.m8n8k4`),
   about half of what a capped V100 sustains; the attention kernel runs the tensor pipe at 25%, the entire gap on
   64K+ reads (250K: 512 s against ~124 s on Flash-Next). Verdict: genuine, 1.3-1.6x on long reads if the
   attention kernel doubles; Volta's mma shape makes it hard.
9. **No parking in the dense engine.** A third conversation re-reads from scratch: 67 s at 64K. Verdict: genuine,
   large when it hits, medium cost (port the MoE engine's snapshot/park machinery).
10. **No shared-prefix reuse across conversations.** Agent fleets share system prompts and tool schemas (5-20K
    tokens); every new conversation reads them again (4-15 s). The multi-agent profile had first-turn reads at
    121 of 275 s. Verdict: genuine, large for agent use, medium cost.
11. **Window of 8 rows with several streams.** 2-3 conversations get ~2.7 rows each, so fewer drafts are
    verified; 8 -> 16 is a 2-3 day change across ~78 sites for 10-20% with 2-3 streams.
12. **CPU experts on FP4-Local decode.** 3.2-3.7 ms of host work per forward, mostly overlapped; 2.5-7% misses.
    Verdict: <= 5%, and shrinks with residency.
13. **Python layer.** Template 1 ms, tokenizer 5-20 ms (toks), ids over the pipe 23 ms at 256K. Nothing left
    over 50 ms.
14. **GLM-5.3-Flash.** 159.5 GiB of experts exceed RAM plus VRAM; 16.9 tok/s is DRAM- and NVMe-bound. Verdict:
    hardware (RAM) only.
15. **The 150 W power cap.** The one compute lever on every tensor-core phase; excluded by decision.

## Ordered by time lost (what a typical turn waits on)

| # | Item | Time lost | Bound by | Genuine? |
|---|---|---|---|---|
| 1 | Decode chain latency (1) | ~5-6 ms of every 11 ms forward: half of all decode time | Volta launch/ramp | exhausted |
| 2 | Idle second card in single-stream decode (2) | the other half of a round | pipeline structure | only via speculation |
| 3 | Dense 27B long-read attention (8) | 256K read 512 s vs 124 s; 64K 69 s vs 27 s | kernel efficiency | yes |
| 4 | FP4-Local staging on reads and follow-ups (4) | ~33% of a cold read, ~0.4 s of every follow-up | VRAM + PCIe | no |
| 5 | Prompt expert kernel occupancy (5) | ~60% of cold-read GPU time at 4x under bound | kernel latency | hard |
| 6 | Re-reads: no parking on the 27B, no shared prefixes (9, 10) | 4-67 s per affected turn, zero otherwise | software | yes |
| 7 | Unaccepted draft positions (3) | ~1.2-2 rows of each 8-row window | drafter quality | yes |
| 8 | Dense 27B decode exchanges and head (7) | ~4 ms of each 23 ms forward | software | yes |
| 9 | Follow-up pipeline bubble (6) | ~0.3 s of a 0.56 s follow-up | pipeline structure | no |
| 10 | Window 8 with several streams (11), CPU experts (12), Python (13) | <= 10% each, situational | software | small |

## Ordered by possible gain (what to expect)

| # | Item | Expected gain | Cost | Notes |
|---|---|---|---|---|
| 1 | Better drafting: tree / multi-candidate verify + prompt-lookup drafts (3) | +15-40% decode on copy-heavy agent output, +5-15% on prose, both engines | medium-high | the forward is flat in rows, so the window is free |
| 2 | Shared-prefix snapshots across conversations (10) | first turns of agents drop from 4-15 s to < 1 s | medium | biggest win for fleets of agents |
| 3 | Parking in the dense engine (9) | removes 67 s re-reads at 64K | medium | port, not design |
| 4 | Dense 27B attention kernel (8) | 1.3-1.6x on 64K+ reads of the 27B | high | Volta mma operand traffic |
| 5 | Dense 27B decode: fp16 exchange, sliced head, smaller draft head (7) | 10-15% decode | low-medium | measured as line items |
| 6 | Window 8 -> 16 (11) | 10-20% with 2-3 streams, 0 alone | medium (2-3 days) | |
| 7 | Prompt expert kernel register diet (5) | 10-20% cold reads if it lands | high | two attempts lost |
| 8 | Entropy-coded expert bytes over PCIe (idea D) | <= 20% of FP4 staging time | high | speculative |
| 9 | Decode launch trimming (1) | <= 5-10% | high | repeatedly measured at 0-2% |
| 10 | FP4 residency, CPU experts, Python (4, 12, 13) | ~0 | | exhausted |

## Ideas (outlandish or not yet designed)

- **A. Tree speculation in the 8-row window.** Verify two candidates at each of four positions instead of one
  chain. Attention needs a tree mask; DeltaNet needs a recurrent state per branch (cheap: a 4-warp recurrence).
  Same forward cost, more accepted tokens.
- **B. Prompt-lookup drafts merged with the MTP's.** Coding agents re-emit file contents; an n-gram match from the
  conversation predicts dozens of tokens at once. Fill the window's rows from both sources.
- **C. Lookahead pipelining for one stream.** While GPU 1 finishes round k, GPU 0 starts round k+1 on the
  assumption that the drafts are accepted; discard on a mismatch. Up to ~1.5x single-stream decode if acceptance
  stays high; the most complex item here.
- **D. Entropy-coded NVFP4 experts in host RAM.** The e2m1 codes are skewed, so a rANS/Huffman stream decoded by a
  GPU kernel before the expert kernel would cut FP4-Local's PCIe bytes by about a fifth.
- **E. Prefetch next-round experts from the drafter's tokens.** The draft tokens' routing is knowable a round
  ahead; stage their likely-miss experts during the current round to cut FP4's CPU misses.
- **F. Hardware.** A third card or 48 GB cards remove item 4 entirely; 192 GB of RAM is the only fix for GLM; the
  150 W cap is the one compute lever on the table.

## Status

- Item 3: prompt-lookup drafting (merged 2026-10-19, `docs/LOOKUP-FORK.md`); draft 7 for the IQ3_S release and the
  27B (v0.4.1). Tree verification (idea A) not done.
- Item 4: FP4-Local's PCIe staging addressed by cold experts in smaller formats (v0.4.1, `docs/EXPERT-MIX.md`):
  reads +45%, expert misses 3-7% to 1-2%, NLL +0.5%.
- Item 10: prefix forking (merged 2026-10-19, `docs/LOOKUP-FORK.md`).
- Item 12: largely moot after the cold experts (CPU experts per forward 9 to 3 ms on code).
- Item 13: the toks tokenizer (merged 2026-10-19, `docs/TOKS.md`).
- Idea B: done (prompt-lookup drafting). Idea C: lookahead pipelining (merged 2026-10-19, `docs/LOOKAHEAD.md`).
- Model-side ideas (same day): 1 done (cold experts); 2 tested as the RCO IQ3_S 27B: same speed as the Q8_0 at
  half the VRAM, its IQ-format GEMVs are unpack-bound; 4 done (draft 7); 3 and 5 open; 6 declined.
- Open: items 1, 2, 5, 6, 7, 8, 9, 11; ideas A, D, E (D and E matter much less at ~95% residency).

## Second pass (2026-10-19, evening): in and out of the box

### The dense 27B is where the gap is

One stream's decode speed is about bandwidth / bytes per forward x tokens accepted per round. On the two V100s:

| 27B at | weight bytes per card | floor per forward at 889 GB/s | plain ceiling | with MTP: prose / code / copy |
|---|---|---|---|---|
| Q8_0 (what runs) | ~13 GB | 15.7 ms | ~64 tok/s | ~140 / 220 / 400 |
| a 4-bit format | ~7.5 GB | 8.4 ms | ~110 tok/s | ~250 / 400 / 700 |
| **measured, Q8_0** | | **23.6 ms** | **42 tok/s** | 80 / 120 / 200 |

So the 27B runs 1.5x off its own Q8 ceiling and about 2.6x off a 4-bit one; published "hundreds of tok/s on a
27B" numbers (4-bit, EAGLE/MTP drafting on code-heavy output, or batched throughput) are physically within reach
here. The MoE models are chain-bound, not bandwidth-bound: they reach hundreds only summed over several streams.
The RCO IQ3_S 27B is the clue: half the bytes gave the same speed, so the IQ-format GEMVs are bound by unpacking
their codes. Fewer bits pay only in a format that is cheap to unpack.

### In the box: efficiency

Dense 27B:

- **D1. Profile one decode forward** (nsys): split the 23.6 ms into GEMVs, the 128 cross-GPU exchanges (~3 ms)
  and the rest. Half a day; it orders everything below.
- **D2. A cheap-to-unpack 4-bit path.** Repack a 4-bit GGUF (unsloth Q4_K_M / UD-Q4_K_XL, values kept exactly) at
  load into an interleaved layout where one 128-bit load is a whole mma fragment, with the int4-to-fp16 bit-trick
  conversion ExLlama and Marlin use (architecture-agnostic). First a cheap test: a Q4_0 / Q4_K_M file through
  today's kernels. If the layout reaches ~80% of bandwidth: plain 42 to ~75 tok/s, MTP prose ~80 to ~170.
- **D3. Cheaper exchanges:** fp16 in decode windows (item 7), and each exchange's host hop overlapped with the next
  layer's weight prefetch. 1.5-2 ms per forward.
- **D4. Wider windows and tree drafting on the dense engine.** A bandwidth-bound forward costs nearly the same for
  16-32 rows as for 8 (5 vs 8 rows measured 29.6 vs 32.5 ms); tree verification fits here better than on the MoE
  engine, where each extra row pulls in experts. Prose from ~2.3 toward ~3.5 tokens per round.
- **D5. More slots for agent fleets.** Weight reads amortize over conversations: 8 streams at 16-32 rows could sum
  to 400-600 tok/s; today capped by 2 slots and the 8-row window.

MoE models:

- **M1. The 16-row window plus more slots** (item 11): the multi-stream lever, now that FP4-Local is ~95% resident.
- **M2. Drop the cold experts' NVFP4 copies from pinned RAM** (~30 GB never read).
- **M3. Trim the forward-graph cache:** up to 48 x ~18 MiB charged to the VRAM budget, up to 0.8 GiB per card.

### In the box: code quality

- **Q1. The nondeterminism under `BNK_EXACT=1`:** two plain runs of the essay prompt diverge at token 175 with exact
  kernels and a static cache. Every A/B depends on exact mode being exact. Suspects: an atomic float reduction, or
  CPU-expert ordering when misses vary.
- **Q2. One conversation-state layer for both engines.** Slots, snapshots, parking, forking and graph caching exist
  twice; the dense copy has no parking or forking. Extracting them makes item 9 a port and gives the dense engine
  forking for free.
- **Q3. An in-repo A/B harness** (`tools/ab.py`): plain against variant, greedy and sampled, NLL / KL, the exactness
  checks. Six throwaway scripts were written for this on 2026-10-19 alone.
- **Q4. Benchmark under the configs' sampling** (temperature 1.0), not only greedy: every number this week was
  greedy, and drafts are accepted less often when sampled.
- **Q5. Prune the `BNK_*` switches:** a registry printed at start; delete the bisection-only ones
  (`BNK_LA_SEQ_SVC`, `BNK_MTP_OWN_STREAM`, ...).
- **Q6. Lookahead's complexity against its payoff:** six bugs to land, pays on copy-heavy output only. The most
  fragile code in the engine; keep it under the A/B harness.

### Out of the box

- **X1. Out of the VM, or onto NVLink.** The machine is a Proxmox guest (the root filesystem is VM 108's disk).
  Peer-to-peer copies take 49-51 ms instead of microseconds, and `cudaHostRegister` is 4x slower inside the engine
  than standalone: both point at the virtualization layer. Bare metal, or passthrough with IOMMU / ACS set up for
  peer access, could make the cross-GPU exchange ~5x cheaper: an immediate gain for the dense engine, and tensor
  parallel reopens for the MoE models. Further: dual SXM2-to-PCIe carrier boards with an NVLink bridge exist in the
  homelab market; NVLink 2.0 between two V100s (~150 GB/s, a few microseconds) changes every multi-GPU tradeoff
  measured so far. First step, no code: check the GPUs' IOMMU groups and ACS in Proxmox.
- **X2. Drafting across conversations** ("suffix decoding"): one shared index over all recent conversations'
  outputs instead of the current conversation's only. Agents repeat tool-call scaffolding and re-read files other
  agents read. Built on the existing lookup index.
- **X3. Long-context state on disk.** A 256K cold read costs 124 s (FP4-Local) or 512 s (27B); a parked conversation
  is 3-4 GB, 1-2 s from NVMe. Saving parked states and shared-prefix snapshots to disk lets a long context survive
  server restarts and serves the same repo dump to a new agent days later without a re-read.
- **X4. Fewer thinking tokens.** For time to answer, reasoning tokens often dominate: a per-client reasoning-effort
  budget beside the loop guard cuts wall-clock time without touching a kernel.
- **X5. A dense megakernel, last.** Recent work beats CUDA graphs at small batch with single-launch decode kernels
  using per-layer dependency counters, not the grid barriers the 2026-10-15 attempt used (the reason it lost).
  Plausible for the 27B's uniform layers; high risk.

### Order

1. D1 (the 27B profile) and the cheap D2 test (a Q4 file through today's kernels): together they decide whether
   the 4-bit layout is a week of work for ~2x on the 27B.
2. In parallel, without code: X1's IOMMU / ACS check. If peer access can be fixed in configuration, it is the
   cheapest multi-GPU win on the list.
3. Q1 and Q3 before the next round of kernel work: exact mode and the harness make every later change faster to
   judge.
4. Then, by measured value: D2, D3, D4 / D5 and M1 (both are the window and slot work), item 9 via Q2, X2, X3.
