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

- 2026-10-19: items 3 (as prompt-lookup drafting; tree verification not done) and 10 done (branch `drafting-prefix`, merged): `docs/LOOKUP-FORK.md`.
