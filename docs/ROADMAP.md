# Performance roadmap

A running list of where bnk's speed comes from next, in the order we plan to work on it. Numbers marked
*estimate* are expectations, not measurements; replace them with results as items land.

## Next steps

| # | What | Why | Expected | Status |
|---|---|---|---|---|
| 1 | ~~Sparse prefill attention~~ | done, see below | | done |
| 2 | ~~Profile the real multi-agent workload~~ | done, see below | | done |
| 3a | ~~Batching feasibility~~ | done, see below: go | | done |
| 3b | ~~Batched decode of several conversations~~ | done, see below | | done |
| 4 | ~~Decode while a prompt is read~~ | done, see below | | done |
| 5 | ~~Batched drafting~~ | done, see below | | done |
| 6 | **More rows per batched forward** (kMaxWindow 8 -> 16: GEMV / MoE kernels) | with 3 conversations each gets ~2.7 rows, so fewer drafts are verified | more tokens per round *(estimate)* | |
| 3c | ~~Sampling on the GPU~~ | done, see below | | done |
| 3d | ~~Drafting cost~~ | done, see below | | done |

## Other areas (from the 78K nsys profile, 2026-10-05)

* **DeltaNet recurrence** (`gdn_rec_k`): ~10% of GPU time while reading a prompt.
* **fp16 GEMMs** (cuBLAS / CUTLASS): ~36% of GPU time while reading a prompt: shapes, kernel picks (wmma vs
  s884), fusing the small ones.
* **Decode is expert-miss bound**: CPU experts run at DRAM bandwidth (~50 GB/s) and the GPU waits on them ~29%
  of the time; a PCIe expert prefetch was measured slower (-21%).
* Older open items: HC fusion, int8 KV, faster `moe_plan`, T=8 GEMV (see 6).
* **First-turn reads dominate the multi-agent benchmark** (121 s of ~275 s in the 3-agent run): the prompt path
  (GEMMs, expert streaming) is now the biggest lever there.

## Done

* **VRAM budget follows the driver** (2026-10-06, v0.1.3): an agent session with more conversations than slots
  (~60 parks/resumes) failed requests with `CUDA VMM create: out of memory`. Cause: forward-graph LRU churn
  (batched layouts vary; 32 kept) refunded each destroyed graph's measured ~18 MiB to the budget, but the driver
  keeps destroyed-graph memory and instantiates the next graph in it (measured ~0): ~1.5 GiB of phantom free
  VRAM, which the expert cache grew into. No refund on destroy; `VramBudget::sync()` re-derives the limit from
  `cudaMemGetInfo` (used + driver free - reserve, below used when under the reserve) on every `make_room` and
  before the cache grows back, reclaiming in up to 4 rounds; `ElasticBuf::ensure` unwinds a partial mapping on
  OOM and retries once. Drift in the same workload: -1,486 MiB -> -134 MiB, no failed requests. Console and
  dashboard show the combined decode rate of all requests in flight (`gen_tps_total`, wall time).
* **Decode while a prompt is read** (2026-10-06, v0.1.2): `bnk serve` runs the other requests' rounds between a
  read's prompt chunks, for `BNK_READ_SHARE` (0.5) of each chunk's time, never after the last chunk (whose logits
  seed the first token). An agent decoding while another agent's 42K-token first turn is read stalled 36.1 s at
  most; now 6.7 s (346 token deliveries during the read), the read itself 36 -> 52 s wall. Closed-loop 3-agent
  throughput unchanged (the GPU is busy either way). Found on the way: the prompt path sized its expert staging
  from the cache's resident count, which misses swaps in flight (old expert unmapped, new one not yet mapped);
  with batching's every-step adaptation that crashed reads ("expert staging too small"). Now sized from the exact
  unmapped count, before every chunk. `tools/serve_check.py`: concurrent serving with interleaved reads == alone.
* **Batched drafting** (2026-10-06, v0.1.2): every conversation's drafter pass of a round in one forward
  (`MtpLayer::run_multi/step_multi`: per-slot drafter KV and sparse attention, shared projections / MoE / head,
  `argmax_prob_rows`). Drafts identical to per-conversation drafting (same forwards, same tokens on the exact
  tier). 3 conversations decoding together: drafting + verify + commit per round 7.99 -> 6.04 ms, 80.8/85.7 ->
  86.9/90.9 tok/s (+6-7%); the closed-loop server benchmark shows no difference beyond noise (rounds there often
  batch fewer conversations).

* **Batched decoding of several conversations** (2026-10-06): the engine holds `--slots` conversations (default 3
  in the server) with their own KV, DeltaNet / PLE state, snapshots and drafter KV; one forward runs every active
  conversation's window (rows concatenated, <= 8): norms, GEMVs, MoE and head once over all rows, attention /
  DeltaNet / PLE per conversation on its own state. `bnk serve` admits up to `slots` requests and decodes them
  together; the rows of a round go first to each pending token, then to drafts round-robin. Exact: each
  conversation's greedy tokens decoded in batched rounds equal its tokens decoded alone (plain and MTP; parking
  checked with 1 and 3 slots). Findings: three conversations mixed in one expert cache miss more (21% vs 14.6%
  one at a time), so with slots > 1 the cache adapts every forward and swaps up to 32 experts (3 conversations:
  72-76 -> 84 tok/s; one conversation: no change); each batched layout is its own CUDA graph (~18 MiB of VRAM:
  slot-ordered layouts, an LRU of 32, charged to the VRAM budget). Measured: engine-level 3 conversations at
  ~40K, 65-68 -> 84 tok/s total (+24%); through the server, 3 agents x 3 turns of up to 1,500 tokens: 31.2 ->
  35.7 tok/s overall (+14%, the same 117 s of first reads in both), ~54 -> ~63 tok/s while decoding (+17%).

* **Drafting and commit overhead** (2026-10-06): an nsys trace grouped by graph launch showed the commit (the
  DeltaNet recurrence replayed over the kept rows, ~0.8 ms of GPU time) hidden inside the drafter's timing, the
  draft argmax on one block (84 us for 144K logits) and 37 small conv-shift launches per commit. Now the commit runs
  on its own stream and overlaps the drafter (joined before anything touches the recurrent state), the argmax
  uses many blocks, the shifts are one kernel, and the drafter no longer waits for the stream before its inputs.
  Drafting + commit per round 2.97 -> 2.46 ms; decode at 78K 61.5 -> 62.3 tok/s (+1.3%, alternating A/B);
  greedy MTP output still identical to plain decoding. `BNK_COMMIT_SAME_STREAM=1`, `BNK_ARGMAX_ONE=1` restore
  the old behaviour.

* **Sampling on the GPU** (2026-10-06): a sampled row cost 3.0 ms of host time (copying 248K logits and an
  nth_element over all of them for top-k 20), ~2 rows per round. `topk_rows` (radix select on order-preserving
  keys, ties to the lowest index, checked against a host reference in `tests/test_topk.cu`) picks each row's
  top-k on the GPU: 0.64 ms for 4 rows, and only k (id, logit) pairs cross PCIe. The host keeps temperature,
  min-p, top-p and the draft acceptance over the k candidates (same code for both paths). 3 agents at ~45-50K:
  time between verify forwards 7.15 -> 3.57 ms (median), decode 54.0 -> 61.9 tok/s (+15%). Used when top_k is
  1..1024 and there are no repetition/presence penalties (those change which tokens are the top k); otherwise,
  or with `BNK_HOST_SAMPLER=1`, the host path as before. Known slow case: top_k 0 on the host (27 ms per row).

* **Batching feasibility** (2026-10-06, `BNK_ROUTE_LOG` + `tools/batch_sim.py`): 3 agents at ~45-50K tokens,
  3,430 verify forwards logged with every row's routing, residency, per-layer GPU timestamps and CPU expert times.
  Per-layer cost model fitted on them (GPU experts R2 0.95, CPU misses R2 0.88) predicts the measured forwards
  within 4.2% (median; bias -2.4%). Merging the 3 conversations' windows round by round: per layer 18.8 resident +
  2.0 missed distinct experts for one conversation become 48.2 + 5.9 for three (the CPU work triples, but runs
  once per step instead of once per conversation). Taking turns 53-54 tok/s total; batch of 2: 69-79 tok/s;
  batch of 3: 75-91 tok/s (the range is per-conversation vs batched drafting). Holding three conversations' KV in
  VRAM costs the expert cache 20-30% of its slots: 74-90 tok/s. Verdict: worth building. Caveats: windows past 8
  rows are extrapolated linearly (the kernels stop at 8 today); the agents were synthetic (repository code).

* **Real multi-agent workload on the parking server** (2026-10-06): 132 requests of the user's coding agent
  (contexts 9-92K): prompt reading 26% of busy time (93% before parking), follow-up turn reads median 1.6 s
  (p90 4.3 s) with 97.8% of each prompt reused, decode median 62 tok/s, 31 requests ran in several time slices,
  no parked conversation dropped. Generation is now the larger share: steps 3a/3b.

* **Sparse prefill attention kernel** (2026-10-05): the attention kernel 88.8 -> 50.1 ms per 8K rows (1.77x,
  `tests/bench_qsa_attn.cu` on real selections); a 78K-token read 63.3 -> 58.8 s (+7.1%, 1,237 -> 1,331 tok/s),
  68.1 -> 63.5 s with MTP. What helped: one launch over the chunk instead of per 64 rows, 8 warps with the output
  kept in accumulator fragments, and loading the next tile's keys/values during the current one. What did not:
  sharing a block between 2-4 neighbouring rows over the union of their selections (the union is 1.9x one row's
  for 4 rows, so loads halve, but the extra masked tensor-core work and lower occupancy made it slower; the
  variant stays in the code, `qsa_attention_prefill_multi(rows=2|4)`, for a later look). `BNK_QSA_PER64=1`
  restores the old per-64-row path.

* **Parked conversations + time slices** (2026-10-05): several agents take turns without re-reading their
  prompts; 3 agents at ~45-50K tokens: turn reads 41 s -> 2.0 s, run 721 s -> 256 s.
* **FlashAttention evaluated** (2026-10-05): upstream needs sm80+; the V100 fork (peisuke, `v100-sm70-support`)
  ties PyTorch SDPA / xFormers in its own report and cannot do QSA's 4-token sparse gather or the output gate. The
  speedup available is in our own kernel (step 1).
