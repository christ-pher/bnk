# Performance roadmap

A running list of where bnk's speed comes from next, in the order we plan to work on it. Numbers marked
*estimate* are expectations, not measurements; replace them with results as items land.

## Next steps

| # | What | Why | Expected | Status |
|---|---|---|---|---|
| 1 | ~~Sparse prefill attention~~ | done, see below | | done |
| 2 | ~~Profile the real multi-agent workload~~ | done, see below | | done |
| 3a | ~~Batching feasibility~~ | done, see below: go | | done |
| 3b | **Batched decode of several conversations** ("level 3"), in stages: several conversations resident at once -> batched kernels with per-conversation positions (windows past 8 rows) -> scheduler, with the MTP drafting batched too | generation is ~74% of busy time with several agents | 3 agents: +35-40% total throughput with per-conversation drafting, +63-66% with batched drafting (simulated, cache shrink included); each conversation at ~25-30 tok/s while batched | next |
| 3c | ~~Sampling on the GPU~~ | done, see below | | done |
| 3d | ~~Drafting cost~~ | done, see below | | done |

## Other areas (from the 78K nsys profile, 2026-10-05)

* **DeltaNet recurrence** (`gdn_rec_k`): ~10% of GPU time while reading a prompt.
* **fp16 GEMMs** (cuBLAS / CUTLASS): ~36% of GPU time while reading a prompt: shapes, kernel picks (wmma vs
  s884), fusing the small ones.
* **Decode is expert-miss bound**: CPU experts run at DRAM bandwidth (~50 GB/s) and the GPU waits on them ~29%
  of the time; a PCIe expert prefetch was measured slower (-21%).
* Older open items: HC fusion, int8 KV, GPU sampler, faster `moe_plan`, T=8 GEMV.

## Done

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
