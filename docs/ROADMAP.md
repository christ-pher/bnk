# Performance roadmap

A running list of where bnk's speed comes from next, in the order we plan to work on it. Numbers marked
*estimate* are expectations, not measurements; replace them with results as items land.

## Next steps

| # | What | Why | Expected | Status |
|---|---|---|---|---|
| 1 | ~~Sparse prefill attention~~ | done, see below | | done |
| 2 | **Profile the real multi-agent workload** on the parking server | after parking (turn reads 41 s -> 2 s in `tools/multi_agent_bench.py`), measure the read vs generate split on real agent traffic | decides how much effort goes into 3 | |
| 3 | **Batched decode of several conversations** ("level 3"), in stages: several conversations resident at once -> batched kernels with per-conversation positions -> scheduler | agents decode in the same step instead of taking turns | +20-50% total throughput with several agents *(estimate)*; each single stream gets slower | |

## Other areas (from the 78K nsys profile, 2026-10-05)

* **DeltaNet recurrence** (`gdn_rec_k`): ~10% of GPU time while reading a prompt.
* **fp16 GEMMs** (cuBLAS / CUTLASS): ~36% of GPU time while reading a prompt: shapes, kernel picks (wmma vs
  s884), fusing the small ones.
* **Decode is expert-miss bound**: CPU experts run at DRAM bandwidth (~50 GB/s) and the GPU waits on them ~29%
  of the time; a PCIe expert prefetch was measured slower (-21%).
* Older open items: HC fusion, int8 KV, GPU sampler, faster `moe_plan`, T=8 GEMV.

## Done

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
