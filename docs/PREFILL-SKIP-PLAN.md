# Long prompts without the second half's work: the plan (branch `prefill-skip`)

After the 2026-10-26 Project Maya review (v1.1.29: #318 the exact long-prompt skip, PR #524 the Engram read-ahead).
The two prompt regimes are in PREFILL-PLAN.md. This plan is about the second one, prompts of ~3.7K new tokens or more,
which are bound by the kernels: 8K 652 tok/s (stages 6.3 + 6.2 s in sequence), 16K 828 (two groups pipelined),
70K 662. None of the four items helps the agent's per-turn suffix (250-800 tokens, link-bound) or cold 2K prompts.

## Why the skip is exact on V4.1

From the GGUF (`deepseek41.*`): `compress_ratios` 0,0, 2 x 18, then **1 for layers 20-39**; `kv_source_layer_ids`
[2, 8, 14, 20]; `index_source_layer_ids` [2, 8, 14, 20, 24, 28, 32, 36]; `candidate_source_layer_id` 20; Engram
layers 1 and 14; sliding window 128.

* Layers 21-39 read earlier positions in only two ways. The first is layer 20's compressed KV and index keys:
  `kv_src` = 20, and the index-key weights exist only on kv-source layers. Layers 24-36 rank with their own index
  queries, but their keys are layer 20's `ikc`. The second is their own 128-position window.
* So a row's output at layer `l >= 20` depends on rows earlier than itself only through layer 20's compressed cache
  (per position, from layer 20's input alone) and through the windows: one window more per layer, (40 - 20) x 127 + 128
  = **2,668 rows**.
* Rows earlier than the last 2,668 therefore need layers 0-19 (everything before 20 feeds 20's cache) plus layer 20's
  **kv-only** part: hyper-connection mix and pre, attention norm, `comp_kv` and `ds_compress` (compressed KV and index
  keys). Nothing of their attention, experts or layers 21-39. The last 2,668 rows run every layer, so the logits, the
  rings decode reads, and the DSpark capture (the last 136 positions) come out as the full work gives them.
  Maya measured KL 1.5e-4 on the last row's logits (the experts' summing order) and the same greedy tokens.
* Checkpoints are taken between `prefill()` calls (generate.cpp: prefill to the checkpoint, checkpoint, prefill the
  rest). Each call ends exact, so every checkpoint is exact. Rows inside a call are never snapshotted.

Our split is layers 0-19 on GPU 0 and 20-39 on GPU 1, so on a long prompt **GPU 1's work almost vanishes**.

## Item 1: the exact skip (BNK_DS_SKIP, default on once it passes)

**Design: the cut is a group boundary.** `prefill()` cuts the new tokens at `a = n - keep` (keep = 2,668) when
`a >= 1,024` (BNK_DS_SKIP_MIN). The rows `[0, a)` go in even groups marked *skip*, and `[a, n)` as now. No buffer
rebasing inside a group is needed.

* A skip group runs normally on every stage before the skip layer `yl` (20). On the stage that owns `yl`, layers
  `l0_ .. yl-1` run fully. At `yl` it runs the kv-only part chunk by chunk and stops: no experts, no head, no hand-off.
  Stages after `yl`'s only append the tokens to their history.
* Conditions, checked at load (`skip_layer_`, -1 = off): `yl` is the first ratio-1 layer and a kv source, every later
  layer has ratio 1, reads `kv_src == yl` and is not a kv source, the Engram layers come before `yl`, and the stage
  owning `yl` is the last stage (it owns everything after it).
* The window rings of layers `yl..39` are zeroed for the skipped positions. That isn't needed for exactness (those
  slots only feed rows inside the cone), but it keeps stale or uninitialized bits out of the cone.
* Cost: the cut adds one group boundary on stage 0, so each non-resident expert streams once more on GPU 0 for the
  suffix group. Stage 1's kv-only work overlaps stage 0's next group.

**Expected** (from the stage profiles: ~0.77 ms a row a stage at 8K):

| prompt | now | with the skip (est.) |
|---|---:|---:|
| 4K | ~470 | ~520 (the cut is ~1.3K rows: little to skip) |
| 8K | 652 (12.5 s) | ~850-950 (stage 0 ~6.6 s + suffix 2.1 s) |
| 16K | 828 | ~1,100 |
| 70K | 662 | ~1,200-1,300 (stage 0 bound, stage 1 nearly idle) |

**Verification:**
1. The copy test bar: ppl 1.037, top-1 99.61% (needs the copy prompt > 3.7K tokens to exercise the skip, else a long
   variant).
2. A/B on a 16K and a 40K prompt with `BNK_DS_SKIP=0/1`: the last row's logits (KL, top-1) and greedy continuations
   of 128 tokens, which should be identical or diverge only late.
3. A follow-up turn after a skipped prompt (decode reads the rings and the compressed cache), plus a rollback to a
   checkpoint taken between two prefill calls.
4. Speed: `bnk pbench` cold 2K / 8K / 16K plus a 40K and a 70K prompt, `BNK_DS_PF_PROF=1` for per-stage times.

## Item 2: give GPU 1 part of the first half (after item 1)

With the skip on, a long prompt runs ~0.3 s on GPU 1 for every ~6 s on GPU 0. Moving the split earlier puts layers
`s..19` on GPU 1 for **all** rows and balances the two stages for long prompts. This is a pure placement change:
`--split s` already exists.

* Model: stage 0 = s layers x all rows; stage 1 = (20 - s) layers x all rows + kv-only + 20 layers x 2,668. For 70K
  pipelined, the balance point is near s = 10-11. That gives **~1.6-1.8x on top of item 1** in theory; less in
  practice, because the earlier layers' experts move to GPU 1's cache and VRAM budget.
* Risks: **decode** (each GPU's expert cache covers different layers, and dense weights move: ~6 GB over the two
  cards), the **agent suffix** (link-bound per layer, probably unchanged), and **8K** (the suffix group gets heavier on
  GPU 1).
* Steps: measure `--split 15` and `--split 10` on long prompts, 8K, the agent suffix bench and decode (the DSpark
  agent bench). If decode or the suffix lose, a **prompt-only split** is the fallback: GPU 1 holds a second copy of
  layers s..19's dense weights (~1.5 GB at s=15) and runs them during prompts only. That is a bigger change, decided
  after the numbers.

**Expected:** 70K ~1,300 -> ~1,600-1,900 if decode holds. Decision point: keep it only if decode and the suffix are
within noise.

## Item 3: DeepSeek's 128-row replay (BNK_DS_SKIP_KEEP, opt-in)

The same mechanism with keep = 128 (DeepSeek's serving replays only the last window through layers 20-39). It is
**not exact**: rows within 2,540 positions of the cut see some garbage windows on layers 21-39, and the rings hold
them. Hours of work once item 1 is in (the keep is a parameter).

* Expected: 8K ~950 -> ~1,150; 4K gains most (the cut moves from 1.3K to 3.9K rows); 70K little.
* Gate: KL and greedy match on the A/B set; ship opt-in only, documented as changing answers.

## Item 4: Engram read-ahead (measure first)

Maya's PR #524 took a Windows box from 300 to 1,304 tok/s at 64K: its random 4K reads were capped at ~13K/s. Our
gather already runs ahead of its group on 32 threads through the mmap (`pf_engram_gather`), and at 8K it was 0.55 s
exposed on stage 0 (-3% since pass 2's reorder).

* Step 0 (10 minutes, GPUs): `BNK_DS_PF_PROF=1` on a cold 40K / 70K prompt; read `engram host` and `wait` per group,
  plus the page-cache state of the 110 GB file (`fincore`, if present).
* If `wait` is > 5% of stage 0: read the whole group's rows early (the next group's during this one: it runs on
  the host), sorted by offset and coalesced into ~1 MiB ranges with `pread`/`madvise(WILLNEED)` on the sorted pages.
  The GPU-facing layout stays the same.
* Expected: <= 4-8%, and only on stage 0. With item 2 that is the critical stage, so the item's value rises after it.

## Results, session 1 (2026-10-27, ds-xs, both GPUs, no drafter, cold prompts from `bnk pbench`)

**Item 1, quality.** The copy test, with its prefix read through the skip (the copy itself is scored in decode
windows, so it also checks decode after a skipped prompt):

| | skip off | skip on | KL to off (mean / max), same top-1 |
|---|---|---|---|
| 4,096-token prefix (1,428 rows skipped) | ppl 1.0365, top-1 99.61% | ppl 1.0366, 99.61% | 0.00069 / 0.068, 99.80% |
| 8,192-token prefix (5,524 skipped) | ppl 1.0058, 99.80% | ppl 1.0046, 99.80% | 0.00023 / 0.085, 100% |

The last row's logits after a 16K prompt: KL 2.0e-6, same top-1. After a 40K prompt: KL 8.6e-4, same top-1. Both
are within what the expert placement and the CPU/link split change from run to run.

**Item 1, speed** (cold, tok/s):

| prompt | skip off | skip on | |
|---|---:|---:|---|
| 4,096 | 690 | 659 | -5%: 1,428 rows skipped don't pay for stage 0's extra group boundary |
| 8,192 | 718 | **973** | +35% |
| 16,000 | 858 | **1,070** | +25% |
| 40,000 | 815 | **1,090** | +34% |
| 70,000 | 656 | **1,001** | +53% |

Hence BNK_DS_SKIP_MIN defaults to 2,048 skipped rows (prompts of 4,716 tokens or more). The break-even point lies
between 1,428 and 5,524 rows and isn't measured more finely.

**Profile, 70K with the skip** (BNK_DS_PF_PROF=1, 73 s): stage 1 spends **~60 ms per skipped 7.5K group** against
5.4-8.3 s on stage 0. Stage 0's expert phase is ~2.6 s per group (~34 GB over PCIe at 80.7 us/MB); its attention
grows with the context, 2.6 -> 5.6 s per group. The tail (2,668 rows) costs 3.5 s on stage 0, then 8.6 s on stage 1
with nothing beside it. **GPU 0 now sets the pace alone: item 2 is the lever.**

**Item 4, step 0:** stage 0's Engram wait per 7.5K group is 27-468 ms, **~2.2 s of 73 s (3%)**. Not worth a
read-ahead before item 2, and even after it at most ~3-5% of stage 0.

**Item 3, keep = 128** (BNK_DS_SKIP_KEEP=128): copy test ppl 1.0371 (4K) / 1.0068 (8K), top-1 99.61% / 99.80%, KL to
skip off 1e-5 / 2e-5 (the copy test doesn't see the difference: its rows attend to the prefix through layer 20's
exact cache). Speed: **4K 1,112, 8K 1,332, 16K 1,350 tok/s**. That is a large gain at 4-16K, but it isn't exact; keep
it opt-in until a long-range quality check (e.g. retrieval from deep in a long prompt) shows no loss.

**The retrieval check (2026-10-27).** This is the check keep = 128 needs, in the zone it affects. There were 25
prompts of ~10K tokens of this repo's docs and sources, a different stretch each (nothing cached), served by
`run.sh ds-xs`. Each holds six access codes ("the access code of orion-45 is 414108") at ~150, 500, 1,000, 1,600 and
2,300 tokens from the end, plus a control at ~6,000. The question asks for all six; answers are greedy, thinking off.
Script: tools/retrieval_test.py (gen / run / cmp).

| | right | by distance (each of the six) | answers identical to skip off | s per prompt |
|---|---|---|---|---:|
| skip off | 144/150 | 24/25 | (reference) | 13.0 |
| exact skip | 144/150 | 24/25 | 24/25 | 11.6 |
| keep = 128 | 144/150 | 24/25 | **25/25** | **9.7** |

The six misses are one prompt in all three configurations: the model answered as if asked to explore the
repository ("I'll start by reading the files...") instead of listing the codes. keep = 128 lost nothing here: every
code right at every distance, and every answer identical to the full computation's.

**keep = 128 is the default** (the user's decision after the retrieval check; BNK_DS_SKIP_KEEP=0 restores the exact
2,668). Mid-size reads, keep = 128 against skip off, with the DSpark drafter (`bnk pbench`, follow-ups after a
cached 16K prefix, the agent's big tool outputs):

| read | skip off | keep = 128 |
|---|---:|---:|
| cold 2.5K | 4.05 s | **2.83 s** |
| cold 3K | 7.94 s (an outlier) | **2.94 s** |
| cold 8K | 11.5 s | **6.2 s** |
| cold 16K (4 runs) | 18.3-18.5 s | **11.8-13.0 s** |
| follow-up +2.5K | 5.11 s | **3.06 s** |
| follow-up +3K | 5.88 s | **3.25 s** |
| follow-up +4K | 7.56 s | **3.97 s** |
| follow-up +800 (no skip) | 2.10 s | 2.10 s |

Decode after them: 29.0 / 32.6 / 31.7 / 38.4 tok/s with keep = 128, 32.7 / 33.7 / 31.6 / 43.1 off. The largest gap
is after the 800-token follow-up, which reads the same way in both runs, so it is the drafter's acceptance varying
from run to run.

**Item 2, the split sweep (2026-10-27, keep = 128, DSpark; `--split` must sit on a KV source: 8, 14 or 20).**

| | split 20 (default) | split 14 | split 8 |
|---|---:|---:|---:|
| cold 8K | 1,293 | 1,271 | 1,308 tok/s |
| cold 16K | 1,345 | 1,600 | **1,686** |
| cold 70K | 1,015 | 1,427 | **1,563** |
| follow-up +250 (cached 16K) | **0.99 s** | 1.10 s | 1.15 s |
| follow-up +800 | **2.14 s** | 2.30 s | 2.51 s |
| verify ms per drafting round (same turn) | **108** | 117 | 137 |

Moving the split earlier buys up to +54% at 70K, but every agent follow-up is 8-17% slower and a decode round 8-27%
dearer. **Split 20 stays.** The long-prompt gain without the decode cost needs the plan's fallback: a prompt-only copy
of layers s..19's dense weights on GPU 1, used while reading prompts and not for decode. That is a bigger change,
not started.

On the way, `--split 14 / 8` crashed on long prompts ("router GEMM failed at layer 14"). It was this branch's bug: a
skipped group's guessed prefetch was bounded by the skip layer, not by the stage. Fixed; master's `--split` was fine.

**Decode after a skipped read (2026-10-27).** keep = 128 against skip off, DSpark, split 20. Six 8K prompts are read
cold, then get a 200-token turn and 512 greedy tokens, all twice (`decode_ab.sh`, BNK_PB_TOKENS=1):

| | keep = 128 | skip off |
|---|---:|---:|
| mean decode | 53.4 tok/s | 54.5 tok/s |
| mean tokens per round | 5.30 | 5.40 |
| mean verify ms per round | **87.0** | **86.7** |
| outputs identical across the two variants | 6 of 12 turns | |
| outputs identical between the two passes of the same variant | 2 of 6 prompts | 4 of 6 prompts |

The cost of a decode round is the same (87.0 vs 86.7 ms). Where the outputs match, the speeds match within 1-3%.
The 2% gap in the mean comes from turns whose text differs (one keep = 128 turn wrote a different answer from its
3rd token, at 29.5 tok/s), so it reflects the drafter's acceptance on different text. Skip off itself varies like
this between two passes over the same prompts: it diverged on 2 of 6, against 4 of 6 for keep = 128 on this small
sample. **No decode loss found.**

## Later: the split's long-prompt gain without its decode cost (item 2b, not started)

**Goal.** Split 8 read 16K at 1,686 and 70K at 1,563 tok/s (split 20: 1,345 / 1,015). But it made every agent
follow-up 8-17% slower and every decode round 8-27% more expensive, so split 20 stays. Item 2b keeps split 20 for
decode and short reads, and lets GPU 1 run layers 8-19 **only for a long prompt's skipped groups**. Those are the
groups where GPU 1 now idles: ~60 ms each against GPU 0's 5-8 s.

**Why decode stays as it is:** decode, its expert caches, the exchange and the drafter keep split 20's layout and
code. The new path runs only inside `prefill()` for groups marked skip. Reads too short to skip (the agent's
250-800-token turns) never touch it.

**Design** (s = 8 first, then 14 for comparison; both are KV-source boundaries):
1. **Prompt-only weights on GPU 1:** a second copy of layers s..19's dense weights (attention, shared expert,
   hyper-connection mixers, compressor/indexer, layer 14's Engram projection, routers). That is **~1.6 GiB at s = 8**
   (layers 0-19 are 2.68 GiB, 0-7 1.09 GiB), taken from GPU 1's expert-cache budget at load: ~170 fewer resident
   experts of 2,108 (~8%). Measure what that costs decode before anything else; if it's too much, s = 14 needs ~0.8 GiB.
2. **Per-layer state that decode reads on GPU 0:** for layers s..19 the skipped groups produce compressed KV, index
   keys, the compressor's open-group state (layers 8 and 14, ratio 2), window rings and the Engram path's nothing
   (stateless). GPU 1 computes them into its own buffers and copies them to GPU 0 by peer copy (P2P works since
   iommu=pt, memory p2p-iommu-pt; push, not pull). At 70K that's ~35K compressed rows x (512 + index dim) halves a
   source layer, ~40 MB: negligible. Rings and the open group must be the state at the group's end, so copy after
   the last skipped group, before the tail group runs on GPU 0.
3. **Experts of layers s..19 on GPU 1:** the non-resident ones stream from host RAM over **GPU 1's own link**, so both
   links work at once. That is the main source of the gain: GPU 0's expert phase was ~2.6 s a group, link-bound. The
   experts resident on GPU 0 have **no host copy**. Options, measure in this order:
   a. peer-copy them GPU 0 -> GPU 1 per layer (~116 resident a layer x 9.4 MB ≈ 1.1 GB, ~0.1 s at P2P rates,
      overlapped with the attention phase like the existing prefetch);
   b. send the group's rows for those experts to GPU 0 and back (a row exchange each layer: latency, and GPU 0 busy);
   c. keep host copies of GPU 0's residents for layers s..19 (RAM cost ~13 GB at s = 8; RAM is 192 GB, ~100 used by
      the expert store).
4. **Pipeline:** a skipped group runs 0..s-1 on GPU 0, hands the residual stream over at layer s (the existing xfer
   path, at a different layer), then s..19 plus layer 20's kv-only on GPU 1. GPU 0 takes the next group meanwhile.
   The tail group (keep = 128 rows) runs on split 20 as now, after the state copy in step 2.
5. **Engram layer 14** lands on GPU 1 for skipped groups: its row gather runs on GPU 1's stage (the existing per-stage
   path, verified working in the split sweep).

**Expected:** close to split 8's long-prompt numbers (16K ~1,650, 70K ~1,500), minus the state copies and the
resident-expert handling; follow-ups and decode unchanged from split 20. Stop if 70K gains less than ~25% over
1,015.

**Verification:**
- **Exactness:** copy test and tools/retrieval_test.py, against split 20 with the same keep. GPU 1's copy of layers
  s..19 must give the same results as GPU 0's; differences can come only from summation order.
- **Decode and follow-ups:** decode_ab.sh with tools/decode_cmp.py (verify ms per round must match split 20), plus
  the 250 / 800 follow-ups.
- **Speed:** cold 8K / 16K / 70K.
- **Risks:**
  - the VRAM taken from GPU 1's cache (step 1);
  - the staging buffers GPU 1 needs for layers s..19's experts;
  - prefix forks and checkpoints: the state copy must finish before any checkpoint, and checkpoints are only taken
    between prefill calls, so this is safe as long as step 2 completes inside prefill().

**Effort:** a few days. Steps 1-2 and 4 are plumbing on existing pieces (stage loading, xfer, P2P push); step 3 is
the open design question.

## Order

1. Item 1: code (no GPUs) -> GPUs: correctness A/B, then speed.
2. Item 4 step 0 in the same GPU session (it's only a profile).
3. Item 3 (a parameter of item 1), A/B in the same session.
4. Item 2: the `--split` sweep in a second GPU session (it reloads the model per split, ~3-5 min each).
