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

## Order

1. Item 1: code (no GPUs) -> GPUs: correctness A/B, then speed.
2. Item 4 step 0 in the same GPU session (it's only a profile).
3. Item 3 (a parameter of item 1), A/B in the same session.
4. Item 2: the `--split` sweep in a second GPU session (it reloads the model per split, ~3-5 min each).
